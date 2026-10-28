"""Gate / receptive-field analysis of a trained model.

Captures gate activations with a forward hook and reports four tables:

  Table 1  per hop: n, mean gate, std, p10, p90, mean coordination
  Table 2  interior vs low-coordination atoms within the same hop shell
  Table 3  hop x element (cells with fewer than 20 atoms are skipped)
  Table 4  residual hop effect after de-meaning over (element, coordination bin)

Table 4 is the decisive one: it separates a genuine distance selectivity from a
gate that merely reproduces the hand-written hop prior.

Usage
-----
    python scripts/analyze_gate.py                    # default checkpoint and data
    python scripts/analyze_gate.py --ckpt <path> --split test --max-graphs 1500
    python scripts/analyze_gate.py --save-json rf.json
"""

from __future__ import annotations

import _path  # noqa: F401 - makes models/ importable
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# reuse the data and split logic of the training script (no import side effects)
import train as T
from adaptive_dimenet import AdaptiveDimeNetPP

SPECIES = {
    1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
    28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 49: "In", 50: "Sn", 51: "Sb",
    52: "Te", 53: "I", 55: "Cs", 56: "Ba", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 0: "X", -1: "other",
}
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
DEFAULT_CKPT = os.getenv(
    "AA_CKPT", os.path.join(_ROOT, "runs", "full", "best.pth"))


# ---------------------------------------------------------------------
# ---------------------------- model / data ----------------------------
# ---------------------------------------------------------------------
def load_model(ckpt, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    mk = dict(st.get("model_kwargs") or {})
    flags = dict(st.get("ablation_flags") or {})
    y_mean = st.get("y_mean")
    n_tasks = int(y_mean.numel()) if y_mean is not None else 1
    model = AdaptiveDimeNetPP(
        num_tasks=n_tasks,
        task_weights=[1.0] * n_tasks,
        **mk, **flags,
    ).to(device)
    model.load_state_dict(st["model"])
    if y_mean is not None and st.get("y_std") is not None:
        model.set_target_stats(y_mean.cpu(), st["y_std"].cpu())
    model.eval()
    print(f"[Ckpt] {os.path.basename(ckpt)} | tasks={n_tasks} | flags={flags}")
    print(f"[Ckpt] best_val={st.get('best_val_metric')} | "
          f"test={st.get('test_e_mae')}")
    return model, st


def build_dataset(split, split_mode):
    shard_files = sorted(glob.glob(os.path.join(T.DATA_DIR, "shard_*.pkl")))
    if not shard_files:
        raise SystemExit(f"[FATAL] no shard_*.pkl under {T.DATA_DIR}")
    split_path, stats = T.ensure_split_index(
        T.DATA_DIR, shard_files, split_mode, T.TASK_NAMES, T.FULL_TASK_LIST,
        T.VAL_RATIO, T.TEST_RATIO, T.SEED)
    ds = T.SplitAwareShardDataset(shard_files, split_path, split,
                                  T.TASK_NAMES, T.FULL_TASK_LIST,
                                  shuffle=False)
    return ds


# ---------------------------------------------------------------------
# ---------------------------- capture gate ----------------------------
# ---------------------------------------------------------------------
def collect(model, ds, device, max_graphs):
    if getattr(model, "rf_gate", None) is None:
        raise SystemExit("[FATAL] this checkpoint has no rf_gate (gate disabled)")
    box = {}

    def hook(_module, _inp, out):
        box["gate"] = out.detach()

    h = model.rf_gate.register_forward_hook(hook)
    hops_l, gate_l, coord_l, z_l = [], [], [], []
    n_seen = 0
    try:
        for batch in tqdm(ds, desc="collecting gate", leave=False, ncols=70):
            batch = batch.to(device)
            with torch.no_grad():
                model(batch)
            g = box["gate"].mean(dim=-1)              # (N,) per-atom gate
            hop = batch.hops_to_adsorbate.detach()
            z = batch.z.detach()
            n_nodes = int(batch.num_nodes)
            coord = torch.bincount(batch.edge_index[0], minlength=n_nodes)
            hops_l.append(hop.cpu().numpy())
            gate_l.append(g.cpu().numpy())
            coord_l.append(coord.cpu().numpy())
            z_l.append(z.cpu().numpy())
            n_seen += int(batch.num_graphs)
            if max_graphs and n_seen >= max_graphs:
                break
    finally:
        h.remove()
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return (cat(hops_l).astype(int), cat(gate_l),
            cat(coord_l).astype(int), cat(z_l).astype(int))


# ---------------------------------------------------------------------
# ----------------------------- analysis -----------------------------
# ---------------------------------------------------------------------
def table_hop(h, g, c):
    print("\n" + "=" * 90)
    print("Table 1  per-hop overview")
    print("=" * 90)
    q25 = float(np.quantile(c, 0.25))
    print(f"  (low-coordination threshold = global 25th percentile = "
          f"coord <= {q25:.1f})")
    print(f"{'hop':>4}{'n':>9}{'mean_gate':>11}{'std':>8}{'p10':>9}{'p90':>9}"
          f"{'mean_coord':>12}{'%edge':>8}")
    for hp in sorted(set(h.tolist())):
        m = h == hp
        print(f"{hp:>4}{int(m.sum()):>9}{g[m].mean():>11.4f}{g[m].std():>8.3f}"
              f"{np.percentile(g[m], 10):>9.4f}{np.percentile(g[m], 90):>9.4f}"
              f"{c[m].mean():>12.1f}{(c[m] <= q25).mean():>8.1%}")
    return q25


def gate_summary(h, g, q25):
    """Effective radius at a 0.5 threshold, half-decay hop and retention band."""
    hops = sorted(set(h.tolist()))
    means = {hp: float(g[h == hp].mean()) for hp in hops}
    g0 = means[hops[0]]
    gmin = min(means.values())
    r_eff = next((hp for hp in hops if means[hp] < 0.5), hops[-1])
    rng_ = g0 - gmin
    half = None
    if rng_ >= 0.02:
        thr = g0 - 0.5 * rng_
        half = next((hp for hp in hops if means[hp] <= thr), hops[-1])
    print("\n" + "=" * 90)
    print("gate summary")
    print("=" * 90)
    print("  per-hop mean: " + "  ".join(f"{hp}:{means[hp]:.4f}" for hp in hops))
    print(f"  R_eff (first < 0.5)   = {r_eff} hops")
    print(f"  half-decay hop        = {half if half is not None else 'undefined'}")
    print(f"  gate range g(0)-min   = {rng_:.4f}")
    print(f"  multiplier 0.5+0.5*gate -> retention "
          f"{0.5 + 0.5 * gmin:.3f} ~ {0.5 + 0.5 * g0:.3f}"
          f"(span only {0.5 * rng_:.3f})")
    return means, r_eff, half, rng_


def table_coord(h, g, c, q25):
    print("\n" + "=" * 90)
    print("Table 2  interior vs low-coordination atoms within a shell")
    print("=" * 90)
    print(f"{'hop':>4}{'interior':>11}{'n':>8}{'low':>11}{'n':>8}{'diff':>10}  "
          f"verdict")
    out = {}
    for hp in sorted(set(h.tolist())):
        m = h == hp
        mi, me = m & (c > q25), m & (c <= q25)
        if mi.sum() < 30 or me.sum() < 30:
            print(f"{hp:>4}{'—':>11}{int(mi.sum()):>8}{'—':>11}{int(me.sum()):>8}"
                  f"{'—':>10}  too few samples")
            continue
        gi, ge = float(g[mi].mean()), float(g[me].mean())
        d = ge - gi
        out[hp] = d
        verdict = ("low-coord clearly higher" if d > 0.03 else
                   "low-coord lower" if d < -0.03 else "no clear difference")
        print(f"{hp:>4}{gi:>11.4f}{int(mi.sum()):>8}{ge:>11.4f}{int(me.sum()):>8}"
              f"{d:>+10.4f}  {verdict}")
    return out


def table_species(h, g, z, min_n=20):
    print("\n" + "=" * 90)
    print("Table 3  hop x element (mean gate / n; n < 20 shown as -)")
    print("=" * 90)
    species = sorted(set(z.tolist()))
    print(f"{'hop':>4}" + "".join(f"{SPECIES.get(s, s):>13}" for s in species))
    for hp in sorted(set(h.tolist())):
        line = f"{hp:>4}"
        for s in species:
            m = (h == hp) & (z == s)
            line += (f"{g[m].mean():>8.4f}/{int(m.sum()):<4d}"
                     if m.sum() >= min_n else f"{'—':>8}/{int(m.sum()):<4d}")
        print(line)


def _demean(values, keys, min_cell=30):
    sums, cnts = {}, {}
    for v, k in zip(values, keys):
        sums[k] = sums.get(k, 0.0) + float(v)
        cnts[k] = cnts.get(k, 0) + 1
    out = np.full(values.shape, np.nan, dtype=float)
    for i, (v, k) in enumerate(zip(values, keys)):
        if cnts[k] >= min_cell:
            out[i] = float(v) - sums[k] / cnts[k]
    return out


def table_adjusted(h, g, c, z, n_bins=8, keep_min=200):
    uniq, cnt = np.unique(z, return_counts=True)
    keep = set(uniq[cnt >= keep_min].tolist())
    zz = np.array([zi if zi in keep else -1 for zi in z])
    qs = np.unique(np.quantile(c, np.linspace(0, 1, n_bins + 1)))
    cb = np.digitize(c, qs[1:-1])
    adj_c = _demean(g, [("c", int(b)) for b in cb])
    adj_z = _demean(g, [("z", int(s)) for s in zz])
    adj_b = _demean(g, [(int(s), int(b)) for s, b in zip(zz, cb)])

    print("\n" + "=" * 90)
    print("Table 4  residual hop effect after controlling (element x coordination)")
    print("=" * 90)
    print(f"{'hop':>4}{'raw':>10}{'coord':>10}{'elem':>10}{'both':>10}{'n':>9}")
    prof = {}
    for hp in sorted(set(h.tolist())):
        m = h == hp
        row = [hp]
        for arr in (g, adj_c, adj_z, adj_b):
            v = arr[m]
            row.append(float(np.nanmean(v)) if np.isfinite(v).any() else None)
        prof[hp] = row
        print(f"{hp:>4}" + "".join(
            f"{x:>10.4f}" if x is not None else f"{'—':>10}" for x in row[1:])
            + f"{int(np.isfinite(adj_b[m]).sum()):>9}")

    rest = [hp for hp in prof if hp != 1]
    if 1 in prof and rest:
        v1 = prof[1][4]
        rest_vals = [prof[hp][4] for hp in rest if prof[hp][4] is not None]
        if v1 is not None and rest_vals:
            rng = max(rest_vals) - min(rest_vals)
            mono = (all(a <= b for a, b in zip(rest_vals, rest_vals[1:]))
                    or all(a >= b for a, b in zip(rest_vals, rest_vals[1:])))
            print(f"\n  first shell (hop=1) vs the rest: "
                  f"{v1 - float(np.mean(rest_vals)):+.4f}")
            print(f"  residual range without hop=1: {rng:.4f}  "
                  f"monotone: {'yes' if mono else 'no'}")
            if rng < 0.05 and not mono:
                print("  -> no receptive radius beyond the first shell: the gate "
                      "behaves like "
                      "an adsorbate-neighbourhood indicator rather than a "
                      "distance-selective filter.")
            elif mono:
                print("  -> still monotone without the first shell: an "
                      "independent distance decay is present.")
            else:
                print("  -> non-trivial structure; interpret hop by hop.")
    return prof


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--split-mode", default=None,
                   help="defaults to SPLIT_MODE in train.py")
    p.add_argument("--max-graphs", type=int, default=1500)
    p.add_argument("--save-json", default=None)
    args = p.parse_args(argv)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"[FATAL] {args.ckpt} not found\n"
                         f"        train a model first: python scripts/train.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, st = load_model(args.ckpt, device)
    ds = build_dataset(args.split, args.split_mode or T.SPLIT_MODE)
    h, g, c, z = collect(model, ds, device, args.max_graphs)
    if h.size == 0:
        raise SystemExit("[FATAL] no gate activations collected")
    print(f"[Data] {h.size:,} atoms | hop {h.min()}..{h.max()} | "
          f"data directory {T.DATA_DIR}")

    q25 = table_hop(h, g, c)
    means, r_eff, half, rng_ = gate_summary(h, g, q25)
    coord_diff = table_coord(h, g, c, q25)
    table_species(h, g, z)
    prof = table_adjusted(h, g, c, z)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "split": args.split,
                       "n_atoms": int(h.size),
                       "hop_mean": means, "r_eff": r_eff,
                       "half_decay": half, "gate_range": rng_,
                       "coord_diff": coord_diff,
                       "adjusted": prof}, f, ensure_ascii=False, indent=2)
        print(f"\n[Save] {args.save_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
