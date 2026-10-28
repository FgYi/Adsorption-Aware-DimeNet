"""Serial leave-one-out ablation runner.

Each configuration removes one component (and, when removal would leave the
run meaningless, the dependent ones) while the data, batch construction,
optimizer, schedule and early stopping stay identical to the full model.

Two details matter. First, the checkpoint directory is derived from the ablation
flags alone, and on case-insensitive file systems the five configurations map
to the same directory, so every run gets an explicit TRAIN_SUB. Second, the
flags are read from the environment, which keeps train.py unchanged.

Usage
-----
    python scripts/run_ablations.py --dry-run
    python scripts/run_ablations.py --yes
    python scripts/run_ablations.py --only wo_qbf,wo_gate --yes

Output
------
    runs/<name>/{best.pth, final.pth}
    ablation_summary.csv / ablation_summary.md
"""

from __future__ import annotations

import _path  # noqa: F401 - makes models/ importable
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(_SCRIPTS)          # repository root
TRAIN = os.path.join(_SCRIPTS, "train.py")

# name -> environment variable (components not listed stay enabled)
CONFIGS = [
    ("full",     {}),
    ("wo_emb",   {"AB_EMB": "0"}),
    ("wo_gate",  {"AB_GATE": "0"}),
    ("wo_qbf",   {"AB_QBF": "0"}),
    ("wo_dual",  {"AB_DUAL": "0"}),
]
EPOCH_MIN = 9.0      # minutes per epoch (48 graphs/batch, validation included)
EARLY_STOP_EPOCHS = 85   # the reference run stopped at epoch 81; used for the ETA


def log(msg=""):
    print(msg, flush=True)


def banner(t):
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def final_of(name):
    return os.path.join(BASE, "runs", name, "final.pth")


def read_metrics(name):
    p = final_of(name)
    if not os.path.exists(p):
        return None
    try:
        import torch
        st = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    te = (st.get("test_e_mae") or {}).get("adsorption_energy")
    sd = st.get("model") or {}
    n_par = (int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
             if sd else None)
    return {"test": te, "params": n_par,
            "flags": st.get("ablation_flags")}


def write_summary(rows, out_csv, out_md):
    base = next((r for r in rows if r["name"] == "full"), None)
    b = base.get("test") if base else None
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "test_MAE_eV", "delta_vs_full", "params", "flags"])
        for r in rows:
            d = (r["test"] - b) if (r["test"] is not None and b is not None) else ""
            w.writerow([r["name"], r["test"], d, r["params"], r["flags"]])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| config | test MAE (eV) | Δ vs full | params |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            d = (f"{r['test'] - b:+.4f}"
                 if (r["test"] is not None and b is not None) else "—")
            t = f"{r['test']:.4f}" if r["test"] is not None else "—"
            n = f"{r['params']:,}" if r["params"] else "—"
            f.write(f"| {r['name']} | {t} | {d} | {n} |\n")
    log(f"\n[Save] {out_csv}")
    log(f"[Save] {out_md}")


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--only", default=None,
                   help="comma separated: full,wo_emb,wo_gate,wo_qbf,wo_dual")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)

    names = [c[0] for c in CONFIGS]
    todo = CONFIGS
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        bad = [w for w in want if w not in names]
        if bad:
            raise SystemExit(f"unknown configuration {bad}; choose from {names}")
        todo = [c for c in CONFIGS if c[0] in want]

    hours = len(todo) * EARLY_STOP_EPOCHS * EPOCH_MIN / 60
    banner("leave-one-out ablation (data = data/, model = AdaptiveDimeNetPP)")
    log(f"  configs  : {[n for n, _ in todo]}")
    log(f"  output   : {os.path.join(BASE, 'runs')}/<name>")
    log(f"  ETA      : ~{hours:.0f} h ({EARLY_STOP_EPOCHS} epochs x "
        f"{EPOCH_MIN:.0f} min/epoch)")
    if args.dry_run:
        log("\n  [dry-run] would run:")
        for n, env in todo:
            e = " ".join(f"{k}={v}" for k, v in {**env, "TRAIN_SUB": n}.items())
            log(f"    {e} python scripts/train.py")
        return 0
    if not args.yes:
        try:
            if input(f"\nStart (~{hours:.0f} h)? [y/N] ").lower() not in ("y", "yes"):
                log("cancelled")
                return 1
        except EOFError:
            return 1

    rows, t_all = [], time.time()
    for i, (name, env_extra) in enumerate(todo, 1):
        banner(f"[{i}/{len(todo)}] {name}")
        if os.path.exists(final_of(name)) and not args.force:
            log(f"  already done, skipping: {final_of(name)}")
        else:
            env = os.environ.copy()
            env.update({"TRAIN_SUB": name, "PYTHONUNBUFFERED": "1",
                        "PYTHONIOENCODING": "utf-8"})
            env.update(env_extra)
            log(f"  TRAIN_SUB={name}  {env_extra}")
            rc = subprocess.call([sys.executable, TRAIN], env=env, cwd=BASE)
            if rc == 130 or rc < 0:
                log(f"[interrupted] {name} stopped; a rerun restarts this config.")
                return 130
            if rc != 0:
                raise SystemExit(f"[FATAL] {name} failed (exit code {rc})")
            log(f"  done in {(time.time() - t_all) / 3600:.2f} h (cumulative)")
        m = read_metrics(name) or {}
        rows.append({"name": name, "test": m.get("test"),
                     "params": m.get("params"), "flags": m.get("flags")})

    write_summary(rows, os.path.join(BASE, "ablation_legacy_summary.csv"),
                  os.path.join(BASE, "ablation_legacy_summary.md"))
    banner("finished")
    log(f"  total {(time.time() - t_all) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
