"""One-command pipeline: train, analyse, ablate, summarise.

Steps
-----
  [1] full model under the bulk split        -> runs/full/
  [2] gate analysis of that checkpoint       -> rf.json + console tables
  [3] four leave-one-out ablations           -> runs/wo_*/
  [4] head-capacity control (HEAD_WIDTH=64)  -> runs/head64/
  [5] summary of val/test per configuration

Properties
----------
- every step checks its artefacts first and skips what is already done;
- step 1 is skipped when a checkpoint already exists, so a running or
  interrupted training is never duplicated or overwritten;
- Ctrl+C leaves the finished checkpoints in place and a rerun resumes;
- --dry-run prints the plan, --yes skips the prompt, --steps picks steps.

Usage
-----
    python scripts/run_all.py --dry-run
    python scripts/run_all.py --yes
    python scripts/run_all.py --yes --steps 2
    python scripts/run_all.py --yes --skip 4
"""

from __future__ import annotations

import _path  # noqa: F401 - makes models/ importable
import argparse
import glob
import os
import subprocess
import sys
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(_SCRIPTS)          # repository root
PY = sys.executable
CKPT = os.path.join(BASE, "runs")
FULL_DIR = os.path.join(CKPT, "full")
FULL_BEST = os.path.join(FULL_DIR, "best.pth")
FULL_FINAL = os.path.join(FULL_DIR, "final.pth")
LOGS = os.path.join(CKPT, "logs")

ETA = {1: "≈13 h", 2: "≈10 min", 3: "≈52 h", 4: "≈13 h", 5: "≈1 min"}
ETA_H = {1: 13.0, 2: 0.17, 3: 52.0, 4: 13.0, 5: 0.02}
DESC = {
    1: "full model (TRAIN_SUB=full)",
    2: "gate analysis (hook on rf_gate)",
    3: "four ablations (TRAIN_SUB + AB_* env)",
    4: "head-capacity control (HEAD_WIDTH=64)",
    5: "summary of val/test per config",
}


def log(msg=""):
    print(msg, flush=True)


def banner(t):
    log()
    log("=" * 78)
    log(f"  {t}")
    log("=" * 78)




class _CRBuf:
    """Terminal semantics: a newline commits a line, a carriage return overwrites."""

    def __init__(self, heartbeat=10.0):
        self.cur = ""
        self.last = time.time()      # heartbeat origin; 0 would fire every char
        self.hb = heartbeat
        self.clock = time.time()

    def feed(self, text):
        out = []
        for ch in text:
            if ch == "\n":
                if self.cur.strip():
                    out.append((self.cur, True))
                self.cur = ""
            elif ch == "\r":
                # carriage return: drop the line, emit a snapshot only after the
                # heartbeat interval
                if self.cur.strip() and time.time() - self.last >= self.hb:
                    out.append((self.cur, False))
                    self.last = time.time()
                self.cur = ""
            else:
                self.cur += ch
        return out

    def flush(self):
        if self.cur.strip():
            cur, self.cur = self.cur, ""
            return [(cur, True)]
        self.cur = ""
        return []

def run(cmd, env_extra=None, log_name=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    if log_name:
        os.makedirs(LOGS, exist_ok=True)
        path = os.path.join(LOGS, log_name)
        log(f"  $ {' '.join(cmd)}")
        log(f"    log -> {path}")
        buf = _CRBuf(heartbeat=10.0)
        with open(path, "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, cwd=BASE, env=env,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
            assert p.stdout is not None
            while True:
                chunk = p.stdout.read(8192)
                if not chunk:
                    break
                for line, committed in buf.feed(chunk.decode("utf-8", "replace")):
                    if line.strip():
                        print(line, flush=True)
                    if committed and line.strip():
                        f.write(line + "\n")
                        f.flush()
            for line, _ in buf.flush():
                if line.strip():
                    print(line, flush=True)
                    f.write(line + "\n")
            rc = p.wait()
    else:
        log(f"  $ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=BASE, env=env)
    return rc


def _recent(path, minutes=30):
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) < minutes * 60


def step1(force=False):
    banner("[1/5] full model")
    if os.path.exists(FULL_FINAL):
        log("  already finished (final.pth present), skipping")
        return 0
    if not force and os.path.exists(FULL_BEST):
        log("  WARNING: full/best.pth already exists (a running or interrupted run)")
        log("     skipping step 1 to avoid competing for the GPU or overwriting.")
        log("     to retrain anyway: python scripts/run_all.py --yes "
            "--force-step1 --steps 1")
        return 0
    log("  ETA 13 h (200 epochs max, patience 25, usually stops around 80-110)")
    return run([PY, os.path.join(_SCRIPTS, "train.py")],
               env_extra={"TRAIN_SUB": "full"}, log_name="step1_full.log")


def step2():
    banner("[2/5] gate analysis")
    if not os.path.exists(FULL_FINAL):
        log("  FAIL: full/final.pth missing, training has not finished")
        log("    the analysis needs the finished weights; an intermediate checkpoint")
        log("    would be cached forever because step 2 skips an existing json.")
        return 1
    out_json = os.path.join(BASE, "rf_singletask_1500.json")
    if os.path.exists(out_json):
        log(f"  {os.path.basename(out_json)} already exists; delete it to rerun")
        return 0
    return run([PY, os.path.join(_SCRIPTS, "analyze_gate.py"),
                "--ckpt", FULL_BEST,
                "--max-graphs", "1500",
                "--save-json", out_json], log_name="step2_rf.log")


def step3(force=False):
    banner("[3/5] four ablations")
    todo = [n for n in ("wo_emb", "wo_gate", "wo_qbf", "wo_dual")
            if force or not os.path.exists(os.path.join(CKPT, n, "final.pth"))]
    if not todo:
        log("  all four are done, skipping")
        return 0
    log(f"  to run: {todo}   ETA {13 * len(todo)} h")
    return run([PY, os.path.join(_SCRIPTS, "run_ablations.py"), "--yes",
                "--only", ",".join(todo)], log_name="step3_ablation.log")


def step4(force=False):
    banner("[4/5] head-capacity control (HEAD_WIDTH=64)")
    d = os.path.join(CKPT, "head64")
    if os.path.exists(os.path.join(d, "final.pth")) and not force:
        log("  already done, skipping")
        return 0
    log("  readout 90,625 -> 43,329 parameters (-52%), everything else fixed; "
        "ETA 13 h")
    return run([PY, os.path.join(_SCRIPTS, "train.py")],
               env_extra={"TRAIN_SUB": "head64", "HEAD_WIDTH": "64"},
               log_name="step4_head64.log")


def step5():
    banner("[5/5] summary")
    try:
        import torch
        log(f"  {'config':<12s}{'val':>10s}{'test':>10s}")
        for d in sorted(glob.glob(os.path.join(CKPT, "*"))):
            if not os.path.isdir(d):
                continue
            val = test = None
            b, f = os.path.join(d, "best.pth"), os.path.join(d, "final.pth")
            if os.path.exists(b):
                st = torch.load(b, map_location="cpu", weights_only=False)
                val = st.get("best_val_metric")
            if os.path.exists(f):
                st = torch.load(f, map_location="cpu", weights_only=False)
                test = (st.get("test_e_mae") or {}).get("adsorption_energy")
            log(f"  {os.path.basename(d):<12s}"
                f"{(f'{val:.4f}' if val else '—'):>10s}"
                f"{(f'{test:.4f}' if test else '—'):>10s}")
    except ImportError:
        log("  (torch unavailable, skipping the summary)")

    log("\n  next steps (manual):")
    log("  1) copy the numbers above into the manuscript")
    log("  2) fill the gate tables from the json into the analysis section")
    log("  3) replace the placeholder ablation numbers with the wo_* results")
    return 0


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1,2,3,4,5",
                    help="steps to run, comma separated")
    ap.add_argument("--skip", default=None, help="steps to skip")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-step1", action="store_true")
    ap.add_argument("--allow-running", action="store_true",
                    help="allow starting while full is still training")
    args = ap.parse_args(argv)

    want = [int(s) for s in args.steps.split(",") if s.strip()]
    if args.skip:
        skip = {int(s) for s in args.skip.split(",") if s.strip()}
        want = [s for s in want if s not in skip]
    want = [s for s in want if s in STEPS]

    banner("run_all — one-command pipeline")
    for s in want:
        log(f"  step {s}: {DESC[s]:<38s}{ETA[s]}")
    total = sum(ETA_H[s] for s in want)
    log(f"  estimated total ~{total:.1f} h (finished steps are skipped)")
    log(f"  output root {CKPT}")

    if args.dry_run:
        log("\n[dry-run] done")
        return 0
    if not args.yes:
        try:
            if input("\nStart? [y/N] ").lower() not in ("y", "yes"):
                log("cancelled")
                return 1
        except EOFError:
            return 1

    # guard: refuse to start while the full model is still training
    if (os.path.exists(FULL_BEST) and not os.path.exists(FULL_FINAL)
            and not args.allow_running):
        banner("full is still training")
        log("  full/best.pth exists but full/final.pth does not, so the run is not")
        log("  finished. Starting now would compete for the GPU and analyse an")
        log("  intermediate checkpoint, polluting the cached json.")
        log("  Wait for final.pth, or pass --allow-running to override.")
        return 2

    t0 = time.time()
    for s in want:
        fn = STEPS[s]
        rc = fn(force=args.force_step1) if s in (1, 3, 4) else fn()
        if rc == 130:
            log(f"\n[interrupted] step {s} stopped; a rerun resumes from here")
            return 130
        if rc != 0:
            log(f"\n[FATAL] step {s} failed (exit code {rc})")
            return rc

    banner("all done")
    log(f"  total {(time.time() - t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
