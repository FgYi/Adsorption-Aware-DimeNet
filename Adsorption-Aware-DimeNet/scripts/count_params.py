#!/usr/bin/env python3
"""
count_params.py - per-module parameter breakdown, with the readout share highlighted.

Usage:
    python count_params.py
"""
import _path  # noqa: F401 - makes models/ importable
import sys

import torch

import train as T
from adaptive_dimenet import AdaptiveDimeNetPP
from baselines import build_model

# top-level parameter names that belong to the readout head
READOUT_KEYS = ("task_heads", "ads_head", "output_model", "lin_out")


def breakdown(model, name):
    groups = {}
    for n, p in model.named_parameters():
        key = n.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    readout = sum(v for k, v in groups.items() if k in READOUT_KEYS)

    print(f"\n=== {name} ===")
    print(f"  total         {total:>10,}")
    print(f"  readout       {readout:>10,}   ({readout / total:5.2%})")
    print(f"  backbone      {total - readout:>10,}")
    print("  groups:")
    for k in sorted(groups, key=lambda x: -groups[x]):
        mark = "  <- readout" if k in READOUT_KEYS else ""
        print(f"    {k:<22s}{groups[k]:>10,}{mark}")
    return total, readout


def main():
    torch.manual_seed(0)

    # 1) ours: same kwargs and ablation flags as in train.py
    flags = {"use_adaptive_embedding": True, "use_distance_gating": True,
             "use_qbf": True, "use_dual_stream": True}
    ours = AdaptiveDimeNetPP(num_tasks=1, task_weights=[1.0],
                             **T.MODEL_KWARGS, **flags)
    breakdown(ours, "Ours (AdaptiveDimeNetPP, EGQD)")

    # 2) the three baselines
    for mt in ("schnet", "dimenet_pp", "visnet"):
        m = build_model(mt, num_tasks=1, task_weights=[1.0],
                        **T.BASELINE_KWARGS[mt])
        breakdown(m, mt)

    print("\nNote: if a baseline readout is much smaller than ours, align the "
          "head width before comparing accuracy, otherwise the gap is partly a "
          "capacity difference.")


if __name__ == "__main__":
    sys.exit(main())
