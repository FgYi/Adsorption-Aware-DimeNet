#!/usr/bin/env python3
"""
smoke_test.py - fast sanity check; must pass before any training run.
"""
import _path  # noqa: F401 - makes models/ importable
import os, glob, pickle
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
import surface_data  # noqa: F401
from baselines import build_model

DATA_DIR = os.getenv('AA_DATA_DIR',
                     os.path.join(os.path.dirname(os.path.dirname(
                         os.path.abspath(__file__))), 'data'))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = sorted(glob.glob(DATA_DIR + r'\shard_*.pkl'))

    # 1) upper bound on Z: the DimeNet++ embedding table holds 95 entries
    zmax = 0
    for sf in tqdm(files, desc="z-scan"):
        with open(sf, 'rb') as f:
            s = pickle.load(f)
        for d in s:
            if d is not None:
                zmax = max(zmax, int(d.z.max()))
        del s
    print(f"[z] dataset max Z = {zmax}  (DimeNet++ baseline requires < 95)")
    assert zmax < 95, "Z >= 95 found: the DimeNet++ embedding table would overflow"

    # 2) forward and backward for all three baselines
    with open(files[0], 'rb') as f:
        shard = pickle.load(f)
    samples = [d for d in shard if d is not None][:4]
    batch = Batch.from_data_list(samples).to(dev)

    for mt in ['schnet', 'dimenet_pp', 'visnet']:
        model = build_model(mt, num_tasks=1, task_weights=[1.0]).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        out = model(batch)
        assert out.shape == (batch.num_graphs, 1), \
            f"{mt}: output shape {tuple(out.shape)} != ({batch.num_graphs}, 1)"
        out.sum().backward()
        print(f"[{mt}] ✓ fwd/bwd | params={n_par:,}")
    print("SMOKE PASS")


if __name__ == '__main__':
    main()