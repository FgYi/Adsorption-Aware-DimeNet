#!/usr/bin/env python3
"""
parity_visnet.py - check that ViSNetBlockPBC matches the reference ViSNetBlock
in the non-periodic limit.
"""
import _path  # noqa: F401 - makes models/ importable
import torch
from torch_geometric.nn.models.visnet import ViSNetBlock
from torch_geometric.nn import radius_graph
from baselines import ViSNetBlockPBC

torch.manual_seed(0)
N, H = 20, 32
z = torch.randint(1, 50, (N,))
pos = torch.randn(N, 3) * 2.0
batch = torch.zeros(N, dtype=torch.long)

kw = dict(num_layers=2, num_heads=4, num_rbf=8, cutoff=5.0,
          max_num_neighbors=160, lmax=1, max_z=100,
          vecnorm_type=None, trainable_vecnorm=False,
          trainable_rbf=False, vertex=False)

official = ViSNetBlock(hidden_channels=H, **kw)
pbc = ViSNetBlockPBC(hidden_channels=H, **kw)
pbc.load_state_dict(official.state_dict())

with torch.no_grad():
    x1, _ = official(z, pos, batch)
    ei = radius_graph(pos, r=5.0, max_num_neighbors=160)
    x2, _ = pbc(z, pos, batch, ei,
                torch.zeros(ei.size(1), 3, dtype=torch.long),
                torch.eye(3).unsqueeze(0))

d = (x1 - x2).abs().max().item()
print(f"x max diff = {d:.2e}   (expected < 1e-4)")
assert d < 1e-4, "re-implementation diverges from the reference"
print("PARITY PASS")