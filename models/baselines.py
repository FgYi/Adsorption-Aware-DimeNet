"""PBC-compatible SchNet / DimeNet++ / ViSNet baselines.

All three share the interface of :class:`AdaptiveDimeNetPP` (multi-task heads,
``y_mean``/``y_std``/``log_vars``), so a single training script can drive them.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet, DimeNetPlusPlus
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter
from torch_geometric.nn.models.visnet import (
    ViSNetBlock, GatedEquivariantBlock,
)


# ============================================================
# ------------------------------ shared ------------------------------
# ============================================================
def _compute_edge_vec(data):
    """
    Periodic edge vectors, oriented dst - src.
    Handles batch=None as well as per-graph (3,3) and batched (3B,3) cells.
    """
    edge_index = data.edge_index
    edge_shift = data.edge_shift
    cell = data.cell
    pos = data.pos
    batch = data.batch

    if cell.dim() == 2:
        cell = cell.view(-1, 3, 3)
    cells = cell.to(pos.dtype)
    src, dst = edge_index[0], edge_index[1]
    edge_cell = cells[batch[src]]
    shift_vec = torch.einsum('ei,eij->ej',
                             edge_shift.to(pos.dtype), edge_cell)
    return pos[dst] + shift_vec - pos[src]


def _make_task_heads(head_dim, hidden_dim, num_tasks):
    """Output heads matching the structure used by AdaptiveDimeNetPP."""
    return nn.ModuleList([
        nn.Sequential(
            nn.Linear(head_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        ) for _ in range(num_tasks)])


class BaselineBase(nn.Module):
    """Shared interface: task_priority, y_mean, y_std, log_vars."""

    # Training preferences. Defaults follow the DimeNet++ recipe; subclasses
    # override them and the training script reads them from the model.
    recommended_lr: float = 1e-4
    recommended_grad_clip: float = 10.0
    enable_grad_finite_check: bool = False

    def __init__(self, num_tasks, task_weights=None,
                 force_energy_task_idx=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.force_energy_task_idx = force_energy_task_idx

        if task_weights is None:
            priority = [0.2] * num_tasks
            if num_tasks > 0:
                priority[0] = 1.0
        else:
            assert len(task_weights) == num_tasks
            priority = list(task_weights)
        self.register_buffer('task_priority',
                             torch.tensor(priority, dtype=torch.float32))

        self.register_buffer('y_mean', torch.zeros(num_tasks))
        self.register_buffer('y_std', torch.ones(num_tasks))

        self.use_multitask_uncertainty = (num_tasks > 1)
        if self.use_multitask_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.register_buffer('log_vars', None)

    def set_target_stats(self, y_mean, y_std):
        self.y_mean.copy_(y_mean)
        self.y_std.copy_(y_std)


# ============================================================
# 1. SchNet (PBC)
# ============================================================
class PBCInteractionGraph(nn.Module):
    """
    Replaces the RadiusInteractionGraph of SchNet.
    SchNet.forward calls self.interaction_graph(pos, batch); the current
    data object is injected via set_data so the PBC edges can be reused.
    """
    def __init__(self, cutoff=5.0, max_num_neighbors=160):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self._data = None

    def set_data(self, data):
        self._data = data

    def forward(self, pos, batch):
        data = self._data
        if (data is None or not hasattr(data, 'edge_shift')
                or data.edge_shift is None or data.edge_shift.numel() == 0):
            # fallback: non-periodic
            ei = radius_graph(pos, r=self.cutoff, batch=batch,
                              max_num_neighbors=self.max_num_neighbors)
            ew = (pos[ei[0]] - pos[ei[1]]).norm(dim=-1)
            return ei, ew

        edge_index = data.edge_index
        edge_vec = _compute_edge_vec(data)
        return edge_index, edge_vec.norm(dim=-1)


class SchNetPBC(BaselineBase):
    """SchNet with PBC and multi-task head。"""

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50,
                 cutoff=5.0, max_num_neighbors=160,
                 num_tasks=1, task_weights=None, force_energy_task_idx=None,
                 **kwargs):
        super().__init__(num_tasks=num_tasks, task_weights=task_weights,
                         force_energy_task_idx=force_energy_task_idx)
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            interaction_graph=PBCInteractionGraph(cutoff, max_num_neighbors),
            readout='add',
        )

        # heads consume hidden_channels // 2 features (after lin1 and act)
        head_dim = hidden_channels // 2
        self.task_heads = _make_task_heads(head_dim, head_dim, num_tasks)

    def forward(self, data):
        self.schnet.interaction_graph.set_data(data)
        z, pos, batch = data.z, data.pos, data.batch

        h = self.schnet.embedding(z)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # lin1 -> act yields [N, H/2]
        h = self.schnet.lin1(h)
        h = self.schnet.act(h)

        # global pooling: mean
        h_graph = scatter(h, batch, dim=0, reduce='mean')

        return torch.cat([head(h_graph) for head in self.task_heads], dim=-1)


# ============================================================
# 2. DimeNet++ (PBC)
# ============================================================
class DimeNetPPPBC(BaselineBase):
    """DimeNet++ with PBC and multi-task head。"""

    def __init__(self, hidden_channels=128, num_blocks=4, int_emb_size=64,
                 basis_emb_size=8, out_emb_channels=256, num_radial=6,
                 num_spherical=7, cutoff=5.0, max_num_neighbors=160,
                 envelope_exponent=5, num_before_skip=1, num_after_skip=2,
                 num_output_layers=3, num_tasks=1, task_weights=None,
                 force_energy_task_idx=None, **kwargs):
        super().__init__(num_tasks=num_tasks, task_weights=task_weights,
                         force_energy_task_idx=force_energy_task_idx)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        out_channels = hidden_channels  # keep the readout width at hidden
        self.dimenet = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )
        self.out_channels = out_channels
        self.task_heads = _make_task_heads(out_channels, out_channels, num_tasks)

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch

        edge_index = data.edge_index
        edge_vec = _compute_edge_vec(data)  # dst - src

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        dist = edge_vec.norm(dim=-1)

        # upstream PyG: pos_ij = pos[idx_i] - pos[idx_j]
        # idx_ji is j->i, so edge_vec[idx_ji] = pos[i] - pos[j]
        pos_ij = edge_vec[idx_ji]
        # upstream PyG: pos_jk = pos[idx_j] - pos[idx_k]
        # idx_kj is k->j, so edge_vec[idx_kj] = pos[j] - pos[k]
        pos_jk = edge_vec[idx_kj]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.dimenet.rbf(dist)
        sbf = self.dimenet.sbf(dist, angle, idx_kj)

        x = self.dimenet.emb(z, rbf, i, j)
        P = self.dimenet.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for inter, out in zip(self.dimenet.interaction_blocks,
                              self.dimenet.output_blocks[1:]):
            x = inter(x, rbf, sbf, idx_kj, idx_ji)
            P = P + out(x, rbf, i, num_nodes=pos.size(0))

        # global pooling: mean
        h_graph = scatter(P, batch, dim=0, reduce='mean')
        return torch.cat([head(h_graph) for head in self.task_heads], dim=-1)


# ============================================================
# 3. ViSNet (PBC)
# ============================================================
class DistancePBC(nn.Module):
    """
    Replaces the internal Distance module of ViSNetBlock.
    The ViSNet reference uses src - dst; this repository uses dst - src
    throughout, which is equivalent for scalar targets (see parity_visnet.py).
    """
    def __init__(self, add_self_loops=True):
        super().__init__()
        self.add_self_loops = add_self_loops

    def forward(self, pos, batch, edge_index, edge_shift, cell):
        if cell.dim() == 2:
            cell = cell.view(-1, 3, 3)
        cells = cell.to(pos.dtype)
        src, dst = edge_index[0], edge_index[1]
        edge_cell = cells[batch[src]]
        shift_vec = torch.einsum('ei,eij->ej',
                                 edge_shift.to(pos.dtype), edge_cell)
        edge_vec = pos[dst] + shift_vec - pos[src]

        if self.add_self_loops:
            n = pos.size(0)
            loop_index = torch.arange(n, device=pos.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            loop_vec = torch.zeros(n, 3, device=pos.device, dtype=pos.dtype)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_vec = torch.cat([edge_vec, loop_vec], dim=0)

        mask = edge_index[0] != edge_index[1]
        edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
        edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        return edge_index, edge_weight, edge_vec


class ViSNetBlockPBC(ViSNetBlock):
    """ViSNetBlock with PBC edge input。"""

    def forward(self, z, pos, batch, edge_index_pbc, edge_shift_pbc, cell):
        x = self.embedding(z)

        # use PBC edges instead of radius_graph
        distance_fn = DistancePBC(add_self_loops=True)
        edge_index, edge_weight, edge_vec = distance_fn(
            pos, batch, edge_index_pbc, edge_shift_pbc, cell)
        edge_attr = self.distance_expansion(edge_weight)

        mask = edge_index[0] != edge_index[1]
        edge_vec = edge_vec.clone()
        edge_vec[mask] = edge_vec[mask] / torch.norm(
            edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)

        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1),
                          dtype=x.dtype, device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](
            x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)
        return x, vec


class EquivariantScalarMT(nn.Module):
    """
    Multi-task EquivariantScalar: emits out_channels scalars.
    GatedEquivariantBlock takes (hidden, out, intermediate=None, scalar_activation=False).
    The first two are required.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.output_network = nn.ModuleList([
            GatedEquivariantBlock(hidden_channels, hidden_channels // 2,
                                  scalar_activation=True),
            GatedEquivariantBlock(hidden_channels // 2, out_channels,
                                  scalar_activation=False),
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return x + v.sum() * 0


class ViSNetPBC(BaselineBase):
    """ViSNet with PBC and multi-task head。"""

    # The gradient norm of the equivariant vector features is far larger than
    # for purely scalar models; the DimeNet++ recipe (1e-4 / 10.0) diverges to
    # NaN by the third epoch, hence the stricter defaults below.
    recommended_lr: float = 3e-5
    recommended_grad_clip: float = 1.0
    enable_grad_finite_check: bool = True

    def __init__(self, hidden_channels=128, num_layers=6, num_heads=8,
                 num_rbf=32, cutoff=5.0, max_num_neighbors=160, lmax=1,
                 max_z=100, vecnorm_type=None, trainable_vecnorm=False,
                 trainable_rbf=False, vertex=False,
                 num_tasks=1, task_weights=None, force_energy_task_idx=None,
                 **kwargs):
        super().__init__(num_tasks=num_tasks, task_weights=task_weights,
                         force_energy_task_idx=force_energy_task_idx)
        self.hidden_channels = hidden_channels

        self.representation_model = ViSNetBlockPBC(
            lmax=lmax, vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads, num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf, trainable_rbf=trainable_rbf,
            max_z=max_z, cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

        self.output_model = EquivariantScalarMT(hidden_channels, hidden_channels)
        self.task_heads = _make_task_heads(hidden_channels, hidden_channels,
                                           num_tasks)

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch

        x, v = self.representation_model(
            z, pos, batch,
            data.edge_index, data.edge_shift, data.cell,
        )
        h = self.output_model.pre_reduce(x, v)  # [N, H]

        # global pooling: mean, as in the adaptive model
        h_graph = scatter(h, batch, dim=0, reduce='mean')
        return torch.cat([head(h_graph) for head in self.task_heads], dim=-1)


# ============================================================
# ---------------------------- factory ----------------------------
# ============================================================
def build_model(model_type, **kwargs):
    if model_type == 'schnet':
        return SchNetPBC(**kwargs)
    elif model_type == 'dimenet_pp':
        return DimeNetPPPBC(**kwargs)
    elif model_type == 'visnet':
        return ViSNetPBC(**kwargs)
    elif model_type == 'adaptive':
        from adaptive_dimenet import AdaptiveDimeNetPP
        return AdaptiveDimeNetPP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")