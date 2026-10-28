"""Adsorption-aware DimeNet++.

Extends DimeNet++ with (i) an adsorbate-relative embedding built from the atom
tag, the hop distance to the adsorbate and a continuous distance prior,
(ii) adsorbate-centred four-body (quadruplet) basis functions injected into
every interaction block, (iii) an optional distance gate, and (iv) a
dual-stream readout over the whole slab and the adsorbate sub-graph.

Periodic boundary conditions are handled by passing edge vectors through the
lattice shift: r = pos[dst] + shift @ cell - pos[src].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer, SphericalBasisLayer, ResidualLayer, OutputPPBlock, triplets,
)


# ---------------------------- utilities ----------------------------
def _make_output_block(num_radial, hidden_channels, out_emb_channels,
                       out_channels, num_layers, act):
    try:
        return OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                             out_channels, num_layers, act, 'zeros')
    except TypeError:  # older PyG without output_initializer
        return OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                             out_channels, num_layers, act)


def _build_key_index(keys):
    order = torch.argsort(keys)
    return keys[order], order


def _lookup(sorted_keys, order, queries):
    """Ordered key lookup; returns the index and a hit mask."""
    if sorted_keys.numel() == 0 or queries.numel() == 0:
        return (torch.zeros(0, dtype=torch.long, device=queries.device),
                torch.zeros(0, dtype=torch.bool, device=queries.device))
    pos = torch.searchsorted(sorted_keys, queries)
    pos = pos.clamp(max=sorted_keys.numel() - 1)
    hit = sorted_keys[pos] == queries
    return order[pos], hit


class Envelope(nn.Module):
    """DimeNet polynomial envelope: env(0)=1, env(1)=0, zero slope at both ends."""

    def __init__(self, exponent=5):
        super().__init__()
        self.e = int(exponent)

    def forward(self, x):
        x = x.clamp(min=0.0, max=1.0)
        if self.e == 1:
            return 1.0 - x
        return 1.0 - (self.e + 1) * x.pow(self.e) + self.e * x.pow(self.e + 1)


class DihedralBasisLayer(nn.Module):
    """Fourier basis of the dihedral angle in [-pi, pi]; size 2*num_freqs."""

    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer(
            'freqs', torch.arange(1, num_freqs + 1, dtype=torch.float32))

    def forward(self, phi):
        phi = phi.unsqueeze(-1)
        freqs = self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(freqs * phi), torch.cos(freqs * phi)], dim=-1)


# ------------------------- embedding block -------------------------
class AdaptiveEmbeddingBlock(nn.Module):
    def __init__(self, hidden_channels, num_radial, cutoff,
                 envelope_exponent=5, use_adaptive=True):
        super().__init__()
        self.use_adaptive = use_adaptive
        H = hidden_channels
        self.emb_z = nn.Embedding(119, H)
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.lin_rbf = nn.Linear(num_radial, H)
        if use_adaptive:
            self.emb_tag = nn.Embedding(3, H)
            self.emb_hops = nn.Embedding(11, H)
            self.emb_rf = nn.Linear(1, H)
            self.lin_node = nn.Linear(4 * H, H)
            self.lin_edge = nn.Linear(3 * H, H)
        else:
            # official DimeNet++ embedding: cat([h_i, h_j, rbf]) -> linear
            self.lin = nn.Linear(3 * H, H)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.emb_z.weight, -1.732, 1.732)
        self.rbf.reset_parameters()
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        self.lin_rbf.bias.data.fill_(0)
        if self.use_adaptive:
            nn.init.uniform_(self.emb_tag.weight, -1.732, 1.732)
            nn.init.uniform_(self.emb_hops.weight, -1.732, 1.732)
            self.emb_rf.reset_parameters()
            glorot_orthogonal(self.lin_node.weight, scale=2.0)
            self.lin_node.bias.data.fill_(0)
            glorot_orthogonal(self.lin_edge.weight, scale=2.0)
            self.lin_edge.bias.data.fill_(0)
        else:
            # official init: normal(0, 1/sqrt(3H))
            self.lin.weight.data.normal_(0, 1.0 / math.sqrt(3 * self.emb_z.embedding_dim))
            self.lin.bias.data.fill_(0)

    def forward(self, z, tag, hops, rf, i, j, rbf):
        rbf_h = F.silu(self.lin_rbf(rbf))  # activation after lin_rbf, as official
        if self.use_adaptive:
            h_z = self.emb_z(z)
            h_tag = self.emb_tag(tag)
            h_hops = self.emb_hops(hops.clamp(0, 10))
            h_rf = self.emb_rf(rf.unsqueeze(-1))
            h_node = self.lin_node(torch.cat([h_z, h_tag, h_hops, h_rf], dim=-1))
            x = F.silu(self.lin_edge(torch.cat([h_node[j], h_node[i], rbf_h], -1)))
            return h_node, x
        else:
            # official DimeNet++: h = emb(z); x = act(lin(cat([h[i], h[j], rbf_h])))
            h = self.emb_z(z)
            x = F.silu(self.lin(torch.cat([h[i], h[j], rbf_h], dim=-1)))
            return h, x


# ------------------------ interaction block ------------------------
class AdaptiveInteractionPPBlock(nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, quad_dim, quad_emb_size,
                 num_before_skip, num_after_skip, act,
                 use_bias=True, use_qbf=True):
        super().__init__()
        self.act = act
        self.use_qbf = use_qbf
        H = hidden_channels
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, H, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        if use_qbf:
            self.lin_qbf1 = nn.Linear(quad_dim, quad_emb_size, bias=False)
            self.lin_qbf2 = nn.Linear(quad_emb_size, int_emb_size, bias=False)
            # small non-zero init: an exact zero would keep lin_qbf gradient-free
            self.qbf_scale = nn.Parameter(torch.full((1,), 1e-2))
        else:
            self.lin_qbf1 = self.lin_qbf2 = None
            self.qbf_scale = None
        self.lin_kj = nn.Linear(H, H, bias=use_bias)
        self.lin_ji = nn.Linear(H, H, bias=use_bias)
        self.lin_down = nn.Linear(H, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, H, bias=False)
        self.lin = nn.Linear(H, H, bias=use_bias)
        self.layers_before_skip = nn.ModuleList(
            [ResidualLayer(H, act) for _ in range(num_before_skip)])
        self.layers_after_skip = nn.ModuleList(
            [ResidualLayer(H, act) for _ in range(num_after_skip)])
        self.reset_parameters()

    def reset_parameters(self):
        for lin in (self.lin_rbf1, self.lin_rbf2, self.lin_sbf1, self.lin_sbf2,
                    self.lin_down, self.lin_up):
            glorot_orthogonal(lin.weight, scale=2.0)
        if self.use_qbf:
            glorot_orthogonal(self.lin_qbf1.weight, scale=2.0)
            glorot_orthogonal(self.lin_qbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        if self.lin_kj.bias is not None:
            self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        if self.lin_ji.bias is not None:
            self.lin_ji.bias.data.fill_(0)
        glorot_orthogonal(self.lin.weight, scale=2.0)
        if self.lin.bias is not None:
            self.lin.bias.data.fill_(0)
        for lyr in self.layers_before_skip:
            lyr.reset_parameters()
        for lyr in self.layers_after_skip:
            lyr.reset_parameters()

    def forward(self, x, rbf, sbf, qbf_rows, qbf_vals, idx_kj, idx_ji, num_edges):
        # activation between the two basis projections; without it they collapse
        # into a single linear map
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        rbf_h = self.lin_rbf2(self.lin_rbf1(rbf))
        x_kj = x_kj * rbf_h
        x_kj = self.act(self.lin_down(x_kj))
        x_kj = x_kj[idx_kj]
        sbf_h = self.lin_sbf2(self.lin_sbf1(sbf))
        x_kj = x_kj * sbf_h
        if self.use_qbf and qbf_vals is not None and qbf_rows is not None \
                and qbf_rows.numel() > 0:
            # sparse QBF: transform only the Q rows that carry quadruplets, then
            # scatter-add back
            # to their triplet rows (several quadruplets on one triplet sum up)
            qbf_h = self.lin_qbf2(self.act(self.lin_qbf1(qbf_vals)))  # (Q, int_emb)
            x_kj = x_kj.index_add(0, qbf_rows,
                                  (self.qbf_scale * qbf_h).to(x_kj.dtype))
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=num_edges, reduce='sum')
        x_kj = self.act(self.lin_up(x_kj))
        h = x_ji + x_kj
        for lyr in self.layers_before_skip:
            h = lyr(h)
        h = self.act(self.lin(h)) + x
        for lyr in self.layers_after_skip:
            h = lyr(h)
        return h


# ---------------------------- main model ----------------------------
class AdaptiveDimeNetPP(nn.Module):
    recommended_lr: float = 1e-4
    recommended_grad_clip: float = 10.0
    enable_grad_finite_check: bool = False
    def __init__(self, hidden_channels=128, num_blocks=4, int_emb_size=64,
                 basis_emb_size=8, out_emb_channels=256, num_radial=6,
                 num_spherical=7, cutoff=6.0, r_quad=4.0, max_num_neighbors=128,
                 envelope_exponent=5, num_before_skip=1, num_after_skip=2,
                 num_output_layers=3, num_tasks=1, force_energy_task_idx=None,
                 task_weights=None,
                 use_adaptive_embedding=True, use_distance_gating=True,
                 use_qbf=True, use_dual_stream=True,
                 head_width=None,
                 num_dihedral_freqs=6, quad_emb_size=32, quad_norm=64.0):
        super().__init__()
        assert num_blocks >= 1, "num_blocks must be >= 1"
        assert r_quad <= cutoff, "r_quad must be <= cutoff"
        act = activation_resolver('swish')
        self.cutoff = cutoff
        self._batch_checked = False
        self.r_quad = float(r_quad)
        self.quad_norm = float(quad_norm)
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks
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
        H = hidden_channels
        self.use_adaptive_embedding = use_adaptive_embedding
        self.use_distance_gating = use_distance_gating
        self.use_qbf = use_qbf
        self.use_dual_stream = use_dual_stream
        self.emb_block = AdaptiveEmbeddingBlock(
            H, num_radial, cutoff, envelope_exponent,
            use_adaptive=use_adaptive_embedding)
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)
        self.dihedral_basis = DihedralBasisLayer(num_dihedral_freqs)
        self.quad_envelope = Envelope(envelope_exponent)
        # QBF = [RBF(l→k), SBF(c,j,k), DBF(φ)]
        # sbf(c, j, k) shares its basis with the main sbf path by design:
        # the redundancy keeps the four-body term self-consistent
        self.quad_dim = (num_radial + num_spherical * num_radial
                         + 2 * num_dihedral_freqs)
        self.interaction_blocks = nn.ModuleList([
            AdaptiveInteractionPPBlock(
                H, int_emb_size, basis_emb_size,
                num_spherical, num_radial, self.quad_dim, quad_emb_size,
                num_before_skip, num_after_skip, act,
                use_bias=True, use_qbf=use_qbf)  # official block carries bias
            for _ in range(num_blocks)
        ])
        self.output_blocks = nn.ModuleList([
            _make_output_block(num_radial, H, out_emb_channels, H,
                               num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])
        if use_adaptive_embedding:
            self.layer_weights = nn.Parameter(
                torch.ones(num_blocks + 1) / (num_blocks + 1))
            self.node_skip_scale = nn.Parameter(torch.zeros(1))  # C3
        else:
            self.register_buffer('layer_weights', None)
            self.register_buffer('node_skip_scale', None)
        if use_distance_gating:
            self.rf_gate = nn.Sequential(
                nn.Linear(H + 1, H), nn.SiLU(), nn.Linear(H, H), nn.Sigmoid())
        else:
            self.rf_gate = None
        # head_width: hidden width of the readout head, default hidden_channels.
        # Set to 64 to halve the readout parameters, which makes the comparison
        # with narrow-readout baselines such as SchNet like-for-like.
        HW = max(2, int(head_width)) if head_width else H
        self.head_width = HW
        if use_dual_stream:
            self.ads_head = nn.Sequential(
                nn.Linear(H * 2, HW), nn.SiLU(), nn.Linear(HW, H))
        else:
            self.ads_head = None
        head_in = H * 2 if use_dual_stream else H
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, HW), nn.SiLU(),
                nn.Linear(HW, HW // 2), nn.SiLU(), nn.Linear(HW // 2, 1))
            for _ in range(num_tasks)])
        self.use_multitask_uncertainty = (num_tasks > 1)
        if self.use_multitask_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.register_buffer('log_vars', None)
        self.register_buffer('y_mean', torch.zeros(num_tasks))
        self.register_buffer('y_std', torch.ones(num_tasks))

    def set_target_stats(self, y_mean, y_std):
        self.y_mean.copy_(y_mean)
        self.y_std.copy_(y_std)

    def _check_first_batch(self, data, batch, edge_index, num_nodes,
                           num_edges, tag, hops):
        """Validate indices of the first batch on the CPU.

        Out-of-range assertions on CUDA are asynchronous, carry no Python
        line number and poison the whole context, so the first batch is
        checked synchronously and reported as a readable exception.
        """
        ei_max, ei_min = int(edge_index.max()), int(edge_index.min())
        if ei_max >= num_nodes:
            raise IndexError(f"edge_index out of range: {ei_max} >= num_nodes={num_nodes}")
        if ei_min < 0:
            raise IndexError("edge_index contains negative values")
        qe = getattr(data, 'quad_edges', None)
        if qe is not None and qe.numel() > 0 and int(qe.max()) >= num_edges:
            raise IndexError(f"quad_edges out of range: {int(qe.max())} >= {num_edges}")
        z_min, z_max = int(data.z.min()), int(data.z.max())
        if z_min < 0 or z_max >= 119:
            raise IndexError(f"z out of range [{z_min},{z_max}] (embedding table 119)")
        if tag is not None and tag.numel() > 0:
            t_min, t_max = int(tag.min()), int(tag.max())
            if t_min < 0 or t_max > 2:
                raise IndexError(f"tag out of range [{t_min},{t_max}] (table 3)")
        if hops is not None and hops.numel() > 0:
            h_min, h_max = int(hops.min()), int(hops.max())
            if h_min < 0 or h_max > 10:
                raise IndexError(f"hops out of range [{h_min},{h_max}] (table 11)")
        self._batch_checked = True

    def forward(self, data):
        z, pos = data.z, data.pos
        num_nodes = z.size(0)
        device = pos.device
        batch = (data.batch if data.batch is not None
                 else torch.zeros(num_nodes, dtype=torch.long, device=device))
        # ---------- attributes resolved on demand ----------
        tag = hops = rf = None
        if self.use_adaptive_embedding or self.use_dual_stream:
            if not hasattr(data, 'tag'):
                raise AttributeError(
                    "use_adaptive_embedding/use_dual_stream require data.tag")
            tag = data.tag
        if self.use_adaptive_embedding or self.use_distance_gating:
            if not hasattr(data, 'rf_weights'):
                raise AttributeError(
                    "use_adaptive_embedding/use_distance_gating require data.rf_weights")
            rf = data.rf_weights
        if self.use_adaptive_embedding:
            if not hasattr(data, 'hops_to_adsorbate'):
                raise AttributeError(
                    "use_adaptive_embedding requires data.hops_to_adsorbate")
            hops = data.hops_to_adsorbate
        # ---------- edges ----------
        ei = getattr(data, 'edge_index', None)
        if ei is not None and ei.numel() > 0:
            edge_index = ei
        else:
            # fallback: non-periodic radius graph, debug only; use preprocessed data
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                      max_num_neighbors=self.max_num_neighbors)
        num_edges = edge_index.size(1)
        if num_edges == 0:
            B = int(batch.max().item()) + 1
            return torch.zeros(B, self.num_tasks, device=device)
        src, dst = edge_index[0], edge_index[1]
        if not getattr(self, '_batch_checked', False):
            self._check_first_batch(data, batch, edge_index, num_nodes,
                                    num_edges, tag, hops)
        # ---------- periodic edge vectors ----------
        # PyG concatenates the (3,3) cells along dim 0, so a batched cell is
        # (3B, 3). Reshaping to (-1, 3, 3) yields (1, 3, 3) for a single graph
        # and (B, 3, 3) for a batch, which keeps cells[batch[src]] in range.
        edge_shift = getattr(data, 'edge_shift', None)
        cell = getattr(data, 'cell', None)
        if edge_shift is not None and edge_shift.size(0) != num_edges:
            raise ValueError(f"edge_shift rows {edge_shift.size(0)} != "
                             f"edges {num_edges} (data/graph mismatch)")
        if (edge_shift is not None and cell is not None
                and torch.abs(cell).sum() > 0):
            if cell.dim() == 2:
                cell = cell.view(-1, 3, 3)
            cells = cell.to(pos.dtype)  # (B,3,3)
            num_graphs_now = (int(batch.max().item()) + 1
                              if batch.numel() > 0 else 1)
            if cells.size(0) != num_graphs_now:
                raise ValueError(
                    f"cell batch mismatch: cells={tuple(cells.shape)} vs "
                    f"num_graphs={num_graphs_now}; per-graph cell must be (3,3) "
                    f"and SurfaceData.__cat_dim__('cell') must be 0")
            edge_cell = cells[batch[src]]  # (E,3,3)
            shift_vec = torch.einsum('ei,eij->ej',
                                     edge_shift.to(pos.dtype), edge_cell)
            edge_vec = pos[dst] + shift_vec - pos[src]  # src→dst
        else:
            edge_vec = pos[dst] - pos[src]
        dist = edge_vec.norm(dim=-1)
        # ---------- triplets and angles (official convention) ----------
        trip_out = triplets(edge_index, num_nodes=num_nodes)
        assert len(trip_out) == 7, "requires triplets() from PyG >= 2.3 (7 returns)"
        col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji = trip_out
        rbf = self.rbf(dist)
        num_triplets = idx_kj.size(0)
        if num_triplets > 0:
            v1 = edge_vec[idx_ji]  # same as the official pos_ij
            v2 = edge_vec[idx_kj]  # same as the official pos_jk
            a = (v1 * v2).sum(-1)
            b = torch.cross(v1, v2, dim=-1).norm(dim=-1)
            angle = torch.atan2(b, a)
            sbf = self.sbf(dist, angle, idx_kj)
        else:
            sbf = torch.zeros(0, self.sbf.num_spherical * self.sbf.num_radial,
                              device=device)
        # ---------- smooth four-body QBF, built on the fly (sparse) ----------
        # Quadruplets only cover a small fraction of all triplets, so the basis
        # is materialised for those rows only ((Q, dim) with Q ~ tens of
        # thousands) and scatter-added back. This is mathematically identical to
        # a dense (M, dim) tensor but skips the linear maps on the zero rows.
        qbf_rows, qbf_vals = None, None
        if self.use_qbf:
            qe = getattr(data, 'quad_edges', None)
            qd = getattr(data, 'quad_dists', None)
            if (qe is not None and qd is not None and qe.numel() > 0
                    and num_triplets > 0 and num_edges > 0):
                e_ji, e_kj, e_lk = qe[:, 0].long(), qe[:, 1].long(), qe[:, 2].long()
                # rebuild the (e_ji, e_kj) -> triplet map in the forward pass
                tkeys = idx_ji * num_edges + idx_kj
                sorted_tkeys, order = _build_key_index(tkeys)
                qkeys = e_ji * num_edges + e_kj
                t_idx, hit = _lookup(sorted_tkeys, order, qkeys)
                if not bool(hit.all()):
                    e_ji, e_kj, e_lk = e_ji[hit], e_kj[hit], e_lk[hit]
                    qd, t_idx = qd[hit], t_idx[hit]
                if e_ji.numel() > 0:
                    # Smooth envelope. Enveloping only d(c, .) would let a
                    # quadruplet appear or vanish with non-zero weight when a
                    # chain edge crosses the cutoff, so every chain edge is
                    # enveloped as well (goes to zero at the cutoff).
                    w = (self.quad_envelope(qd / self.r_quad).prod(dim=-1)
                         * self.quad_envelope(dist[e_kj] / self.cutoff)
                         * self.quad_envelope(dist[e_lk] / self.cutoff))
                    # standard dihedral along c-j-k-l
                    r_c = edge_vec[e_ji]   # j→c
                    r_k = -edge_vec[e_kj]  # j→k
                    b1, b2, b3 = -r_c, r_k, -edge_vec[e_lk]
                    n1 = torch.cross(b1, b2, dim=-1)
                    n2 = torch.cross(b2, b3, dim=-1)
                    # b2 must be normalised before projection; otherwise the
                    # y component scales with d(j,k) and phi is not the true
                    # dihedral angle.
                    b2_hat = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    m1 = torch.cross(n1, b2_hat, dim=-1)
                    x_ = (n1 * n2).sum(-1)
                    y_ = (m1 * n2).sum(-1)
                    phi = torch.atan2(y_, x_)  # (Q,)
                    dbf = self.dihedral_basis(phi)
                    feat = torch.cat([rbf[e_lk], sbf[t_idx], dbf], dim=-1)
                    qbf_vals = (feat * w.unsqueeze(-1)) / self.quad_norm
                    qbf_rows = t_idx
        # ---------- embedding and interaction ----------
        i, j = edge_index[1], edge_index[0]
        h_node, x = self.emb_block(z, tag, hops, rf, i, j, rbf)
        P = self.output_blocks[0](x, rbf, i, num_nodes=num_nodes)
        if self.use_adaptive_embedding:
            P = P + self.node_skip_scale * h_node  # C3
        layer_outputs = [P]
        for int_block, out_block in zip(self.interaction_blocks,
                                        self.output_blocks[1:]):
            x = int_block(x, rbf, sbf, qbf_rows, qbf_vals, idx_kj, idx_ji, num_edges)
            P = P + out_block(x, rbf, i, num_nodes=num_nodes)
            layer_outputs.append(P)
        if self.layer_weights is not None:
            w = F.softmax(self.layer_weights, dim=0)
            h_out = sum(w[k] * layer_outputs[k]
                        for k in range(self.num_blocks + 1))
        else:
            h_out = layer_outputs[0]
            for k in range(1, len(layer_outputs)):
                h_out = h_out + layer_outputs[k]
        if self.use_distance_gating:
            gate = self.rf_gate(torch.cat([h_out, rf.unsqueeze(-1)], dim=-1))
            h_out = h_out * (0.5 + 0.5 * gate)
        h_global = scatter(h_out, batch, dim=0, reduce='mean')
        num_graphs = h_global.size(0)
        if self.use_dual_stream:
            ads_mask = (tag == 2)
            if ads_mask.any():
                ads_batch = batch[ads_mask]
                h_ads_sum = scatter(h_out[ads_mask], ads_batch, dim=0,
                                    dim_size=num_graphs, reduce='sum')
                ads_count = scatter(torch.ones(ads_mask.sum(), device=device),
                                    ads_batch, dim=0, dim_size=num_graphs,
                                    reduce='sum')
                h_ads_mean = h_ads_sum / ads_count.clamp(min=1).unsqueeze(-1)
                h_ads_mean[ads_count == 0] = h_global[ads_count == 0]
            else:
                h_ads_mean = h_global.clone()
            h_ads = self.ads_head(torch.cat([h_ads_mean, h_global], dim=-1))
            combined = torch.cat([h_global, h_ads], dim=-1)
        else:
            combined = h_global
        return torch.cat([head(combined) for head in self.task_heads], dim=-1)