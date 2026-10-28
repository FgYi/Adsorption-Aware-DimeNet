"""Raw dump -> sharded pickles.

Turns the text dump into :class:`SurfaceData` objects with periodic neighbour
lists, DimeNet++ triplets and adsorbate-centred quadruplets. The pipeline is
deliberately fail-loudly: a zero-sample result exits non-zero, and a per-stage
funnel points at the stage that broke.
"""

import _path  # noqa: F401 - makes models/ importable
import os, re, gc, sys, math, pickle, glob
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from pymatgen.core.periodic_table import Element
from surface_data import SurfaceData

# ---------------------------- config ----------------------------
CUTOFF = 5.0  # if 6.0 is required, raise MAX_NUM_NEIGHBORS to 192 as well
R_QUAD = 3.5
MAX_QUADS_PER_ADS = 4096
MAX_HOPS = 10
SHARD_SIZE = 5000
MAX_NUM_NEIGHBORS = 160
CHUNK = 256
NUM_RADIAL = 6
NUM_SPHERICAL = 7
ENVELOPE_EXPONENT = 5
TASK_NAMES = [
    "adsorption_energy", "band_gap", "formation_energy",
    "energy_per_atom", "fermi_level",
]
PROGRESS_EVERY = 2000
NEAR_COINCIDENT_THR = 0.5  # near-coincident atom threshold (Angstrom)
MAX_FAIL_SNIPPETS = 3  # failing snippets printed per category
assert R_QUAD <= CUTOFF
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Preprocess] Using device: {DEVICE}")


# ------------------- periodic neighbour construction -------------------
def get_periodic_shifts(cell_np, pos_np, cutoff):
    """Integer lattice shifts that can place a neighbour inside the cutoff.

    Complete because |n_k| <= floor(cutoff * |b_k|) + 1; then pruned without
    loss of accuracy to |t| <= cutoff + D.
    """
    inv = np.linalg.inv(cell_np)
    b_norms = np.linalg.norm(inv, axis=0)
    n_max = np.floor(cutoff * b_norms + 1e-9).astype(np.int64) + 1
    n_max = np.minimum(n_max, 8)
    r = [np.arange(-int(n), int(n) + 1) for n in n_max]
    g = np.meshgrid(r[0], r[1], r[2], indexing='ij')
    shifts = np.stack([x.reshape(-1) for x in g], axis=1).astype(np.int64)
    if len(pos_np) > 1:
        D = float(np.linalg.norm(
            pos_np[:, None, :] - pos_np[None, :, :], axis=-1).max())
    else:
        D = 0.0
    t_vec = shifts @ cell_np
    keep = np.linalg.norm(t_vec, axis=1) <= cutoff + D + 1e-6
    shifts = shifts[keep]
    if not (shifts == 0).all(axis=1).any():
        shifts = np.vstack([shifts, np.zeros((1, 3), dtype=np.int64)])
    if shifts.shape[0] > 512:
        raise RuntimeError(
            f"unexpectedly many periodic images ({shifts.shape[0]}): "
            f"cell too small or cutoff too large, check the data")
    return shifts


def build_neighbor_list(pos_t, cell_np, cutoff, max_num_neighbors, device):
    """Return edge_index, edge_shift, dist and an info dict.

    Every ordered pair inside the cutoff is enumerated explicitly (including
    the same atom across different images), which is symmetric by construction.
    A degree overflow raises instead of silently truncating.
    """
    N = pos_t.size(0)
    if (cell_np is not None and np.isfinite(cell_np).all()
            and abs(np.linalg.det(cell_np)) > 1e-8):
        periodic = True
    else:
        periodic = False
        cell_np = np.zeros((3, 3), dtype=np.float64)
    if periodic:
        shifts_np = get_periodic_shifts(cell_np, pos_t.detach().cpu().numpy(), cutoff)
    else:
        shifts_np = np.zeros((1, 3), dtype=np.int64)
    shifts_t = torch.tensor(shifts_np, dtype=torch.float32, device=device)
    cell_t = torch.tensor(np.asarray(cell_np), dtype=torch.float32, device=device)
    zero_sid = int(np.where((shifts_np == 0).all(axis=1))[0][0])
    src_l, dst_l, sid_l, dist_l = [], [], [], []
    for s in range(shifts_t.size(0)):
        sh = shifts_t[s] @ cell_t
        for i0 in range(0, N, CHUNK):
            i1 = min(i0 + CHUNK, N)
            diff = pos_t[None, :, :] + sh.view(1, 1, 3) - pos_t[i0:i1, None, :]
            d = diff.norm(dim=-1)
            mask = d <= cutoff
            if s == zero_sid:
                mask[torch.arange(i1 - i0, device=device),
                torch.arange(i0, i1, device=device)] = False
            rows, cols = mask.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            src_l.append(rows + i0)
            dst_l.append(cols)
            sid_l.append(torch.full((rows.numel(),), s, dtype=torch.long, device=device))
            dist_l.append(d[rows, cols])
    empty_info = {'periodic': periodic, 'max_degree': 0, 'est_triplets': 0,
                  'min_dist': float('inf')}
    empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
    if not src_l:
        return (empty_ei, torch.zeros(0, 3, dtype=torch.long, device=device),
                torch.zeros(0, device=device), empty_info)
    src = torch.cat(src_l);
    dst = torch.cat(dst_l)
    sid = torch.cat(sid_l);
    dist = torch.cat(dist_l)
    deg = torch.bincount(src, minlength=N)
    max_deg = int(deg.max().item())
    if max_deg > max_num_neighbors:  # A2
        raise RuntimeError(
            f"node degree {max_deg} exceeds max_num_neighbors={max_num_neighbors}.\n"
            f"Typical full-PBC degrees for close-packed metals: CUTOFF=5.0 -> "
            f"~60-80 (set 160), "
            f"CUTOFF=6.0 -> ~110-140 (set 192).\n"
            f"If the observed degree is far outside this range, check the "
            f"near-coincident statistics or for a malformed lattice.")
    if dist.numel():
        min_d = float(dist.min().item())
    else:
        min_d = float('inf')
    info = {'periodic': periodic, 'max_degree': max_deg,
            'est_triplets': int((deg.float() ** 2).sum().item()),
            'min_dist': min_d}
    edge_index = torch.stack([src, dst], dim=0)
    edge_shift = torch.tensor(shifts_np, dtype=torch.long, device=device)[sid]
    return edge_index, edge_shift, dist, info


# ===================== hops / rf(A6) =====================
def compute_hops_and_rf(tags, edge_index_np, num_nodes, max_hops=MAX_HOPS):
    ads_mask = (tags == 2)
    if not ads_mask.any():
        return (np.full(num_nodes, max_hops, dtype=np.int64),
                np.zeros(num_nodes, dtype=np.float32))
    adj = defaultdict(list)
    for s, d_ in zip(edge_index_np[0].tolist(), edge_index_np[1].tolist()):
        adj[s].append(d_)
    hops = np.full(num_nodes, max_hops, dtype=np.int64)
    hops[ads_mask] = 0
    frontier = np.where(ads_mask)[0].tolist()
    for step in range(1, max_hops):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if hops[v] > step:
                    hops[v] = step
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    rf = np.clip(1.0 - hops / max_hops, 0.0, 1.0).astype(np.float32)
    return hops, rf


# ---------------------- quadruplet enumeration ----------------------
def compute_quadruplets(tags, edge_index, dist, num_nodes,
                        r_quad=R_QUAD, max_quads=MAX_QUADS_PER_ADS,
                        stats=None):
    """Enumerate chains c-j-k-l around each adsorbate atom c inside R_QUAD.

    Only shortest-image edges take part in the enumeration, which keeps the
    combinatorics bounded. Above the per-adsorbate cap the closest quadruplets
    are kept, ordered by max(d(c,j), d(c,k), d(c,l)), because those are the
    least attenuated by the envelope.

    Returns quad_edges (M, 3) = [e_ji, e_kj, e_lk] and quad_dists (M, 3).
    """
    M = edge_index.size(1)
    empty_e = np.zeros((0, 3), dtype=np.int64)
    empty_d = np.zeros((0, 3), dtype=np.float32)
    ads_idx = np.where(tags == 2)[0]
    if len(ads_idx) == 0 or M == 0:
        if stats is not None:
            stats.setdefault('quad_per_ads', []).extend([0] * len(ads_idx))
        return empty_e, empty_d
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    d_np = dist.cpu().numpy()
    best = {}
    for e in range(M):
        key = (int(src[e]), int(dst[e]))
        cur = best.get(key)
        if cur is None or d_np[e] < cur[0]:
            best[key] = (d_np[e], e)
    in_best = [dict() for _ in range(num_nodes)]
    for (a, b), (dd, e) in best.items():
        in_best[b][a] = (dd, e)
    quad_edges, quad_dists = [], []
    capped = 0
    per_ads = []
    for c in ads_idx:
        ball = {x: (dd, e) for x, (dd, e) in in_best[c].items()
                if dd <= r_quad and x != c}
        if not ball:
            per_ads.append(0)
            continue
        ball_set = set(ball.keys())
        nbrs = {}
        for x in ball:
            nbrs[x] = [(y, e) for y, (_, e) in in_best[x].items()
                       if y in ball_set]
        cq_e, cq_d, cq_key = [], [], []
        for j, (dj, e_ji) in ball.items():
            for k, e_kj in nbrs[j]:
                if k == j:
                    continue
                dk = ball[k][0]
                for l, e_lk in nbrs[k]:
                    if l == j or l == k:
                        continue
                    dl = ball[l][0]
                    cq_e.append((e_ji, e_kj, e_lk))
                    cq_d.append((dj, dk, dl))
                    cq_key.append(max(dj, dk, dl))
        nq = len(cq_e)
        if nq > max_quads:
            order = np.argsort(np.asarray(cq_key), kind='stable')[:max_quads]
            cq_e = [cq_e[i] for i in order]
            cq_d = [cq_d[i] for i in order]
            nq = max_quads
            capped += 1
        per_ads.append(nq)
        quad_edges.extend(cq_e)
        quad_dists.extend(cq_d)
    if stats is not None:
        stats.setdefault('quad_per_ads', []).extend(per_ads)
        stats['quad_capped'] = stats.get('quad_capped', 0) + capped
        stats['ads_total'] = stats.get('ads_total', 0) + len(ads_idx)
    if not quad_edges:
        return empty_e, empty_d
    return (np.asarray(quad_edges, dtype=np.int64),
            np.asarray(quad_dists, dtype=np.float32))


# ---------------------------- processor ----------------------------
class MultiTaskGraphProcessor:
    # ---- separator regex, full-width and half-width tolerant ----
    # The separator token is followed by either a full-width (U+FF1A) or a
    # half-width colon. The punctuation is written as \u escapes so that an
    # editor cannot silently normalise it.
    _SEP_LINE_RE = re.compile(r'\u8868\u9762[\uff1a:][^\n]*\n')  # whole line
    _SEP_ANY_RE = re.compile(r'\u8868\u9762[\uff1a:]')  # fallback, no newline
    # character classes for full-width and half-width colons and brackets
    _C = r'[\uff1a:]'
    _LP = r'[\uff08(]'
    _RP = r'[)\uff09]'

    def __init__(self, coord_path, save_dir):
        self.coord_path = coord_path
        self.save_dir = save_dir
        self.stats = {}

    def _bump(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n

    @staticmethod
    def _extract_float(text, pattern):
        m = re.search(pattern, text)
        if m is None:
            return float('nan')  # A5
        try:
            return float(m.group(1))
        except ValueError:
            return float('nan')

    def _split_bulk_surface(self, block_text):
        """Split a block into its bulk and surface sections.

        Tolerates full-width and half-width colons, extra text on the
        separator line, trailing whitespace and a missing final newline.
        """
        m = self._SEP_LINE_RE.search(block_text)
        if m:
            return block_text[:m.start()], block_text[m.end():], True
        m = self._SEP_ANY_RE.search(block_text)
        if m:
            rest = block_text[m.end():]
            nl = rest.find('\n')
            surf = '' if nl < 0 else rest[nl + 1:]
            return block_text[:m.start()], surf, True
        return block_text, '', False

    def _parse_slab(self, slab_text):
        lines = slab_text.strip().split('\n')
        if not lines:
            return None
        header = lines[0].split(',')
        ref_energy = float('nan')
        if len(header) >= 4:
            try:
                ref_energy = float(header[3].strip())
            except ValueError:
                pass
        fm = re.search(r'frame(\d+)', lines[0])
        frame_num = int(fm.group(1)) if fm else -1
        random_id = header[0].strip() if header else ''
        info_line = lines[1] if len(lines) > 1 else ''
        mm = re.search(r'miller=\(([^)]+)\)', info_line)
        miller_index = mm.group(1) if mm else ''
        total_energy = float('nan')
        cell = np.zeros((3, 3), dtype=np.float64)
        lat_line = next((l for l in lines if 'Lattice=' in l), None)
        if lat_line:
            em = re.search(r'energy=([-\d.eE+-]+)', lat_line)
            if em:
                try:
                    total_energy = float(em.group(1))
                except ValueError:
                    pass
            lm = re.search(r'Lattice="([^"]+)"', lat_line)
            if lm:
                vals = lm.group(1).split()
                if len(vals) == 9:
                    try:
                        cell = np.array(vals, dtype=np.float64).reshape(3, 3)
                    except ValueError:
                        pass
        atomic_numbers, coords, tags, forces = [], [], [], []
        atom_start = False
        for line in lines:
            if 'Lattice=' in line:
                atom_start = True
                continue
            if not atom_start or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:  # D1
                Z = Element(parts[0]).Z
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                tag_val = int(parts[5])
                fx, fy, fz = float(parts[6]), float(parts[7]), float(parts[8])
            except (ValueError, IndexError, KeyError):
                self._bump('atom_parse_fail')
                continue
            atomic_numbers.append(Z)
            coords.append([x, y, z])
            tags.append(tag_val)
            forces.append([fx, fy, fz])
        if not atomic_numbers:
            return None
        return {
            'ref_energy': ref_energy, 'total_energy': total_energy,
            'cell': cell, 'frame': frame_num, 'random_id': random_id,
            'miller_index': miller_index, 'atomic_numbers': atomic_numbers,
            'coords': coords, 'tags': tags, 'forces': forces,
        }

    def _build_graph(self, parsed):
        positions = np.asarray(parsed['coords'], dtype=np.float32)
        tags_arr = np.asarray(parsed['tags'], dtype=np.int64)
        forces_arr = np.asarray(parsed['forces'], dtype=np.float32)
        num_nodes = len(positions)
        if num_nodes == 0:
            return None
        pos_dev = torch.as_tensor(positions, device=DEVICE)
        edge_index, edge_shift, dist_t, info = build_neighbor_list(
            pos_dev, parsed['cell'], CUTOFF, MAX_NUM_NEIGHBORS, DEVICE)
        if edge_index.size(1) == 0:
            self._bump('empty_graph')
            return None
        self.stats.setdefault('deg_list', []).append(info['max_degree'])
        self.stats.setdefault('trip_list', []).append(info['est_triplets'])
        self.stats.setdefault('edge_list', []).append(int(edge_index.size(1)))
        if not info['periodic']:
            self._bump('non_periodic_cell')
        if info['min_dist'] < NEAR_COINCIDENT_THR:
            self._bump('near_coincident_graphs')
            self.stats['min_dist_seen'] = min(
                self.stats.get('min_dist_seen', float('inf')), info['min_dist'])
        edge_index = edge_index.cpu()
        edge_shift = edge_shift.cpu()
        dist_cpu = dist_t.cpu()
        hops, rf = compute_hops_and_rf(tags_arr, edge_index.numpy(), num_nodes)
        quad_edges, quad_dists = compute_quadruplets(
            tags_arr, edge_index, dist_cpu, num_nodes, stats=self.stats)
        ref, tot = parsed['ref_energy'], parsed['total_energy']
        if math.isfinite(ref) and math.isfinite(tot):
            ads_e = tot - ref
        else:
            ads_e = float('nan')
            self._bump('nan_ads_e')
        y_values = [ads_e] + [parsed.get(f'_bulk_{n}', float('nan'))
                              for n in TASK_NAMES[1:]]
        data = SurfaceData(
            z=torch.as_tensor(parsed['atomic_numbers'], dtype=torch.long),
            pos=torch.as_tensor(positions),
            tag=torch.as_tensor(parsed['tags'], dtype=torch.long),
            force=torch.as_tensor(forces_arr),
            cell=torch.as_tensor(parsed['cell'], dtype=torch.float32),
            edge_index=edge_index,
            edge_shift=edge_shift,
            hops_to_adsorbate=torch.as_tensor(hops, dtype=torch.long),
            rf_weights=torch.as_tensor(rf, dtype=torch.float32),
            quad_edges=torch.as_tensor(quad_edges, dtype=torch.long),
            quad_dists=torch.as_tensor(quad_dists, dtype=torch.float32),
            y=torch.as_tensor(y_values, dtype=torch.float32),
            sample_weight=torch.tensor([parsed['sample_weight']],
                                       dtype=torch.float32),
        )
        data.material_id = parsed['material_id']
        data.frame = parsed['frame']
        data.miller_index = parsed['miller_index']
        data.random_id = parsed['random_id']
        return data

    def _process_bulk_block(self, block_text):
        self._bump('blocks_seen')
        # robust split, with per-stage funnel counters
        bulk_part, surface_part, sep_found = self._split_bulk_surface(block_text)
        if not sep_found:
            self._bump('blocks_no_sep')
            if self.stats['blocks_no_sep'] <= MAX_FAIL_SNIPPETS:
                print(f"\n[WARN] no surface separator in block, "
                      f"first 200 chars: {repr(block_text[:200])}\n")
            return
        if not surface_part.strip():
            self._bump('blocks_sep_no_slabs')
            if self.stats['blocks_sep_no_slabs'] <= MAX_FAIL_SNIPPETS:
                print(f"\n[WARN] nothing after the separator, "
                      f"first 200 chars: {repr(block_text[:200])}\n")
        # label regexes accept full-width and half-width punctuation
        mpid_m = re.search(r'\u6750\u6599ID\s*' + self._C + r'\s*(mp-\d+)',
                           bulk_part)  # material id
        material_id = mpid_m.group(1) if mpid_m else 'unknown'
        if material_id == 'unknown':
            self._bump('material_unknown')
        bulk_labels = {
            'band_gap': self._extract_float(  # band gap
                bulk_part, r'\u5e26\u9699\s*' + self._LP + r'\s*eV\s*' + self._RP +
                           r'\s*' + self._C + r'\s*([-\d.eE+-]+)'),
            'formation_energy': self._extract_float(  # formation energy
                bulk_part, r'\u5f62\u6210\u80fd\s*' + self._LP + r'\s*eV/atom\s*' +
                           self._RP + r'\s*' + self._C + r'\s*([-\d.eE+-]+)'),
            'energy_per_atom': self._extract_float(  # energy per atom
                bulk_part, r'\u80fd\u91cf\s*' + self._LP + r'\s*eV/atom\s*' +
                           self._RP + r'\s*' + self._C + r'\s*([-\d.eE+-]+)'),
            'fermi_level': self._extract_float(  # Fermi level
                bulk_part, r'\u8d39\u7c73\u80fd\u7ea7\s*' + self._C +
                           r'\s*([-\d.eE+-]+)'),
        }
        for k, v in bulk_labels.items():
            if not math.isfinite(v):
                self._bump(f'label_nan::{k}')
        src_m = re.search(r'\u6765\u6e90\s*' + self._C + r'\s*(\S+)',
                          bulk_part)  # data source
        # source label as written in the dump (theory by default)
        source = src_m.group(1) if src_m else '理论预测'
        # upweight experimentally synthesised substrates
        sample_weight = 1.5 if source == '实验合成' else 1.0
        n_slab = 0
        for slab_text in surface_part.split('### '):
            if not slab_text.strip():
                continue
            n_slab += 1
            self._bump('slabs_seen')
            parsed = self._parse_slab(slab_text)
            if parsed is None:
                self._bump('slab_parse_fail')
                if self.stats['slab_parse_fail'] <= MAX_FAIL_SNIPPETS:
                    print(f"\n[WARN] slab parse failed, "
                          f"first 200 chars: {repr(slab_text[:200])}\n")
                continue
            parsed.update({
                'material_id': material_id,
                'sample_weight': sample_weight,
                **{f'_bulk_{k}': v for k, v in bulk_labels.items()},
            })
            data = self._build_graph(parsed)
            if data is not None:
                yield data
        if n_slab == 0 and surface_part.strip():
            self._bump('blocks_sep_no_slabs')

    # ---------------- v4: Preflight ----------------
    def _preflight(self, blocks):
        """Validate the first block before processing the whole file."""
        if not blocks:
            return
        b = blocks[0]
        _, surface_part, ok = self._split_bulk_surface(b)
        if not ok:
            print("[Preflight] FAIL: no surface separator in the first block")
            print(f"[Preflight] first 200 chars: {repr(b[:200])}")
            return
        n_slabs = len([s for s in surface_part.split('### ') if s.strip()])
        print(f"[Preflight] OK: separator found, first block has {n_slabs} slabs")
        for slab_text in surface_part.split('### '):
            if not slab_text.strip():
                continue
            parsed = self._parse_slab(slab_text)
            if parsed is None:
                print(f"[Preflight] FAIL: first slab did not parse: "
                      f"{repr(slab_text[:200])}")
            else:
                print(f"[Preflight] OK: first slab parsed. "
                      f"(N={len(parsed['atomic_numbers'])}, "
                      f"frame={parsed['frame']})")
            break

    # ---------------- progress and summary ----------------
    def _progress_line(self, total):
        q = np.asarray(self.stats.get('quad_per_ads', []), dtype=np.float64)
        deg = np.asarray(self.stats.get('deg_list', []), dtype=np.float64)
        if q.size == 0:
            return
        capped_pct = 100.0 * self.stats.get('quad_capped', 0) / max(
            self.stats.get('ads_total', 1), 1)
        print(f"[Progress {total:>7d} graphs] "
              f"quads/ads p50={np.percentile(q, 50):.0f} "
              f"p95={np.percentile(q, 95):.0f} max={q.max():.0f} | "
              f"capped={capped_pct:.1f}% | "
              f"max_deg p50={np.percentile(deg, 50):.0f} "
              f"p95={np.percentile(deg, 95):.0f} max={deg.max():.0f}")

    def process_and_save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for tmp in glob.glob(os.path.join(self.save_dir, 'shard_*.pkl.tmp')):
            os.remove(tmp)  # drop leftovers from an interrupted run
        with open(self.coord_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('\r\n', '\n').replace('\r', '\n')
        blocks = [b for b in content.split('[BULK]\n')[1:] if b.strip()]
        if not blocks:
            print("[ERROR] no [BULK] block found; check the path and the format")
            sys.exit(1)  # never exit zero on empty output
        print(f"[Input] {len(blocks)} [BULK] blocks")
        self._preflight(blocks)
        shard_data, shard_idx, total = [], 0, 0
        for block_text in tqdm(blocks, desc="BULK blocks"):
            for data in self._process_bulk_block(block_text):
                shard_data.append(data)
                total += 1
                if total % PROGRESS_EVERY == 0:
                    self._progress_line(total)
                if len(shard_data) >= SHARD_SIZE:
                    self._flush_shard(shard_data, shard_idx)
                    shard_data, shard_idx = [], shard_idx + 1
                    gc.collect()
        if shard_data:
            self._flush_shard(shard_data, shard_idx)
            n_shards = shard_idx + 1
        else:
            n_shards = shard_idx
        for t in sorted(glob.glob(os.path.join(self.save_dir, 'shard_*.pkl.tmp'))):
            os.replace(t, t[:-len('.tmp')])
        for old in glob.glob(os.path.join(self.save_dir, 'shard_*.pkl')):
            m = re.search(r'shard_(\d+)\.pkl$', old)
            if m and int(m.group(1)) >= n_shards:
                os.remove(old)
        config = {
            'CUTOFF': CUTOFF, 'R_QUAD': R_QUAD,
            'MAX_QUADS_PER_ADS': MAX_QUADS_PER_ADS, 'MAX_HOPS': MAX_HOPS,
            'MAX_NUM_NEIGHBORS': MAX_NUM_NEIGHBORS,
            'NUM_RADIAL': NUM_RADIAL, 'NUM_SPHERICAL': NUM_SPHERICAL,
            'ENVELOPE_EXPONENT': ENVELOPE_EXPONENT,
        }
        with open(os.path.join(self.save_dir, 'preprocess_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        self._final_summary(n_shards, total)
        if total == 0:  # never exit zero on empty output
            print("[ERROR] 0 samples; use the [Parse] funnel above to locate the break")
            sys.exit(1)

    def _final_summary(self, n_shards, total):
        print("\n" + "=" * 60)
        print(f"[Done] {total} samples → {n_shards} shards")
        # parse funnel: a stage dropping to zero localises the failure
        blocks_seen = self.stats.get('blocks_seen', 0)
        print(f"[Parse] blocks={blocks_seen} | "
              f"no_sep={self.stats.get('blocks_no_sep', 0)} | "
              f"sep_no_slab={self.stats.get('blocks_sep_no_slabs', 0)} | "
              f"slabs={self.stats.get('slabs_seen', 0)} | "
              f"parse_fail={self.stats.get('slab_parse_fail', 0)} | "
              f"graph_fail={self.stats.get('empty_graph', 0)}")
        if self.stats.get('material_unknown'):
            print(f"[Labels] ⚠ {self.stats['material_unknown']}/{blocks_seen} "
                  f"block(s) without a material id; those samples get "
                  f"material_id='unknown', "
                  f"and the bulk split would merge them into one group!")
        for name in TASK_NAMES[1:]:
            n = self.stats.get(f'label_nan::{name}', 0)
            if n:
                print(f"[Labels] {name}: missing in {n}/{blocks_seen} blocks (NaN) "
                      f"- if this share is high, check the label regexes")
        deg = np.asarray(self.stats.get('deg_list', []), dtype=np.float64)
        trip = np.asarray(self.stats.get('trip_list', []), dtype=np.float64)
        edge = np.asarray(self.stats.get('edge_list', []), dtype=np.float64)
        q = np.asarray(self.stats.get('quad_per_ads', []), dtype=np.float64)
        if deg.size:
            print(f"[Graph] edges/graph: mean={edge.mean():.0f} max={edge.max():.0f} | "
                  f"max_degree: p50={np.percentile(deg, 50):.0f} "
                  f"p95={np.percentile(deg, 95):.0f} max={deg.max():.0f}")
        if trip.size:
            avg_t = trip.mean()
            print(f"[Cost]  est triplets/graph: mean={avg_t:.0f} "
                  f"p95={np.percentile(trip, 95):.0f} max={trip.max():.0f}")
            sug = int(max(1, min(32, 4_000_000 // max(avg_t, 1))))
            print(f"[Cost]  suggested BATCH_SIZE ~ {sug} "
                  f"(for ~4M triplets per batch; tune to the available memory)")
        if q.size:
            capped_pct = 100.0 * self.stats.get('quad_capped', 0) / max(
                self.stats.get('ads_total', 1), 1)
            print(f"[Quad]  quads/adsorbate: p50={np.percentile(q, 50):.0f} "
                  f"p95={np.percentile(q, 95):.0f} max={q.max():.0f} | "
                  f"capped={capped_pct:.2f}%")
            if capped_pct > 5.0:
                print("[Quad]  WARNING: high truncation rate; lower R_QUAD to 3.0 "
                      "and re-run the preprocessing")
        if self.stats.get('near_coincident_graphs', 0):
            print(f"[WARN] {self.stats['near_coincident_graphs']} graph(s) contain "
                  f"near-coincident atoms ("
                  f"(min_dist={self.stats.get('min_dist_seen', float('inf')):.3f} Å < "
                  f"{NEAR_COINCIDENT_THR} A); this inflates degrees and edge counts")
        if self.stats.get('non_periodic_cell', 0):
            print(f"[WARN] {self.stats['non_periodic_cell']} graph(s) with a missing "
                  f"or invalid cell, "
                  f"falling back to a non-periodic neighbour list")
        if self.stats.get('nan_ads_e', 0):
            print(f"[WARN] {self.stats['nan_ads_e']} sample(s) without an adsorption "
                  f"energy (NaN), "
                  f"filtered out during training")
        if self.stats.get('atom_parse_fail', 0):
            print(f"[WARN] {self.stats['atom_parse_fail']} atom line(s) skipped")
        try:
            disk = sum(os.path.getsize(
                os.path.join(self.save_dir, f)) for f in os.listdir(self.save_dir)
                       if f.startswith('shard_') and f.endswith('.pkl'))
            print(f"[Disk]  shards total = {disk / 1e9:.2f} GB")
        except OSError:
            pass
        print("=" * 60)

    def _flush_shard(self, data, idx):
        path = os.path.join(self.save_dir, f'shard_{idx:04d}.pkl.tmp')
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [shard {idx:04d}] {len(data)} samples → {path}")


if __name__ == "__main__":
    MultiTaskGraphProcessor(
        coord_path=os.getenv('AA_RAW_FILE',
                         os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))), 'data', 'raw',
                             'final.txt')),
        save_dir=os.getenv('AA_DATA_DIR',
                       os.path.join(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))), 'data')),
    ).process_and_save()