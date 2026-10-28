"""Training and evaluation for the adaptive model and the three baselines.

Selected with MODEL_TYPE; components are switched off through AB_EMB / AB_GATE /
AB_QBF / AB_DUAL, and the output directory through TRAIN_SUB. Normalisation
statistics come from the training split only, and the splits are cached under a
key that includes the seed, the task set and the split fractions.
"""

import _path  # noqa: F401 - makes models/ importable
import os, glob, pickle, random, math, hashlib
# directory of this file, so the repository can be relocated as a whole
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.data import Batch
from torch.utils.data import IterableDataset
from tqdm import tqdm
import surface_data  # noqa: F401 - registers SurfaceData for unpickling

# ------------------------- model selection -------------------------
MODEL_TYPE = 'adaptive'  # 'adaptive' | 'schnet' | 'dimenet_pp' | 'visnet'

BASELINE_KWARGS = {
    'schnet': dict(
        hidden_channels=128, num_filters=128, num_interactions=6,
        num_gaussians=50, cutoff=5.0, max_num_neighbors=160,
    ),
    'dimenet_pp': dict(
        hidden_channels=128, num_blocks=4, int_emb_size=64, basis_emb_size=8,
        out_emb_channels=256, num_radial=6, num_spherical=7,
        cutoff=5.0, max_num_neighbors=160,
    ),
    'visnet': dict(
        hidden_channels=128, num_layers=6, num_heads=8, num_rbf=32,
        cutoff=5.0, max_num_neighbors=160, lmax=1,
    ),
}

if MODEL_TYPE == 'adaptive':
    from adaptive_dimenet import AdaptiveDimeNetPP
    MODEL_NAME = "AdaptiveDimeNetPP"
else:
    MODEL_NAME = MODEL_TYPE
    from baselines import build_model

# ---------------------------- config ----------------------------
# All four components are enabled by default. For the ablation study each
# component is disabled through an environment variable; unset means enabled:
#   $env:AB_EMB='0'   → use_adaptive_embedding = False
#   $env:AB_GATE='0'  → use_distance_gating    = False
#   $env:AB_QBF='0'   → use_qbf                = False
#   $env:AB_DUAL='0'  → use_dual_stream        = False
def _flag(name, default_on=True):
    v = os.getenv(name)
    return default_on if v is None else (v.strip() not in ('0', 'false', 'False', ''))


ABLATION_FLAGS = {
    'use_adaptive_embedding': _flag('AB_EMB'),
    'use_distance_gating': _flag('AB_GATE'),
    'use_qbf': _flag('AB_QBF'),
    'use_dual_stream': _flag('AB_DUAL'),
}
LOAD_BEST = False
USE_FORCE = False
# Data provenance: the split cache in this directory
#   split_bulk_s168_adsorption_energy_vr0.2_tr0.2_e02ed433ec.pkl  (2026-10-10 18:41)
# is the product of the cache key in this script, and the shards date from 2026-10-08.
# A second directory with fingerprint e228725d27 holds a different snapshot and
# must not be mixed with this one.
DATA_DIR = os.getenv('AA_DATA_DIR', os.path.join(_ROOT, 'data'))
SPLIT_MODE = 'bulk'
TASK_NAMES = ["adsorption_energy"]
TASK_WEIGHTS = [1.0]
FORCE_WEIGHT = 0.1
FULL_TASK_LIST = ["adsorption_energy", "band_gap", "formation_energy",
                  "energy_per_atom", "fermi_level"]
FORCE_ENERGY_TASK_NAME = "energy_per_atom"
FORCE_ENERGY_TASK_IDX = (TASK_NAMES.index("energy_per_atom")
                         if "energy_per_atom" in TASK_NAMES else None)
MAX_TRIPLETS_PER_BATCH = 4_000_000
MAX_GRAPHS_PER_BATCH = 48
MAX_EPOCHS = 200
WEIGHT_DECAY = 1e-4
PATIENCE = 25
# The original checkpoints directory holds the final.pth quoted in the paper;
# a rerun would overwrite it, so the output goes to a separate directory.
SAVE_DIR = os.getenv('AA_RUN_DIR', os.path.join(_ROOT, 'runs'))
VAL_RATIO = 0.2
TEST_RATIO = 0.2
SEED = 168
FORCE_WEIGHT = 0.1
# NOTE: the learning rate and gradient clipping are declared by the model
# class (recommended_lr / recommended_grad_clip) and read after construction.
EXPECTED_PP_CONFIG = {
    'CUTOFF': 5.0, 'R_QUAD': 3.5, 'MAX_NUM_NEIGHBORS': 160,
    'NUM_RADIAL': 6, 'NUM_SPHERICAL': 7, 'ENVELOPE_EXPONENT': 5,
    'MAX_HOPS': 10,
}
# HEAD_WIDTH: hidden width of the readout head (default: hidden_channels).
# '64' halves the readout parameters for the head-capacity control experiment.
_HEAD_WIDTH = os.getenv('HEAD_WIDTH')
MODEL_KWARGS = dict(
    hidden_channels=128, num_blocks=4, int_emb_size=64, basis_emb_size=8,
    out_emb_channels=256, num_radial=6, num_spherical=7,
    cutoff=5.0, r_quad=3.5, max_num_neighbors=160, envelope_exponent=5,
    num_before_skip=1, num_after_skip=2, num_output_layers=3,
    num_dihedral_freqs=6, quad_emb_size=32,
    **({'head_width': int(_HEAD_WIDTH)} if _HEAD_WIDTH else {}),
)


# ---------------------------- helpers ----------------------------
def set_seed(seed):  # D2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flags_tag(flags):  # B3
    return (('E' if flags['use_adaptive_embedding'] else 'e')
            + ('G' if flags['use_distance_gating'] else 'g')
            + ('Q' if flags['use_qbf'] else 'q')
            + ('D' if flags['use_dual_stream'] else 'd'))


def validate_preprocess_config(data_dir, expected):
    path = os.path.join(data_dir, 'preprocess_config.pkl')
    if not os.path.exists(path):
        print("[WARN] preprocess_config.pkl not found, skipping the check")
        return
    with open(path, 'rb') as f:
        pp = pickle.load(f)
    missing = [k for k in expected if k not in pp]
    if missing:
        print(f"[ERROR] preprocess config lacks {missing}; this usually means "
              f"the data predates the current pipeline, "
              f"re-run preprocess.py")
        raise ValueError("incompatible or outdated preprocessed data")
    mismatches = [f"  {k}: train={v} vs preprocess={pp[k]}"
                  for k, v in expected.items() if pp[k] != v]
    if mismatches:
        print("[ERROR] preprocessing and training configurations differ:")
        print('\n'.join(mismatches))
        raise ValueError("config mismatch: re-run the preprocessing or update "
                         "the training configuration")
    print("[OK] preprocessing and training configurations agree")


def check_shard_schema(data_dir, shard_files, flags):
    """Check that the shards carry every field this model reads."""
    with open(shard_files[0], 'rb') as f:
        shard = pickle.load(f)
    d = next((x for x in shard if x is not None), None)
    if d is None:
        return
    required = ['edge_index', 'edge_shift', 'pos', 'z', 'tag', 'y']
    if flags.get('use_qbf', False):
        required += ['quad_edges', 'quad_dists']
    missing = [k for k in required if not hasattr(d, k)]
    if missing:
        raise RuntimeError(
            f"shard is missing {missing}; re-run preprocess.py")
    print("[OK] shard fields validated")


def _dataset_fingerprint(shard_files):
    """Data fingerprint: any re-preprocessing invalidates the split caches."""
    h = hashlib.md5()
    for sf in shard_files:
        st = os.stat(sf)
        h.update(f"{os.path.basename(sf)}|{st.st_size}|{int(st.st_mtime)};".encode())
    return h.hexdigest()[:10]


# ===================== Split / Stats(A4 / E3) =====================
def _get_group_key(data, mode='bulk'):
    if mode == 'site':
        return f"{getattr(data, 'material_id', 'unknown')}_{getattr(data, 'random_id', 'unknown')}"
    elif mode == 'miller':
        return f"{getattr(data, 'material_id', '?')}_{getattr(data, 'miller_index', '?')}"
    elif mode == 'bulk':
        return str(getattr(data, 'material_id', 'unknown'))
    return str(getattr(data, 'material_id', 'unknown'))


def ensure_split_index(data_dir, shard_files, split_mode, task_names,
                       full_task_list, val_ratio, test_ratio, seed):
    fp = _dataset_fingerprint(shard_files)
    # The cache key covers the seed, the task set and the split fractions, so
    # changing any of them cannot silently reuse a stale split or y_mean/y_std.
    key_sig = (f"{split_mode}_s{seed}_"
               f"{'-'.join(sorted(task_names))}_"
               f"vr{val_ratio}_tr{test_ratio}_{fp}")
    split_path = os.path.join(data_dir, f'split_{key_sig}.pkl')
    stats_path = os.path.join(data_dir, f'ystats_{key_sig}.pt')
    if os.path.exists(split_path) and os.path.exists(stats_path):
        print(f"[INFO] using cached split/stats (key={key_sig})")
        with open(split_path, 'rb') as f:
            split_map = pickle.load(f)
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        return split_path, stats
    print(f"[INFO] building split (mode={split_mode}, tasks={task_names}, "
          f"key={key_sig})")
    task_idx_map = {n: full_task_list.index(n) for n in task_names}
    group_to_samples = defaultdict(list)
    sample_y = {}
    for s_idx, sf in enumerate(tqdm(shard_files, desc="scanning shards")):
        with open(sf, 'rb') as f:
            shard = pickle.load(f)
        for d_idx, d in enumerate(shard):
            if d is None:
                continue
            key = f"{s_idx}_{d_idx}"
            gkey = key if split_mode == 'frame' else _get_group_key(d, split_mode)
            group_to_samples[gkey].append(key)
            sample_y[key] = [float(d.y[task_idx_map[n]]) for n in task_names]
        del shard
    groups = sorted(group_to_samples.keys())
    np.random.seed(seed)
    np.random.shuffle(groups)
    n = len(groups)
    nv, nt = int(n * val_ratio), int(n * test_ratio)
    val_g, test_g = set(groups[:nv]), set(groups[nv:nv + nt])
    split_map = {}
    for g in groups:
        sp = 'val' if g in val_g else ('test' if g in test_g else 'train')
        for k in group_to_samples[g]:
            split_map[k] = sp
    y_mean_list, y_std_list = [], []
    for t_i, t_name in enumerate(task_names):
        vals = [sample_y[k][t_i] for k, sp in split_map.items()
                if sp == 'train' and math.isfinite(sample_y[k][t_i])]
        assert vals, f"task '{t_name}' has no valid label in the train split"
        m, s = float(np.mean(vals)), max(float(np.std(vals)), 1e-8)
        y_mean_list.append(m)
        y_std_list.append(s)
        print(f"  [{t_name}] train-only: mean={m:.4f}, std={s:.4f}, n={len(vals)}")
    stats = {'y_mean': torch.tensor(y_mean_list, dtype=torch.float32),
             'y_std': torch.tensor(y_std_list, dtype=torch.float32)}
    with open(split_path, 'wb') as f:
        pickle.dump(split_map, f)
    torch.save(stats, stats_path)
    cnt = Counter(split_map.values())
    print(f"[Split:{split_mode}] groups={n} | "
          f"train={cnt['train']} val={cnt['val']} test={cnt['test']}")
    return split_path, stats


def _est_triplets(d):
    """Estimate triplets per graph as sum_v deg(v)^2, the order of triplets()."""
    if d.edge_index is None or d.edge_index.numel() == 0:
        return 0
    deg = torch.bincount(d.edge_index[0], minlength=d.num_nodes)
    return int((deg.float() ** 2).sum().item())


# ---------------------------- dataset ----------------------------
class SplitAwareShardDataset(IterableDataset):
    """num_workers must stay 0: an IterableDataset is not sharded automatically,
    so every worker would load the whole shard itself."""

    def __init__(self, shard_files, split_index_path, split,
                 task_names, full_task_list, shuffle=True):
        super().__init__()
        self.shard_files = list(shard_files)
        self.split = split
        self.shuffle = shuffle
        self.task_idx = [full_task_list.index(n) for n in task_names]
        with open(split_index_path, 'rb') as f:
            full_map = pickle.load(f)
        self.shard_sample_map = defaultdict(list)
        for key, sp in full_map.items():
            if sp == split:
                s, d = key.split('_')
                self.shard_sample_map[int(s)].append(int(d))
        del full_map
        total = sum(len(v) for v in self.shard_sample_map.values())
        print(f"[{split:>5}] {total:>6} samples "
              f"across {len(self.shard_sample_map)} shards")

    def __iter__(self):
        self._buf, self._buf_cost = [], 0
        indices = list(self.shard_sample_map.keys())
        if self.shuffle:
            random.shuffle(indices)
        for s_idx in indices:
            with open(self.shard_files[s_idx], 'rb') as f:
                shard = pickle.load(f)
            d_indices = list(self.shard_sample_map[s_idx])
            if self.shuffle:
                random.shuffle(d_indices)
            for d_idx in d_indices:
                d = shard[d_idx]
                if d is None:
                    continue
                y_sub = d.y[self.task_idx]
                if torch.isnan(y_sub).any():
                    continue
                dc = d.clone()
                dc.y = y_sub
                dc.material_id = getattr(d, 'material_id', None)
                dc.frame = getattr(d, 'frame', None)
                dc.miller_index = getattr(d, 'miller_index', None)
                dc.random_id = getattr(d, 'random_id', None)
                cost = _est_triplets(dc)
                if self._buf and (self._buf_cost + cost > MAX_TRIPLETS_PER_BATCH
                                  or len(self._buf) >= MAX_GRAPHS_PER_BATCH):
                    yield Batch.from_data_list(self._buf)
                    self._buf, self._buf_cost = [], 0
                self._buf.append(dc)
                self._buf_cost += cost
            del shard
        if self._buf:
            yield Batch.from_data_list(self._buf)
            self._buf, self._buf_cost = [], 0

    def __len__(self):
        return sum(len(v) for v in self.shard_sample_map.values())


# ===================== EarlyStopping / Loss =====================
class EarlyStopping:
    def __init__(self, patience=25, delta=1e-4):
        self.patience, self.delta = patience, delta
        self.best, self.counter, self.best_state = float('inf'), 0, None

    def __call__(self, metric, model):
        if metric < self.best - self.delta:
            self.best = metric
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_joint_loss(model, energy_pred, energy_target,
                       pos, batch_vec, force_true, sample_weight,
                       y_mean, y_std):
    log_vars = model.log_vars.clamp(min=-6.0, max=4.0)
    precision_e = torch.exp(-log_vars)
    e_loss_raw = F.smooth_l1_loss(energy_pred, energy_target, reduction='none')
    weighted_e = precision_e * e_loss_raw * model.task_priority.unsqueeze(0)
    w_e_loss = 0.5 * weighted_e.sum(dim=-1)
    sw = sample_weight.view(-1)
    reg_e = 0.5 * log_vars.sum()
    if not USE_FORCE:
        total_loss = (sw * w_e_loss).mean() + reg_e
        return total_loss, {'e_loss': e_loss_raw[:, 0].mean().item(), 'f_loss': 0.0}
    assert pos.requires_grad, "pos must require grad when USE_FORCE is set"
    e_phys = energy_pred * y_std + y_mean
    n_atoms = scatter(torch.ones(pos.size(0), device=pos.device),
                      batch_vec, dim=0, reduce='sum')
    total_energy = (e_phys[:, FORCE_ENERGY_TASK_IDX] * n_atoms).sum()
    force_pred = -torch.autograd.grad(
        total_energy, pos, retain_graph=True, create_graph=True,
        only_inputs=True)[0]
    f_loss_raw = F.mse_loss(force_pred, force_true, reduction='none').mean(dim=-1)
    f_count = scatter(torch.ones_like(f_loss_raw), batch_vec, dim=0,
                      reduce='sum').clamp(min=1)
    w_f_loss = scatter(f_loss_raw, batch_vec, dim=0, reduce='sum') / f_count
    total_loss = (sw * (w_e_loss + FORCE_WEIGHT * w_f_loss)).mean() + reg_e
    return total_loss, {'e_loss': e_loss_raw[:, 0].mean().item(),
                        'f_loss': f_loss_raw.mean().item()}


def compute_energy_loss_baseline(energy_pred, energy_target, sample_weight):
    e_loss = F.smooth_l1_loss(energy_pred, energy_target,
                              reduction='none').mean(dim=-1)
    sw = sample_weight.view(-1)
    return (sw * e_loss).mean(), {'e_loss': e_loss.mean().item(), 'f_loss': 0.0}


def combined_metric(e_metrics, f_metrics, priority):
    if not e_metrics:
        return float('inf')
    names = list(e_metrics.keys())
    maes = torch.tensor([e_metrics[n] for n in names], dtype=torch.float32)
    w = priority[:len(names)].detach().cpu()
    w = w / w.sum()
    m = float((w * maes).sum())
    if USE_FORCE:
        fm = f_metrics.get('force_mae', float('nan'))
        if fm == fm:
            m += FORCE_WEIGHT * fm
    return m


# ===================== Train / Eval (E2 / E4) =====================
def train_epoch(model, loader, optimizer, y_mean, y_std, device,
                epoch=0, use_uncertainty_loss=False,
                grad_clip=10.0, finite_check=False):
    """One training step, with non-finite losses and gradients skipped.

    The clipping value and the finite check are passed in from main(), which
    reads them from the model.
    """
    model.train()
    total_loss, n = 0.0, 0
    n_skipped = 0
    log_accum = defaultdict(float)
    for batch in tqdm(iter(loader), desc=f"Train Ep{epoch + 1}", leave=True):
        batch = batch.to(device)
        if batch.y.dim() == 1:
            batch.y = batch.y.view(batch.num_graphs, -1)
        optimizer.zero_grad(set_to_none=True)
        if USE_FORCE:
            batch.pos.requires_grad_(True)
        energy_pred = model(batch)
        e_target = (batch.y - y_mean) / y_std
        if use_uncertainty_loss:
            loss, logs = compute_joint_loss(
                model, energy_pred, e_target, batch.pos, batch.batch,
                batch.force if USE_FORCE else None,
                batch.sample_weight, y_mean, y_std)
        else:
            loss, logs = compute_energy_loss_baseline(
                energy_pred, e_target, batch.sample_weight)
        # guard 1: skip the batch when the loss is not finite
        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            del loss, energy_pred, e_target, batch
            continue
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # guard 2: skip the batch when any gradient is not finite
        if finite_check and not torch.isfinite(grad_norm):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            del loss, energy_pred, e_target, batch
            continue
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                log_accum[k] += v * batch.num_graphs
        del loss, energy_pred, e_target, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if n_skipped > 0:
        print(f"  [WARN] {n_skipped} batches skipped (non-finite loss/grad)")
    avg = {k: v / max(n, 1) for k, v in log_accum.items()}
    return total_loss / max(n, 1), avg


def evaluate(model, loader, y_mean, y_std, device, desc="Eval"):
    """Energy metrics run under no_grad; forces need a second, differentiable
    forward pass when they are enabled."""
    model.eval()
    all_ep, all_et, all_fp, all_ft = [], [], [], []
    n = 0
    for batch in tqdm(iter(loader), desc=desc, leave=True):
        batch = batch.to(device)
        if batch.y.dim() == 1:
            batch.y = batch.y.view(batch.num_graphs, -1)
        with torch.no_grad():
            pred = model(batch)
            e_phys = pred * y_std + y_mean
            all_ep.append(e_phys.cpu())
            all_et.append(batch.y.cpu())
            n += batch.num_graphs
        if USE_FORCE:
            batch.pos.requires_grad_(True)
            with torch.enable_grad():
                pred_f = model(batch)
                e_phys_f = pred_f * y_std + y_mean
                nat = scatter(torch.ones(batch.pos.size(0), device=device),
                              batch.batch, dim=0, reduce='sum')
                total_e = (e_phys_f[:, FORCE_ENERGY_TASK_IDX] * nat).sum()
                f_pred = -torch.autograd.grad(
                    total_e, batch.pos, retain_graph=False,
                    create_graph=False, only_inputs=True)[0]
            all_fp.append(f_pred.detach().cpu())
            all_ft.append(batch.force.cpu())
        del batch, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if n == 0:
        return float('inf'), {}, {}
    ep, et = torch.cat(all_ep), torch.cat(all_et)
    mae = (ep - et).abs().mean(dim=0)
    e_metrics = {name: mae[i].item() for i, name in enumerate(TASK_NAMES)}
    if USE_FORCE and all_fp:
        fp, ft = torch.cat(all_fp), torch.cat(all_ft)
        f_metrics = {'force_mae': (fp - ft).abs().mean().item(),
                     'force_cos_sim': F.cosine_similarity(fp, ft, dim=-1).mean().item()}
    else:
        f_metrics = {'force_mae': float('nan'), 'force_cos_sim': float('nan')}
    metric = combined_metric(e_metrics, f_metrics, model.task_priority)
    return metric, e_metrics, f_metrics


def load_best_model(model, save_dir, device, y_mean, y_std):
    path = os.path.join(save_dir, 'best.pth')
    if not os.path.exists(path):
        print(f"[Load] best.pth not found: {path}")
        return False
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
        if 'y_mean' in state:
            y_mean.copy_(state['y_mean'].to(device))
        if 'y_std' in state:
            y_std.copy_(state['y_std'].to(device))
        print(f"[Load] ✓ best.pth | val_metric={state.get('best_val_metric', '?')}")
    else:
        model.load_state_dict(state)
        print("[Load] best.pth loaded (plain state_dict)")
    model.to(device)
    return True


# ===================== Main =====================
def main():
    set_seed(SEED)  # D2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Env] torch={torch.__version__}, "
          f"pyg={torch_geometric.__version__} (pin this for reproducibility)")
    print(f"★ Model: {MODEL_NAME}")
    print(f"★ Force: {'ON' if USE_FORCE else 'OFF'}")
    if MODEL_TYPE == 'adaptive':
        print(f"★ Ablation flags: {ABLATION_FLAGS}")
    if USE_FORCE:
        assert FORCE_ENERGY_TASK_IDX is not None
        print(f"★ Force energy task: '{FORCE_ENERGY_TASK_NAME}' "
              f"(idx={FORCE_ENERGY_TASK_IDX})")
    label_dim = len(TASK_NAMES)
    assert len(TASK_WEIGHTS) == label_dim
    validate_preprocess_config(DATA_DIR, EXPECTED_PP_CONFIG)
    shard_files = sorted(glob.glob(os.path.join(DATA_DIR, 'shard_*.pkl')))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {DATA_DIR}")
    print(f"[Shards] {len(shard_files)} files found")

    # baselines do not read the quadruplet fields
    if MODEL_TYPE == 'adaptive':
        check_shard_schema(DATA_DIR, shard_files, ABLATION_FLAGS)
    else:
        check_shard_schema(DATA_DIR, shard_files, {'use_qbf': False})

    split_path, stats = ensure_split_index(
        DATA_DIR, shard_files, SPLIT_MODE, TASK_NAMES, FULL_TASK_LIST,
        VAL_RATIO, TEST_RATIO, SEED)
    y_mean = stats['y_mean'].to(device)
    y_std = stats['y_std'].to(device)
    train_set = SplitAwareShardDataset(shard_files, split_path, 'train',
                                       TASK_NAMES, FULL_TASK_LIST, True)
    val_set = SplitAwareShardDataset(shard_files, split_path, 'val',
                                     TASK_NAMES, FULL_TASK_LIST, False)
    test_set = SplitAwareShardDataset(shard_files, split_path, 'test',
                                      TASK_NAMES, FULL_TASK_LIST, False)
    train_loader, val_loader, test_loader = train_set, val_set, test_set

    # ------------------------ model construction ------------------------
    if MODEL_TYPE == 'adaptive':
        model = AdaptiveDimeNetPP(
            num_tasks=label_dim,
            force_energy_task_idx=FORCE_ENERGY_TASK_IDX,
            task_weights=TASK_WEIGHTS,
            **MODEL_KWARGS, **ABLATION_FLAGS,
        ).to(device)
        used_model_kwargs = MODEL_KWARGS
        used_flags = ABLATION_FLAGS
    else:
        model = build_model(
            MODEL_TYPE,
            num_tasks=label_dim,
            force_energy_task_idx=FORCE_ENERGY_TASK_IDX,
            task_weights=TASK_WEIGHTS,
            **BASELINE_KWARGS[MODEL_TYPE],
        ).to(device)
        used_model_kwargs = BASELINE_KWARGS[MODEL_TYPE]
        used_flags = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # training hyper-parameters are read from the model
    LR = model.recommended_lr
    GRAD_CLIP = model.recommended_grad_clip
    finite_check = model.enable_grad_finite_check
    print(f"★ LR={LR:.1e} | GRAD_CLIP={GRAD_CLIP:.1f} | "
          f"grad_finite_check={'ON' if finite_check else 'OFF'}")

    model.set_target_stats(y_mean.cpu(), y_std.cpu())
    use_uncertainty_loss = (label_dim > 1)

    # one output directory per configuration
    # On case-insensitive file systems `ablation_EGQD`, `egqd` and `EGqD`
    # all resolve to the same directory, so any ablation would overwrite the
    # full model. TRAIN_SUB therefore overrides the sub-directory name:
    #   $env:TRAIN_SUB='wo_qbf'; python train.py
    _sub = os.getenv('TRAIN_SUB', '').strip()
    if _sub:
        save_dir = os.path.join(SAVE_DIR, _sub)
        print(f"[WARN] TRAIN_SUB overrides the sub-directory: {_sub}")
    elif MODEL_TYPE == 'adaptive':
        save_dir = os.path.join(SAVE_DIR, f"ablation_{flags_tag(ABLATION_FLAGS)}")
    else:
        save_dir = os.path.join(SAVE_DIR, f"baseline_{MODEL_TYPE}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"★ Save dir: {save_dir}")

    if LOAD_BEST:
        if not load_best_model(model, save_dir, device, y_mean, y_std):
            return
        model.set_target_stats(y_mean.cpu(), y_std.cpu())
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
        stopper = EarlyStopping(patience=PATIENCE)
        best_combined = float('inf')
        for epoch in range(MAX_EPOCHS):
            tr_loss, tr_logs = train_epoch(
                model, train_loader, optimizer, y_mean, y_std, device,
                epoch=epoch, use_uncertainty_loss=use_uncertainty_loss,
                grad_clip=GRAD_CLIP, finite_check=finite_check)
            # stop when the loss stays NaN for several epochs
            if not math.isfinite(tr_loss):
                print(f"[FATAL] training loss={tr_loss} at epoch {epoch+1}, stopping")
                break
            if torch.cuda.is_available():
                print(f"  [Mem] peak={torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
                torch.cuda.reset_peak_memory_stats()
            val_metric, val_e, val_f = evaluate(
                model, val_loader, y_mean, y_std, device,
                desc=f"Val Ep{epoch + 1}")
            if not math.isfinite(val_metric):
                print(f"[FATAL] validation metric={val_metric} at epoch {epoch+1}, stopping")
                break
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            e_str = " ".join(f"{k}={v:.4f}" for k, v in val_e.items())
            # only the adaptive model with use_qbf reads interaction_blocks
            qbf_str = ""
            if MODEL_TYPE == 'adaptive' and ABLATION_FLAGS['use_qbf']:
                qbf_str = (f" |qbf_s|="
                           f"{model.interaction_blocks[0].qbf_scale.abs().item():.4f}")
            f_str = (f" F_MAE={val_f['force_mae']:.4f} "
                     f"F_Cos={val_f['force_cos_sim']:.4f}"
                     if USE_FORCE else "")
            print(f"Ep {epoch + 1:03d} | TrL={tr_loss:.5f} "
                  f"VaM={val_metric:.5f} | {e_str}{f_str}{qbf_str} | LR={lr:.1e}")
            if val_metric < best_combined:
                best_combined = val_metric
                torch.save({
                    'model': model.state_dict(),
                    'best_val_metric': best_combined,
                    'val_e_mae': val_e, 'val_f_mae': val_f,
                    'y_mean': y_mean.cpu(), 'y_std': y_std.cpu(),
                    'model_type': MODEL_NAME, 'use_force': USE_FORCE,
                    'force_energy_task': FORCE_ENERGY_TASK_NAME,
                    'ablation_flags': used_flags,
                    'model_kwargs': used_model_kwargs,
                    'pp_config': EXPECTED_PP_CONFIG,
                }, os.path.join(save_dir, 'best.pth'))
                print(f"  ✓ Saved best (metric={best_combined:.4f})")
            if stopper(val_metric, model):
                print(f"\n⏹ Early stop @ epoch {epoch + 1} (patience={PATIENCE})")
                break
        if stopper.best_state:
            model.load_state_dict({k: v.to(device)
                                   for k, v in stopper.best_state.items()})
            print(f"Loaded best weights (metric={stopper.best:.4f})")
        # same guard for the baselines
        if MODEL_TYPE == 'adaptive' and ABLATION_FLAGS['use_qbf']:
            scales = [b.qbf_scale.abs().item()
                      for b in model.interaction_blocks]
            print(f"[QBF] |qbf_scale| per interaction block = "
                  f"{[f'{s:.4f}' for s in scales]} (near 0 means the branch is "
                  f"unused)")

    _, test_e, test_f = evaluate(model, test_loader, y_mean, y_std, device,
                                 desc="Test")
    print("\n" + "=" * 50)
    print("=== Test Results ===")
    print("=" * 50)
    if not test_e:  # D9
        print("  [WARN] test split is empty, no metrics to report")
    else:
        for name in TASK_NAMES:
            print(f"  {name}: MAE = {test_e[name]:.4f}")
    if USE_FORCE:
        print(f"  force: MAE = {test_f['force_mae']:.4f} eV/Å | "
              f"CosSim = {test_f['force_cos_sim']:.4f}")
    else:
        print("  force: DISABLED")
    print("=" * 50)
    torch.save({
        'model': model.state_dict(),
        'test_e_mae': test_e, 'test_f_mae': test_f,
        'y_mean': y_mean.cpu(), 'y_std': y_std.cpu(),
        'model_type': MODEL_NAME,
        'ablation_flags': used_flags,
        'model_kwargs': used_model_kwargs,
        'pp_config': EXPECTED_PP_CONFIG,
    }, os.path.join(save_dir, 'final.pth'))
    print(f"\nSaved to {save_dir}/final.pth")


if __name__ == '__main__':
    main()