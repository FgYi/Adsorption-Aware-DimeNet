# Adsorption-Aware DimeNet++

Adsorption-energy prediction for ammonia (NH3) on metal-oxide surfaces with a
DimeNet++ backbone extended by adsorbate-relative inputs, adsorbate-centred
four-body features and a dual-stream readout.

This repository contains the code used for the experiments in the accompanying
manuscript, including the full data pipeline, the training script for the
proposed model and three baselines (SchNet, DimeNet++, ViSNet), the gate
analysis tooling, and the ablation runner.

---

## Results

Material-disjoint (`bulk`) split, single-task training, validation MAE in eV
(lower is better). All four models share the same data, batch construction,
optimizer, schedule and early-stopping rule.

| Model      | Params | Val MAE (eV) |
|------------|--------|--------------|
| SchNet     | 0.60 M | 0.5450       |
| DimeNet++  | 2.07 M | 0.4725       |
| ViSNet     | 1.74 M | 0.4895       |
| **Ours**   | 2.38 M | **0.4123**   |

Held-out test MAE of the selected snapshot: **0.4263 eV**. Two independent runs
gave 0.4262 and 0.4263 eV, i.e. the test error is reproducible to below 1 meV.

The distance gate is analysed in `scripts/analyze_gate.py`: feature retention
stays within 0.648–0.682 over the whole test split, and after conditioning on
element and coordination the residual hop dependence is a first-shell
enhancement of +0.028 with no monotone decay. In other words the gate provides
an approximately uniform attenuation rather than a distance-selective receptive
field, and the accuracy originates in the adsorbate-relative embedding, the
four-body basis and the dual-stream readout.

---

## Repository layout

```
.
├── models/
│   ├── adaptive_dimenet.py   # proposed architecture
│   ├── baselines.py          # SchNet / DimeNet++ / ViSNet with PBC support
│   └── surface_data.py       # SurfaceData, TaskSpec, TargetScaler
├── scripts/
│   ├── preprocess.py         # raw dump -> sharded pickles (PBC edges, quads)
│   ├── train.py              # training / evaluation (adaptive + baselines)
│   ├── analyze_gate.py       # gate statistics and receptive-field analysis
│   ├── run_ablations.py      # leave-one-out ablation runner
│   ├── run_all.py            # one-command pipeline (train -> analyse -> ablate)
│   ├── count_params.py       # per-module parameter breakdown
│   ├── smoke_test.py         # fast sanity check of the three baselines
│   └── parity_visnet.py      # numerical parity check of the ViSNet re-implementation
├── data/                     # NOT tracked (see data/README.md)
└── runs/                     # NOT tracked (checkpoints, logs)
```

All scripts are run from the repository root and only need the repository plus a
`data/` directory; paths can be overridden with environment variables (below).

---

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# PyTorch matching your CUDA driver, e.g. CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Tested with `torch==2.11.0+cu128`, `torch-geometric==2.8.0.post1`,
`pymatgen`, `numpy`, `tqdm` on a single RTX 5090 (32 GB).

---

## Data

The dataset is not distributed with this repository. `data/README.md` documents
the expected layout and the exact field schema of the sharded pickles so that
compatible data can be built with `scripts/preprocess.py`.

```bash
export AA_RAW_FILE=/path/to/final.txt     # raw coordinate dump
export AA_DATA_DIR=/path/to/data          # output directory
python scripts/preprocess.py
```

Preprocessing enumerates periodic images, builds PBC neighbour lists
(cutoff 5.0 Å, at most 160 neighbours), extracts DimeNet++ triplets, enumerates
adsorbate-centred quadruplets (radius 3.5 Å, at most 4096 per adsorbate,
shortest-image de-duplication) and computes hop distances and the continuous
distance prior `rf = 1 - h/10`.

---

## Quick start

```bash
# 1) sanity check: does the data load, do the baselines run?
python scripts/smoke_test.py
python scripts/parity_visnet.py

# 2) train the proposed model on the bulk split (~13 h on one RTX 5090)
python scripts/train.py

# 3) analyse the gate of the trained model (~10 min)
python scripts/analyze_gate.py --ckpt runs/ablation_EGQD/best.pth \
    --max-graphs 1500 --save-json rf.json

# 4) ablation and baseline parity runs
python scripts/run_ablations.py --dry-run
python scripts/run_ablations.py --yes
```

`scripts/run_all.py` chains steps 1-3 (and optionally the ablations) with
resume support:

```bash
python scripts/run_all.py --dry-run
python scripts/run_all.py --yes
```

---

## Configuration

Everything is configured by editing the constants at the top of
`scripts/train.py`, or by environment variables where noted.

| Variable | Meaning | Default |
|---|---|---|
| `AA_DATA_DIR` | directory holding `shard_*.pkl` and caches | `./data` |
| `AA_RUN_DIR` | output directory for checkpoints | `./runs` |
| `AA_CKPT` | checkpoint used by `analyze_gate.py` | `runs/full/best.pth` |
| `AA_RAW_FILE` | raw coordinate dump for `preprocess.py` | `data/raw/final.txt` |
| `TRAIN_SUB` | sub-directory under `AA_RUN_DIR` for this run | `ablation_<tag>_<task>` |
| `AB_EMB`, `AB_GATE`, `AB_QBF`, `AB_DUAL` | set to `0` to remove one component | all enabled |
| `HEAD_WIDTH` | hidden width of the readout head | `128` (equal to `hidden_channels`) |

Key training settings (see `scripts/train.py`): AdamW, lr 1e-4, weight decay
1e-4, cosine annealing from 1e-4 to 1e-7 over 200 epochs stepped once per epoch,
gradient clipping 10.0, early stopping on validation MAE with patience 25,
batches capped at 48 graphs / 4e6 triplets, smooth-L1 loss in normalized target
space.

Because the checkpoint directory is derived from the ablation flags only, two
configurations that differ merely in letter case collide on case-insensitive
file systems (Windows). Use `TRAIN_SUB` to give every run its own directory:

```bash
TRAIN_SUB=wo_qbf AB_QBF=0 python scripts/train.py
```

---

## Reproducing the reported numbers

| Step | Command | Output |
|---|---|---|
| proposed model | `python scripts/train.py` | `runs/ablation_EGQD/best.pth`, `final.pth` |
| baselines | edit `MODEL_TYPE` to `schnet` / `dimenet_pp` / `visnet` and re-run | `runs/baseline_<name>/` |
| gate analysis | `python scripts/analyze_gate.py --max-graphs 1500` | per-hop table + JSON |
| ablation | `python scripts/run_ablations.py --yes` | `runs/wo_*/`, `ablation_legacy_summary.md` |

The splits are cached as `split_<mode>_s<seed>_<tasks>_vr<..>_tr<..>_<fp>.pkl`
and are recomputed automatically if the shard files change.

---

## Notes and limitations

- The `bulk` split is the strictest protocol (no test material occurs in
  training). Results for `miller` and `site` are not reported here.
- The gate multiplier is `0.5 + 0.5 g`, so the gate attenuates features but
  never suppresses them; an absolute radius defined by `g < 0.5` is not
  meaningful for this model.
- The hop prior `rf` is a precomputed function of the graph topology and enters
  both the embedding and the gate input, so part of any gate–hop relation is
  built in by construction.
- The three baselines use this repository's re-implementations (PBC edges,
  mean pooling, a shared head design), not the authors' original code, and
  therefore do not reproduce published numbers exactly.

---

## Citation

If you use this code, please cite the accompanying manuscript (see
`CITATION.cff`). Citation details will be updated upon publication.

## License

MIT — see `LICENSE`.
