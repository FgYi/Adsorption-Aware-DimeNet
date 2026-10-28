# Data

This directory is **not** tracked by git (see `.gitignore`). It must contain the
sharded dataset produced by `scripts/preprocess.py`, or by an equivalent
pipeline, together with the cached splits.

```
data/
├── preprocess_config.pkl      # preprocessing constants, checked at train time
├── shard_0000.pkl ... shard_XXXX.pkl
├── split_bulk_s168_<tasks>_vr0.2_tr0.2_<fp>.pkl   # generated on first run
└── ystats_bulk_s168_<tasks>_vr0.2_tr0.2_<fp>.pt   # generated on first run
```

## Shard format

Each shard is a `pickle`-ed **list** of `SurfaceData` objects (or `None` for
dropped configurations). `SurfaceData` is a `torch_geometric.data.Data`
subclass defined in `models/surface_data.py`, so importing that module is
required before unpickling:

| Field | Shape / type | Meaning |
|---|---|---|
| `z` | `(N,)` long | atomic numbers |
| `pos` | `(N, 3)` float | Cartesian coordinates (Å) |
| `tag` | `(N,)` long | 0 = fixed substrate, 1 = relaxable substrate, 2 = adsorbate |
| `cell` | `(3, 3)` float | lattice matrix |
| `edge_index` | `(2, E)` long | periodic neighbour list (both directions) |
| `edge_shift` | `(E, 3)` long | integer lattice shift per edge, `r = r_dst + s·A − r_src` |
| `hops_to_adsorbate` | `(N,)` long | BFS hops to the nearest adsorbate atom, capped at 10 |
| `rf_weights` | `(N,)` float | continuous prior `rf = 1 − hops/10` |
| `quad_edges` | `(M, 3)` long | quadruplet → triplet-edge indices |
| `quad_dists` | `(M, 3)` float | radial distances of the three chain edges |
| `y` | `(T,)` float | target values, first entry is the adsorption energy (eV) |
| `sample_weight` | `(1,)` float | 1.5 for experimentally synthesised substrates, else 1.0 |
| `force` | `(N, 3)` float | DFT forces (stored but not used by the energy-only model) |

Attributes `material_id`, `frame`, `miller_index` and `random_id` are stored on
each object and are used to build the grouped splits.

## Preprocessing constants

`preprocess_config.pkl` stores `CUTOFF`, `R_QUAD`, `MAX_QUADS_PER_ADS`,
`MAX_HOPS`, `MAX_NUM_NEIGHBORS`, `NUM_RADIAL`, `NUM_SPHERICAL` and
`ENVELOPE_EXPONENT`. `scripts/train.py` validates them against
`EXPECTED_PP_CONFIG` and aborts on any mismatch, so that stale data cannot be
used silently.
