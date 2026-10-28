"""Shared SurfaceData definition.

Defining the class once avoids the pickle/pathology where preprocessing and
training each declare their own copy: the pickle then stores
``__main__.SurfaceData`` and training only loads it by accident of a
same-named module-level class.
"""

import torch
from torch_geometric.data import Data


class SurfaceData(Data):
    """
    Fields:
      z                 (N,)     atomic numbers
      pos               (N,3)    Cartesian coordinates
      tag               (N,)     0/1/2: fixed substrate / relaxable / adsorbate
      force             (N,3)    DFT forces
      cell              (3,3)    lattice; batched as (3B,3), reshaped in the model
      edge_index        (2,E)    periodic neighbour list, src -> dst
      edge_shift        (E,3)    integer lattice shift per edge (not an index)
      hops_to_adsorbate (N,)     BFS hops to the adsorbate (MAX_HOPS if absent)
      rf_weights        (N,)     continuous distance prior (0 if no adsorbate)
      quad_edges        (M,3)    chain-edge indices [e_ji, e_kj, e_lk]
      quad_dists        (M,3)    [d(c,j), d(c,k), d(c,l)], minimum-image distances
      y                 (T,)     targets; NaN when unavailable
      sample_weight    (1,)
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'quad_edges':
            return self.edge_index.size(1)
        if key in ('edge_shift', 'cell', 'quad_dists'):
            return 0  # shifts, distances and cells are not indices
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('quad_edges', 'quad_dists', 'edge_shift', 'cell'):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)