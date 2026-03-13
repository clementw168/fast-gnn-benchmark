"""
SEAL dataset for link prediction.

Uses a C++ extension (seal.cpp) compiled on first import via
torch.utils.cpp_extension.load for fast subgraph extraction and DRNL labeling.

Usage
-----
    from fast_gnn_benchmark.data.seal_dataset import SEALDatasetCpp

    dataset = FixLinkPropPredDataset(name="ogbl-collab", root="./datasets/ogbl/")
    seal_train = SEALDatasetCpp(dataset, split_type=SplitType.TRAIN,
                                num_neighbors=[20, 10])
    loader = DataLoader(seal_train, batch_size=32, shuffle=True,
                        num_workers=4, persistent_workers=True)
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from fast_gnn_benchmark.data.link_dataloader import (
    cannonize_positive_edges,
    rejection_sampling_negative_edges,
)
from fast_gnn_benchmark.schemas.dataset_models import SplitType

# ---------------------------------------------------------------------------
# Lazy-load the C++ extension (compiled once, cached by torch).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_extension() -> Any:
    from torch.utils.cpp_extension import load

    _csrc_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "csrc"))

    return load(
        name="seal_cpp",
        sources=[os.path.join(_csrc_dir, "seal.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a COO edge_index (shape [2, E]) to CSR (row_ptr, col_idx)."""
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)

    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]

    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(row_ptr[1:], src, 1)
    np.cumsum(row_ptr, out=row_ptr)

    return row_ptr, dst


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SEALDatasetCpp(Dataset):
    """PyG Dataset that extracts SEAL subgraphs on-the-fly using C++.

    Each item is a PyG ``Data`` object containing:
        - ``x``        : node features (or None)
        - ``edge_index``: local edge index (target edge removed)
        - ``z``        : DRNL node labels (int64)
        - ``node_id``  : original global node ids
        - ``y``        : edge label (1=positive, 0=negative)
        - ``num_nodes``: number of nodes in the subgraph

    Args:
        dataset:       An OGB-style dataset with ``data``, ``num_nodes``,
                       and ``split`` attributes.
        split_type:    One of SplitType.TRAIN / VAL / TEST.
        num_neighbors: Per-hop neighbour limit list, e.g. ``[20, 10, 5]``.
                       ``len(num_neighbors)`` determines the number of hops.
                       At each hop h, every fringe node samples at most
                       ``num_neighbors[h-1]`` new neighbours independently.
                       Use ``-1`` for a hop to impose no limit.
        max_rejection_sampling_iterations:
                       Iterations for negative edge sampling during training.
    """

    def __init__(
        self,
        dataset: Any,
        split_type: SplitType = SplitType.TRAIN,
        num_neighbors: list[int] = [20, 10],
        max_rejection_sampling_iterations: int = 3,
    ):
        super().__init__()

        ext = _load_extension()
        self.num_neighbors = list(num_neighbors)

        data = dataset.data
        num_nodes: int = data.num_nodes

        self._node_features: torch.Tensor | None = data.x if hasattr(data, "x") else None

        edge_index = data.edge_index.cpu()
        self.row_ptr, self.col_idx = edge_index_to_csr(edge_index, num_nodes)

        # Each DataLoader worker gets its own SEALExtractor (via pickle).
        # The extractor holds two flat arrays of size num_nodes for O(1)
        # visited/g2l lookups without per-call hash map allocation.
        self._extractor = ext.SEALExtractor(num_nodes)

        # ----------------------------------------------------------------
        # Build target edge list for this split
        # ----------------------------------------------------------------
        device = torch.device("cpu")

        if split_type == SplitType.TRAIN:
            pos_edges, non_neg_ids = cannonize_positive_edges(
                dataset, num_nodes, device, remove_self_loops=True
            )
            neg_edges = rejection_sampling_negative_edges(
                pos_edges.shape[1],
                non_neg_ids,
                num_nodes,
                max_rejection_sampling_iterations,
                device,
            )
        else:
            split_key = "valid" if split_type == SplitType.VAL else "test"
            pos_edges = dataset.split[split_key]["edge"].T
            neg_edges = dataset.split[split_key]["edge_neg"].T

        self.target_edges = torch.cat([pos_edges, neg_edges], dim=1)  # [2, E]
        self.labels = torch.cat(
            [
                torch.ones(pos_edges.shape[1], dtype=torch.float),
                torch.zeros(neg_edges.shape[1], dtype=torch.float),
            ]
        )

    # ------------------------------------------------------------------

    def len(self) -> int:
        return self.target_edges.shape[1]

    def get(self, idx: int) -> Data:
        src = int(self.target_edges[0, idx])
        dst = int(self.target_edges[1, idx])

        result = self._extractor.extract(
            self.row_ptr,
            self.col_idx,
            src,
            dst,
            self.num_neighbors,
            idx,  # per-sample seed for reproducibility
        )

        # vec_to_numpy in C++ transfers ownership with zero copies.
        # torch.from_numpy shares the same buffer (also zero-copy).
        node_ids = torch.from_numpy(result["node_ids"])
        edge_index = torch.stack(
            [
                torch.from_numpy(result["edge_src"]),
                torch.from_numpy(result["edge_dst"]),
            ],
            dim=0,
        )
        z = torch.from_numpy(result["z_drnl"])
        y = self.labels[idx].unsqueeze(0)

        x = None
        if self._node_features is not None:
            x = self._node_features[node_ids]

        return Data(
            x=x,
            edge_index=edge_index,
            z=z,
            node_id=node_ids,
            y=y,
            num_nodes=len(node_ids),
        )


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    from fast_gnn_benchmark.data.dataset.ogbl import FixLinkPropPredDataset

    dataset = FixLinkPropPredDataset(name="ogbl-ppa", root="./datasets/ogbl/")
    seal_train = SEALDatasetCpp(dataset, split_type=SplitType.TRAIN, num_neighbors=[20, 10])
    loader = DataLoader(
        seal_train, batch_size=2048, shuffle=True,
        num_workers=32, persistent_workers=True,
    )
    for batch in tqdm(loader, desc="Processing batch"):
        pass
