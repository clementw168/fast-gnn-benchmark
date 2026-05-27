"""
SEAL dataset for link prediction.

Uses a C++ extension (seal.cpp) compiled on first import via
torch.utils.cpp_extension.load for fast subgraph extraction and DRNL labeling.

Usage
-----
    from fast_gnn_benchmark.data.seal_dataset import OfflineSealLoader, SealLoader

    dataset = FixLinkPropPredDataset(name="ogbl-collab", root="./datasets/ogbl/")

    # Online — extracts subgraphs on the fly:
    #   loader = SealLoader(dataset, split_type=SplitType.TRAIN,
    #                       num_neighbors=[20, 10], batch_size=32,
    #                       num_workers=4, persistent_workers=True)

    # Offline — precomputes once, fast on every subsequent run:
    #   loader = OfflineSealLoader(dataset, root="./datasets/seal_cache/",
    #                              split_type=SplitType.TRAIN,
    #                              num_neighbors=[20, 10], batch_size=32,
    #                              precompute_workers=8, num_workers=4,
    #                              persistent_workers=True)
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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
            pos_edges, non_neg_ids = cannonize_positive_edges(dataset, num_nodes, device, remove_self_loops=True)
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

        # For val/test: shuffle with a fixed seed so every batch contains a
        # proportional mix of positive and negative edges.  Without this,
        # positives come first and `limit_val_batches` would only ever see
        # positive samples, making hit@k = 1.0 trivially.
        if split_type != SplitType.TRAIN:
            generator = torch.Generator().manual_seed(0)
            perm = torch.randperm(self.target_edges.shape[1], generator=generator)
            self.target_edges = self.target_edges[:, perm]
            self.labels = self.labels[perm]

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


class SealLoader:
    """Builds a SEALDatasetCpp from an OGB-style dataset and wraps it with PyG's DataLoader.

    Each batch is a PyG Batch of subgraphs with:
        - x         : node features [total_nodes, feat_dim] (or None)
        - edge_index: local edge indices [2, total_edges]
        - z         : DRNL labels [total_nodes], int64
        - node_id   : original global node ids [total_nodes]
        - y         : edge labels [batch_size], float (1=positive, 0=negative)
        - batch     : graph membership [total_nodes]
        - ptr       : subgraph boundaries [batch_size + 1]
    """

    def __init__(
        self,
        dataset: Any,
        split_type: SplitType = SplitType.TRAIN,
        num_neighbors: list[int] = [20, 10],
        batch_size: int = 32,
        max_rejection_sampling_iterations: int = 3,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        seal_dataset = SEALDatasetCpp(
            dataset,
            split_type=split_type,
            num_neighbors=num_neighbors,
            max_rejection_sampling_iterations=max_rejection_sampling_iterations,
        )

        self._loader = DataLoader(
            seal_dataset,
            batch_size=batch_size,
            shuffle=(split_type == SplitType.TRAIN),
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


# ---------------------------------------------------------------------------
# Offline (precomputed) SEAL dataset
# ---------------------------------------------------------------------------

_MAX_CACHED_CHUNKS = 8


class OfflineSEALDataset(Dataset):
    """PyG Dataset that precomputes SEAL subgraphs once and saves them to disk.

    On first instantiation all subgraphs are extracted with the C++ extension
    and stored as chunked ``.pt`` files under::

        {root}/seal_{split}_k{h1}-{h2}/
            metadata.pt          # labels + bookkeeping
            chunk_000000.pt      # list of up to ``chunk_size`` Data objects
            chunk_000001.pt
            ...

    Subsequent instantiations with the same (root, split, num_neighbors) skip
    extraction entirely and load directly from disk.

    Each ``Data`` object is identical to what ``SEALDatasetCpp`` returns:
        - ``x``         : node features (or None)
        - ``edge_index``: local edge index (target edge removed)
        - ``z``         : DRNL node labels (int64)
        - ``node_id``   : original global node ids
        - ``y``         : edge label (1=positive, 0=negative)
        - ``num_nodes`` : number of nodes in the subgraph

    .. note::
        For the **training** split, negative edges are sampled once during
        precomputation and then fixed.  Pass ``force_reprocess=True`` to
        draw a fresh set of negatives.

    Args:
        dataset:        An OGB-style dataset with ``data`` and ``split``.
        root:           Root directory for cached files.
        split_type:     ``SplitType.TRAIN`` / ``VAL`` / ``TEST``.
        num_neighbors:  Per-hop neighbour limit, e.g. ``[20, 10]``.
        max_rejection_sampling_iterations:
                        Iterations for negative-edge rejection sampling
                        (training split only).
        chunk_size:     Number of ``Data`` objects per chunk file.
                        Larger values → fewer files, higher per-load memory
                        spike.  1 000 is a good default.
        num_workers:    Parallel workers used *only during precomputation*.
                        Set to 0 for single-process extraction.
        force_reprocess:
                        Recompute even if a cache already exists.
    """

    def __init__(
        self,
        dataset: Any,
        root: str,
        split_type: SplitType = SplitType.TRAIN,
        num_neighbors: list[int] = [20, 10],
        max_rejection_sampling_iterations: int = 3,
        chunk_size: int = 1000,
        num_workers: int = 0,
        force_reprocess: bool = False,
    ):
        super().__init__()

        # Config-derived cache directory so different configs never collide.
        neighbors_str = "-".join(map(str, num_neighbors))
        split_name = split_type.name.lower()
        self._processed_dir = os.path.join(root, f"seal_{split_name}_k{neighbors_str}")
        self._metadata_path = os.path.join(self._processed_dir, "metadata.pt")
        self._input_chunk_size = chunk_size

        # Per-worker in-memory chunk cache (never shared across workers).
        self._chunk_cache: dict[int, list[Data]] = {}

        if force_reprocess or not os.path.exists(self._metadata_path):
            # Fresh precomputation from scratch.
            self._precompute(dataset, split_type, num_neighbors, max_rejection_sampling_iterations, num_workers, start_idx=0)
        else:
            # Metadata exists — check whether any chunks are missing and resume.
            meta = torch.load(self._metadata_path, weights_only=False)
            num_chunks = math.ceil(meta["num_graphs"] / meta["chunk_size"])
            first_missing = next(
                (i for i in range(num_chunks) if not os.path.exists(self._chunk_path(i))),
                None,
            )
            if first_missing is not None:
                start_idx = first_missing * meta["chunk_size"]
                split_name = split_type.name.lower()
                print(
                    f"[SEAL] Resuming {split_name} precomputation from "
                    f"index {start_idx}/{meta['num_graphs']} (chunk {first_missing}/{num_chunks})"
                )
                self._precompute(
                    dataset, split_type, num_neighbors, max_rejection_sampling_iterations, num_workers,
                    start_idx=start_idx,
                )

        meta = torch.load(self._metadata_path, weights_only=False)
        self._labels: torch.Tensor = meta["labels"]
        self._num_graphs: int = meta["num_graphs"]
        self._chunk_size: int = meta["chunk_size"]

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute(
        self,
        dataset: Any,
        split_type: SplitType,
        num_neighbors: list[int],
        max_rejection_sampling_iterations: int,
        num_workers: int,
        start_idx: int = 0,
    ) -> None:
        from torch.utils.data import Subset

        os.makedirs(self._processed_dir, exist_ok=True)

        online = SEALDatasetCpp(
            dataset,
            split_type=split_type,
            num_neighbors=num_neighbors,
            max_rejection_sampling_iterations=max_rejection_sampling_iterations,
        )
        n = len(online)
        split_name = split_type.name.lower()

        if start_idx == 0:
            # Fresh run: write metadata (including the full edge list so that a
            # later resume uses the exact same positive/negative edges).
            torch.save(
                {
                    "labels": online.labels,
                    "target_edges": online.target_edges,
                    "num_graphs": n,
                    "chunk_size": self._input_chunk_size,
                },
                self._metadata_path,
            )
        else:
            # Resumed run: restore the original edge list so the subgraphs
            # produced here are consistent with the already-saved chunks.
            saved = torch.load(self._metadata_path, weights_only=False)
            online.target_edges = saved["target_edges"]
            online.labels = saved["labels"]

        chunk: list[Data] = []
        chunk_idx = start_idx // self._input_chunk_size
        remaining = n - start_idx

        if num_workers > 0:
            # The default 'file_descriptor' sharing strategy hits OS fd limits when
            # many workers share large tensors (target_edges, labels, node features).
            # Switch to 'file_system' for the duration of precomputation.
            prev_strategy = mp.get_sharing_strategy()
            mp.set_sharing_strategy("file_system")
            try:
                source: Any = DataLoader(
                    Subset(online, list(range(start_idx, n))),
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    persistent_workers=False,
                    collate_fn=lambda batch: batch[0],  # unwrap single-item list
                )
                for data in tqdm(source, total=remaining, desc=f"Precomputing SEAL [{split_name}]"):
                    chunk.append(data)
                    if len(chunk) == self._input_chunk_size:
                        self._save_chunk(chunk_idx, chunk)
                        chunk = []
                        chunk_idx += 1
            finally:
                mp.set_sharing_strategy(prev_strategy)
        else:
            for data in tqdm(
                (online.get(i) for i in range(start_idx, n)),
                total=remaining,
                desc=f"Precomputing SEAL [{split_name}]",
            ):
                chunk.append(data)
                if len(chunk) == self._input_chunk_size:
                    self._save_chunk(chunk_idx, chunk)
                    chunk = []
                    chunk_idx += 1

        if chunk:  # flush last (possibly partial) chunk
            self._save_chunk(chunk_idx, chunk)

    def _chunk_path(self, chunk_idx: int) -> str:
        return os.path.join(self._processed_dir, f"chunk_{chunk_idx:08d}.pt")

    def _save_chunk(self, chunk_idx: int, chunk: list[Data]) -> None:
        torch.save(chunk, self._chunk_path(chunk_idx))

    # ------------------------------------------------------------------
    # Pickling — workers must start with an empty cache
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_chunk_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def len(self) -> int:
        return self._num_graphs

    def get(self, idx: int) -> Data:
        chunk_idx = idx // self._chunk_size
        local_idx = idx % self._chunk_size

        if chunk_idx not in self._chunk_cache:
            # FIFO eviction: keep at most _MAX_CACHED_CHUNKS chunks per worker.
            if len(self._chunk_cache) >= _MAX_CACHED_CHUNKS:
                oldest = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest]
            self._chunk_cache[chunk_idx] = torch.load(self._chunk_path(chunk_idx), weights_only=False)

        return self._chunk_cache[chunk_idx][local_idx]


class OfflineSealLoader:
    """Precomputes SEAL subgraphs once, then wraps them with PyG's DataLoader.

    The first call triggers on-disk precomputation via ``OfflineSEALDataset``;
    every subsequent call (same root + config) skips extraction and loads
    directly from the cached chunk files.

    Each batch is a PyG Batch of subgraphs with:
        - x         : node features [total_nodes, feat_dim] (or None)
        - edge_index: local edge indices [2, total_edges]
        - z         : DRNL labels [total_nodes], int64
        - node_id   : original global node ids [total_nodes]
        - y         : edge labels [batch_size], float (1=positive, 0=negative)
        - batch     : graph membership [total_nodes]
        - ptr       : subgraph boundaries [batch_size + 1]

    Args:
        dataset:        An OGB-style dataset with ``data`` and ``split``.
        root:           Directory for cached chunk files.
        split_type:     ``SplitType.TRAIN`` / ``VAL`` / ``TEST``.
        num_neighbors:  Per-hop neighbour limit, e.g. ``[20, 10]``.
        batch_size:     Batch size for the DataLoader.
        max_rejection_sampling_iterations:
                        Iterations for negative-edge rejection sampling
                        (training split only).
        chunk_size:     ``Data`` objects per chunk file (precompute time only).
        precompute_workers:
                        Parallel workers for the one-time extraction step.
        num_workers:    DataLoader workers for batch iteration.
        persistent_workers:
                        Keep DataLoader workers alive between epochs.
        force_reprocess:
                        Recompute even if a cache already exists.
    """

    def __init__(
        self,
        dataset: Any,
        root: str,
        split_type: SplitType = SplitType.TRAIN,
        num_neighbors: list[int] = [20, 10],
        batch_size: int = 32,
        max_rejection_sampling_iterations: int = 3,
        chunk_size: int = 1000,
        precompute_workers: int = 0,
        num_workers: int = 0,
        persistent_workers: bool = False,
        force_reprocess: bool = False,
    ):
        seal_dataset = OfflineSEALDataset(
            dataset,
            root=root,
            split_type=split_type,
            num_neighbors=num_neighbors,
            max_rejection_sampling_iterations=max_rejection_sampling_iterations,
            chunk_size=chunk_size,
            num_workers=precompute_workers,
            force_reprocess=force_reprocess,
        )

        self._loader = DataLoader(
            seal_dataset,
            batch_size=batch_size,
            shuffle=(split_type == SplitType.TRAIN),
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


if __name__ == "__main__":
    from fast_gnn_benchmark.data.dataset.ogbl import FixLinkPropPredDataset

    dataset = FixLinkPropPredDataset(name="ogbl-ppa", root="./datasets/ogbl/")
    loader = OfflineSealLoader(
        dataset,
        root="./datasets/seal_cache/",
        split_type=SplitType.TRAIN,
        num_neighbors=[20, 10],
        batch_size=2048,
        precompute_workers=32,
        num_workers=8,
        persistent_workers=True,
    )
    for i, batch in enumerate(tqdm(loader, desc="Processing batch")):
        print("--------------------------------")
        print(i)
        print("batch: ", batch)
        print("batch.x.shape: ", batch.x.shape)
        print("batch.edge_index.shape: ", batch.edge_index.shape)
        print("batch.z.shape: ", batch.z.shape)
        print("batch.node_id.shape: ", batch.node_id.shape)
        print("batch.y.shape: ", batch.y.shape)
        print("batch.num_nodes: ", batch.num_nodes)

        print()
        print("batch.x: ", batch.x)
        print("x non zero: ", (batch.x != 0).sum())
        print("batch.edge_index: ", batch.edge_index)
        print("batch.z: ", batch.z)
        print("batch.node_id: ", batch.node_id)
        print("batch.y: ", batch.y)
        print("batch.num_nodes: ", batch.num_nodes)
        if i > 3:
            break
