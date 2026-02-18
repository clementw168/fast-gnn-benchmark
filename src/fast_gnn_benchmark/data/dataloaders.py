import os
from collections.abc import Callable, Iterator
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import (
    ClusterData,
    ClusterLoader,
    DataLoader,
    GraphSAINTRandomWalkSampler,
    NeighborLoader,
    RandomNodeLoader,
)
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import degree, subgraph
from tqdm import tqdm

from fast_gnn_benchmark.data.utils import to_undirected
from fast_gnn_benchmark.schemas.dataset_models import SplitType


class SingleGraphDataset(InMemoryDataset):
    def __init__(
        self,
        data: Data,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self._data_obj = data
        super().__init__(".", transform, pre_transform)

        # Collate into PyG's internal storage format
        data_list: list[Data] = [self._data_obj]
        self.data, self.slices = self.collate(data_list)

    def get(self, idx: int) -> Data:
        # Standard PyG pattern — retrieves from self.data/self.slices
        return super().get(idx)  # type: ignore

    def len(self) -> int:
        return 1


def add_compute_mask(data: Data, split_type: SplitType) -> Data:
    match split_type:
        case SplitType.TRAIN:
            compute_mask = data.train_mask
        case SplitType.VAL:
            compute_mask = data.val_mask
        case SplitType.TEST:
            compute_mask = data.test_mask
        case _:
            raise ValueError(f"Invalid split type: {split_type}")

    data.compute_mask = compute_mask

    return data


def inductive_subgraph(data: Data, split_type: SplitType) -> Data:
    match split_type:
        case SplitType.TRAIN:
            nodes_to_keep = data.train_mask
        case SplitType.VAL:
            nodes_to_keep = torch.logical_or(data.val_mask, data.train_mask)
        case SplitType.TEST:
            nodes_to_keep = torch.logical_or(data.test_mask, torch.logical_or(data.val_mask, data.train_mask))

    subgraph_edges, _ = subgraph(nodes_to_keep, data.edge_index, relabel_nodes=True)  # type: ignore

    return Data(
        x=data.x[nodes_to_keep],  # type: ignore
        edge_index=subgraph_edges,
        y=data.y[nodes_to_keep],  # type: ignore
        train_mask=data.train_mask[nodes_to_keep],
        val_mask=data.val_mask[nodes_to_keep],
        test_mask=data.test_mask[nodes_to_keep],
    )


"""
All data loaders should return a Data object with the following attributes:
- x: Tensor[float, (num_nodes, num_features)]
- edge_index: Tensor[int, (2, num_edges)]
- y: Tensor[int, (num_nodes,)]
- compute_mask: Tensor[bool, (num_nodes,)]  (boolean tensor of shape [num_nodes] indicating which nodes should be used for computing the loss)

This is mandatory to be compatible with the step functions.
"""


class BaseDataLoader:
    def __init__(
        self,
        dataset: Any,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        self.split_type = split_type

        self.data_loader = DataLoader(
            dataset,  # type: ignore
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def __len__(self) -> int:
        return len(self.data_loader)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        for batch in self.data_loader:
            batch_ = add_compute_mask(batch, self.split_type)
            yield batch_


class RandomNodeLoaderWrapper:
    def __init__(
        self,
        dataset: Any,
        num_workers: int = 0,
        num_parts: int = 1,
        shuffle: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        self.split_type = split_type

        data = dataset[0]
        assert isinstance(data, Data)

        self.device = torch.accelerator.current_accelerator() or torch.device("cpu")

        self.data_loader = RandomNodeLoader(
            data,
            num_workers=num_workers,
            num_parts=num_parts,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.to_sparse_tensor = ToSparseTensor()

    def __len__(self) -> int:
        return len(self.data_loader)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        for batch in self.data_loader:
            subgraph = add_compute_mask(batch, self.split_type)

            if self.device != torch.device("mps"):  # Bug in MPS backend for s
                subgraph = self.to_sparse_tensor(subgraph)
                subgraph.edge_index = subgraph.adj_t

            yield subgraph


class OptimizedRandomNodeLoader:
    def __init__(
        self,
        dataset: Any,
        num_parts: int,
        on_device: bool = True,
        pin_memory: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        data = dataset[0]
        assert isinstance(data, Data)

        self.num_parts = num_parts
        self.num_nodes = data.x.shape[0]  # type: ignore
        self.split_type = split_type

        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cpu") and pin_memory:
            print("Warning: pin_memory is set to True but device is cpu. Setting pin_memory to False.")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # +1 to avoid having a last batch with less nodes than the others
        self.num_sample_per_batch = self.num_nodes // num_parts + 1
        print(f"Num sample per batch: {self.num_sample_per_batch}")
        self.data = data
        self.to_sparse_tensor = ToSparseTensor()

    def __len__(self) -> int:
        return self.num_parts

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        data = self.data.to(self.device)  # type: ignore
        permutations = torch.randperm(self.num_nodes, pin_memory=self.pin_memory, device=self.device)
        for i in range(self.num_parts):
            start_idx = i * self.num_sample_per_batch
            end_idx = start_idx + self.num_sample_per_batch

            subgraph = data.subgraph(permutations[start_idx:end_idx])
            subgraph = add_compute_mask(subgraph, self.split_type)
            if self.device != torch.device("mps"):  # Bug in MPS backend for s
                subgraph = self.to_sparse_tensor(subgraph)
                subgraph.edge_index = subgraph.adj_t  # respect the common storage format for all models
            yield subgraph

        del data


class RandomNodeLoaderWithReplacement:
    def __init__(
        self,
        dataset: Any,
        proportion: float,
        on_device: bool = True,
        pin_memory: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        data = dataset[0]
        assert isinstance(data, Data)

        self.num_nodes = data.x.shape[0]  # type: ignore
        self.split_type = split_type

        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cpu") and pin_memory:
            print("Warning: pin_memory is set to True but device is cpu. Setting pin_memory to False.")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # +1 to avoid having a last batch with less nodes than the others
        self.num_sample_per_batch = int(self.num_nodes * proportion)
        self.num_parts = ceil(self.num_nodes / self.num_sample_per_batch)
        print(f"Num sample per batch: {self.num_sample_per_batch}")
        self.data = data
        self.to_sparse_tensor = ToSparseTensor()

    def __len__(self) -> int:
        return self.num_parts

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        data = self.data.to(self.device)  # type: ignore
        for i in range(self.num_parts):
            indices = torch.randperm(self.num_nodes, pin_memory=self.pin_memory, device=self.device)[
                : self.num_sample_per_batch
            ]
            subgraph = data.subgraph(indices)
            subgraph = add_compute_mask(subgraph, self.split_type)
            if self.device != torch.device("mps"):  # Bug in MPS backend for s
                subgraph = self.to_sparse_tensor(subgraph)
                subgraph.edge_index = subgraph.adj_t  # respect the common storage format for all models
            yield subgraph

        del data


class OptimizedRandomNodeLoaderDebug:
    """A version returning the partition nodes for debugging purposes."""

    def __init__(
        self,
        dataset: Any,
        num_parts: int,
        on_device: bool = True,
        pin_memory: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        data = dataset[0]
        assert isinstance(data, Data)

        self.num_parts = num_parts
        self.num_nodes = data.x.shape[0]  # type: ignore
        self.split_type = split_type

        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cpu") and pin_memory:
            print("Warning: pin_memory is set to True but device is cpu. Setting pin_memory to False.")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # +1 to avoid having a last batch with less nodes than the others
        self.num_sample_per_batch = self.num_nodes // num_parts + 1

        print(f"Num sample per batch: {self.num_sample_per_batch}")
        self.data = data
        self.to_sparse_tensor = ToSparseTensor()

    def __len__(self) -> int:
        return self.num_parts

    def __iter__(self) -> Iterator[tuple[Data, torch.Tensor]]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[tuple[Data, torch.Tensor]]:
        data = self.data.to(self.device)  # type: ignore
        permutations = torch.randperm(self.num_nodes, pin_memory=self.pin_memory, device=self.device)
        for i in range(self.num_parts):
            start_idx = i * self.num_sample_per_batch
            end_idx = start_idx + self.num_sample_per_batch

            subgraph = data.subgraph(permutations[start_idx:end_idx])
            subgraph = add_compute_mask(subgraph, self.split_type)
            if self.device != torch.device("mps"):  # Bug in MPS backend for s
                subgraph = self.to_sparse_tensor(subgraph)
                subgraph.edge_index = subgraph.adj_t  # respect the common storage format for all models
            yield subgraph, permutations[start_idx:end_idx]

        del data


class NeighborLoaderWrapper:
    def __init__(
        self,
        dataset: Any,
        num_neighbors: list[int],
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split_type: SplitType = SplitType.TRAIN,
        on_device: bool = True,
    ):
        self.split_type = split_type

        data = dataset[0]
        assert isinstance(data, Data)

        if on_device and num_workers == 0:
            device = torch.accelerator.current_accelerator()
            if device == torch.device("mps") or device is None:
                if device == torch.device("mps"):
                    print("MPS is not supported for NeighborLoader. Using CPU instead.")
                device = torch.device("cpu")

            data = data.to(device)

        match self.split_type:
            case SplitType.TRAIN:
                mask = data.train_mask
            case SplitType.VAL:
                mask = data.val_mask
            case SplitType.TEST:
                mask = data.test_mask
            case _:
                raise ValueError(f"Invalid split type: {self.split_type}")

        self.data_loader = NeighborLoader(
            data,
            input_nodes=mask,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return len(self.data_loader)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        for batch in self.data_loader:
            batch_ = add_compute_mask(batch, self.split_type)
            batch_mask = torch.zeros(batch_.x.shape[0], dtype=torch.bool).to(batch_.compute_mask.device)  # type: ignore
            batch_mask[: batch_.batch_size] = True
            # Mask is from correct split and only keep the seeds of Neighborhood sampler
            batch_.compute_mask = torch.logical_and(batch_.compute_mask, batch_mask)
            yield batch_


class RandomWalkLoaderWrapper:
    def __init__(
        self,
        dataset: Any,
        walk_length: int,
        num_seeds: int,
        compute_normalization_stats: bool = False,
        num_stats_samples: int = 1000,
        split_type: SplitType = SplitType.TRAIN,
    ):
        self.split_type = split_type

        data = dataset[0]
        num_nodes = data.num_nodes
        assert isinstance(data, Data)

        num_walks_per_epoch = ceil(num_nodes / (walk_length * num_seeds))

        self.data_loader = GraphSAINTRandomWalkSampler(
            data,
            batch_size=num_seeds,
            walk_length=walk_length,
            num_steps=num_walks_per_epoch,
            sample_coverage=num_stats_samples if compute_normalization_stats else 0,
        )

        self.device = torch.accelerator.current_accelerator() or torch.device("cpu")

        self.to_sparse_tensor = ToSparseTensor()

    def __len__(self) -> int:
        return len(self.data_loader)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        for batch in self.data_loader:
            batch_ = add_compute_mask(batch, self.split_type)

            if self.device != torch.device("mps"):  # Bug in MPS backend for s
                batch_ = self.to_sparse_tensor(batch_)
                batch_.edge_index = batch_.adj_t

            yield batch_


class ClusterLoaderWrapper:
    def __init__(
        self,
        dataset: Any,
        num_clusters: int,
        num_parts: int,
        split_type: SplitType = SplitType.TRAIN,
        shuffle: bool = True,
    ):
        self.split_type = split_type

        data = dataset[0]
        assert isinstance(data, Data)
        cluster_data = ClusterData(data, num_parts=num_clusters)  # 1. Create subgraphs.
        cluster_loader = ClusterLoader(cluster_data, batch_size=num_clusters // num_parts, shuffle=shuffle)
        self.data_loader = cluster_loader

    def __len__(self) -> int:
        return len(self.data_loader)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        for batch in self.data_loader:
            batch_ = add_compute_mask(batch, self.split_type)

            yield batch_


class DropEdgeLoader:
    def __init__(
        self,
        dataset: Any,
        drop_edge_ratio: float,
        on_device: bool = True,
        pin_memory: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        data = dataset[0]
        assert isinstance(data, Data)

        self.num_nodes = data.x.shape[0]  # type: ignore
        self.split_type = split_type
        self.drop_edge_ratio = drop_edge_ratio

        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cpu") and pin_memory:
            print("Warning: pin_memory is set to True but device is cpu. Setting pin_memory to False.")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        self.data = data
        self.num_edges = data.edge_index.shape[1]  # type: ignore
        self.to_sparse_tensor = ToSparseTensor()

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        edge_mask = torch.rand(self.num_edges, device=self.device) > self.drop_edge_ratio

        subgraph = self.data.clone()
        subgraph.edge_index = subgraph.edge_index[:, edge_mask]  # type: ignore
        subgraph = add_compute_mask(subgraph, self.split_type)
        if self.device != torch.device("mps"):  # Bug in MPS backend for s
            subgraph = self.to_sparse_tensor(subgraph)
            subgraph.edge_index = subgraph.adj_t  # respect the common storage format for all models
        yield subgraph


class PPRNodeLoader:
    def __init__(
        self,
        dataset: Any,
        num_parts: int,
        node_budget: int,
        alpha: float = 0.85,
        ppr_iterations: int = 4,
        on_device: bool = True,
        pin_memory: bool = False,
        split_type: SplitType = SplitType.TRAIN,
    ):
        data = dataset[0]
        assert isinstance(data, Data)
        self.data = data

        self.num_parts = num_parts
        self.num_nodes = self.data.x.shape[0]  # type: ignore
        self.node_budget = node_budget
        self.alpha = alpha
        self.ppr_iterations = ppr_iterations

        self.split_type = split_type

        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
            if self.device == torch.device("mps"):
                print("MPS is not supported for PPRNodeLoader. Using CPU instead.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if self.device == torch.device("cpu") and pin_memory:
            print("Warning: pin_memory is set to True but device is cpu. Setting pin_memory to False.")
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        match self.split_type:
            case SplitType.TRAIN:
                candidate_mask = data.train_mask
            case SplitType.VAL:
                candidate_mask = data.val_mask
            case SplitType.TEST:
                candidate_mask = data.test_mask
            case _:
                raise ValueError(f"Invalid split type: {self.split_type}")

        self.candidates = torch.where(candidate_mask)[0].to(self.device)

        self.num_candidates = len(self.candidates)

        self.num_sample_per_batch = self.num_candidates // self.num_parts
        print(
            f"Node budget: {self.node_budget}, num_sample_per_batch: {self.num_sample_per_batch}, num_candidates: {self.num_candidates}, num_nodes: {self.num_nodes}"
        )
        if self.node_budget < self.num_sample_per_batch:
            raise ValueError(
                f"Node budget must be greater than or equal to the number of nodes: {self.node_budget} < {self.num_sample_per_batch}"
            )

        self.to_sparse_tensor = ToSparseTensor()

        self.normalized_adjacency_matrix = self.get_normalized_adjacency_matrix(data)

    def get_normalized_adjacency_matrix(self, data: Data) -> torch.Tensor:
        edge_index: torch.Tensor = data.edge_index  # type: ignore
        num_nodes: int = data.num_nodes  # type: ignore
        degrees = degree(edge_index[0], num_nodes=data.num_nodes)
        inv_degrees = torch.where(degrees > 0, torch.ones_like(degrees) / degrees, torch.zeros_like(degrees))
        values = inv_degrees[edge_index[1]]
        return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to_sparse_csr().to(self.device)

    def __len__(self) -> int:
        return self.num_parts

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def ppr_sampling(self, nodes_of_interest: torch.Tensor) -> torch.Tensor:
        pi = torch.zeros(self.num_nodes, device=self.device)
        pi[nodes_of_interest] = 1 / len(nodes_of_interest)
        for _ in range(self.ppr_iterations):
            pi = self.alpha * self.normalized_adjacency_matrix @ pi + (1 - self.alpha) * pi

        return torch.argsort(pi, descending=True)[: self.node_budget]

    def add_compute_mask(self, subgraph: Data, partition_nodes: torch.Tensor) -> Data:
        match self.split_type:
            case SplitType.TRAIN:
                compute_mask = subgraph.train_mask
            case SplitType.VAL:
                compute_mask = subgraph.val_mask
            case SplitType.TEST:
                compute_mask = subgraph.test_mask
            case _:
                raise ValueError(f"Invalid split type: {self.split_type}")

        partition_mask = torch.zeros_like(compute_mask, device=self.device, dtype=torch.bool)
        partition_mask[partition_nodes] = True

        subgraph.compute_mask = torch.logical_and(compute_mask, partition_mask)

        return subgraph

    @torch.no_grad()
    def get_iterator(self) -> Iterator[Data]:
        data = self.data.to(self.device)  # type: ignore
        permutations = self.candidates[
            torch.randperm(self.num_candidates, pin_memory=self.pin_memory, device=self.device)
        ]
        for i in range(self.num_parts):
            partition_nodes = permutations[i * self.num_sample_per_batch : (i + 1) * self.num_sample_per_batch]
            data = self.add_compute_mask(data, partition_nodes)

            ppr_sampled_nodes = self.ppr_sampling(partition_nodes)

            subgraph = data.subgraph(ppr_sampled_nodes)
            if self.device != torch.device("mps"):  # Bug in MPS backend for sparse tensors
                subgraph = self.to_sparse_tensor(subgraph)
                subgraph.edge_index = subgraph.adj_t  # respect the common storage format for all models

            yield subgraph

        del data
        del permutations


class OnDiskMemmapsRandomNodeLoader:
    def __init__(
        self,
        dataset: Any,
        num_parts: int,
        use_blocks: bool = True,
        block_size: int = 1_000_000,
        num_workers: int = -1,
        split_type: SplitType = SplitType.TRAIN,
    ):
        self.dataset = dataset
        self.num_parts = num_parts
        self.split_type = split_type

        self.num_nodes = self.dataset.num_nodes

        self.use_blocks = use_blocks
        if self.use_blocks:
            self.block_size = block_size
            self.num_blocks = (self.num_nodes + block_size - 1) // block_size
        self.num_workers = num_workers

        self.num_sample_per_batch = (
            self.num_nodes // self.num_parts + 1
        )  # +1 to avoid having a last batch with less nodes than the others

        self.original_root = self.dataset.original_root
        self.memmaps_folder = os.path.join(self.original_root, "memmaps")
        os.makedirs(self.memmaps_folder, exist_ok=True)

        self.to_sparse_tensor = ToSparseTensor()

        self.process()

    def __len__(self) -> int:
        return self.num_parts

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def create_features_memmap(self) -> None:
        self.features_path = os.path.join(self.memmaps_folder, "node_feat.f32")
        if os.path.exists(self.features_path):
            print("Features memmap found")

            return

        print("Features memmap not found, creating...")
        features = np.load(os.path.join(self.original_root, "raw", "data.npz"))["node_feat"]
        features_mm = np.memmap(self.features_path, mode="w+", dtype=np.float32, shape=features.shape)
        features_mm[:] = features
        features_mm.flush()
        del features_mm, features

    def create_features_blocks_memmaps(self) -> None:
        self.features_block_folder = os.path.join(self.memmaps_folder, str(self.block_size))
        self.features_blocks_files = [
            os.path.join(self.features_block_folder, f"features_b{block_index}.f32")
            for block_index in range(self.num_blocks)
        ]
        self.features_blocks_counts_file = os.path.join(self.features_block_folder, "features_blocks_counts.npy")

        if all(os.path.exists(file) for file in self.features_blocks_files):
            print("Features block memmaps found")
            return

        print("Features block memmaps not found, creating...")
        os.makedirs(self.features_block_folder, exist_ok=True)
        features = np.load(os.path.join(self.original_root, "raw", "data.npz"))["node_feat"]

        counts_per_block = []
        for block_index in tqdm(range(self.num_blocks), total=self.num_blocks):
            start_idx = block_index * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_nodes)
            features_mm = np.memmap(
                self.features_blocks_files[block_index],
                mode="w+",
                dtype=np.float32,
                shape=(end_idx - start_idx, self.dataset.dim_features),
            )
            counts_per_block.append(end_idx - start_idx)
            features_mm[:] = features[start_idx:end_idx]
            features_mm.flush()
            del features_mm

        np.save(self.features_blocks_counts_file, np.array(counts_per_block))

    def create_edges_memmaps(self) -> None:
        self.edges_src_path = os.path.join(self.memmaps_folder, "edges_src.i64")
        self.edges_dst_path = os.path.join(self.memmaps_folder, "edges_dst.i64")
        if os.path.exists(self.edges_src_path) and os.path.exists(self.edges_dst_path):
            print("Edges src and dst memmaps found")
            return

        print("Edges src or dst memmap not found, creating...")
        edges_index = np.load(os.path.join(self.original_root, "raw", "data.npz"))["edge_index"]
        num_edges = edges_index.shape[1]
        self.num_edges = num_edges

        edges_src_mm = np.memmap(self.edges_src_path, mode="w+", dtype=np.int64, shape=(num_edges,))
        edges_src_mm[:] = edges_index[0]
        edges_src_mm.flush()
        del edges_src_mm

        edges_dst_mm = np.memmap(self.edges_dst_path, mode="w+", dtype=np.int64, shape=(num_edges,))
        edges_dst_mm[:] = edges_index[1]
        edges_dst_mm.flush()
        del edges_dst_mm

        del edges_index

    def create_edges_blocks_memmaps(self) -> None:
        self.edge_block_folder = os.path.join(self.memmaps_folder, str(self.block_size))
        self.edges_block_counts_file = os.path.join(self.edge_block_folder, "edges_block_counts.npy")
        self.edges_src_blocks_files = [
            os.path.join(self.edge_block_folder, f"edges_src_b{block_index}.i64")
            for block_index in range(self.num_blocks)
        ]
        self.edges_dst_blocks_files = [
            os.path.join(self.edge_block_folder, f"edges_dst_b{block_index}.i64")
            for block_index in range(self.num_blocks)
        ]

        if all(os.path.exists(file) for file in self.edges_src_blocks_files) and all(
            os.path.exists(file) for file in self.edges_dst_blocks_files
        ):
            print("Edge block memmaps found")
            return

        print("Edge block memmaps not found, creating...")
        os.makedirs(self.edge_block_folder, exist_ok=True)
        edges_index = np.load(os.path.join(self.original_root, "raw", "data.npz"))["edge_index"]

        edges_src = edges_index[0]
        edges_dst = edges_index[1]

        node_block_ids = edges_src // self.block_size

        counts_per_block = np.bincount(node_block_ids, minlength=self.num_blocks)

        np.save(self.edges_block_counts_file, counts_per_block)

        for block_index, count in tqdm(enumerate(counts_per_block), total=self.num_blocks):
            edges_src_mm = np.memmap(
                self.edges_src_blocks_files[block_index],
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            )
            edges_dst_mm = np.memmap(
                self.edges_dst_blocks_files[block_index],
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            )
            edges_src_mm[:] = edges_src[node_block_ids == block_index]
            edges_src_mm.flush()
            del edges_src_mm

            edges_dst_mm[:] = edges_dst[node_block_ids == block_index]
            edges_dst_mm.flush()
            del edges_dst_mm

        del edges_index

    def create_labels_memmap(self) -> None:
        self.labels_path = os.path.join(self.memmaps_folder, "labels.i64")
        if os.path.exists(self.labels_path):
            print("Labels memmap found")
            return

        print("Labels memmap not found, creating...")
        labels = np.load(os.path.join(self.original_root, "raw", "node-label.npz"))["node_label"].flatten()

        labels[np.isnan(labels)] = -1

        labels = labels.astype(np.int64)

        labels_mm = np.memmap(self.labels_path, mode="w+", dtype=np.int64, shape=labels.shape)
        labels_mm[:] = labels
        labels_mm.flush()
        del labels_mm, labels

    def create_split_memmaps(self) -> None:
        self.split_mask_path = os.path.join(self.memmaps_folder, "split_mask.i8")
        if os.path.exists(self.split_mask_path):
            print("Split mask memmap found")
            return

        print("Split mask memmap not found, creating...")
        split_folder = os.path.join(self.original_root, "split", "time")

        train_idx = pd.read_csv(os.path.join(split_folder, "train.csv.gz"), compression="gzip", header=None).values.T[0]
        valid_idx = pd.read_csv(os.path.join(split_folder, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
        test_idx = pd.read_csv(os.path.join(split_folder, "test.csv.gz"), compression="gzip", header=None).values.T[0]

        labels = np.load(os.path.join(self.original_root, "raw", "node-label.npz"))["node_label"].flatten()

        split_mask = np.zeros(labels.shape[0], dtype=np.int8)
        split_mask[train_idx] = 1  # train
        split_mask[valid_idx] = 2  # valid
        split_mask[test_idx] = 3  # test
        split_mask[np.isnan(labels)] = 0  # not use in split if no label

        split_mask_mm = np.memmap(self.split_mask_path, mode="w+", dtype=np.int8, shape=(labels.shape[0],))
        split_mask_mm[:] = split_mask
        split_mask_mm.flush()

        del split_mask_mm, split_mask, train_idx, valid_idx, test_idx, labels

    def set_getters(self):
        if self.use_blocks:
            self.edges_block_counts = np.load(self.edges_block_counts_file)
            self.features_blocks_counts = np.load(self.features_blocks_counts_file)

            self.features_blocks_mm = [
                np.memmap(file, mode="r", dtype=np.float32, shape=(count, self.dataset.dim_features))
                for file, count in zip(self.features_blocks_files, self.features_blocks_counts, strict=True)
            ]
            self.edges_blocks_src_mm = [
                np.memmap(file, mode="r", dtype=np.int64, shape=(count,))
                for file, count in zip(self.edges_src_blocks_files, self.edges_block_counts, strict=True)
            ]
            self.edges_blocks_dst_mm = [
                np.memmap(file, mode="r", dtype=np.int64, shape=(count,))
                for file, count in zip(self.edges_dst_blocks_files, self.edges_block_counts, strict=True)
            ]
        else:
            self.features_mm = np.memmap(
                self.features_path, mode="r", dtype=np.float32, shape=(self.num_nodes, self.dataset.dim_features)
            )
            self.edges_src_mm = np.memmap(self.edges_src_path, mode="r", dtype=np.int64, shape=(self.num_edges,))
            self.edges_dst_mm = np.memmap(self.edges_dst_path, mode="r", dtype=np.int64, shape=(self.num_edges,))

        self.labels_mm = np.memmap(self.labels_path, mode="r", dtype=np.int64, shape=(self.num_nodes,))
        self.split_mask_mm = np.memmap(self.split_mask_path, mode="r", dtype=np.int8, shape=(self.num_nodes,))

    def process(self):
        if self.use_blocks:
            self.create_features_blocks_memmaps()
            self.create_edges_blocks_memmaps()
        else:
            self.create_features_memmap()
            self.create_edges_memmaps()
        self.create_labels_memmap()
        self.create_split_memmaps()
        self.set_getters()

    def process_features_block(self, block_idx: int, batch_idx: np.ndarray) -> np.ndarray:
        block_start = block_idx * self.block_size
        block_end = block_start + self.block_size
        local_idx = batch_idx[(batch_idx >= block_start) & (batch_idx < block_end)] - block_start
        return self.features_blocks_mm[block_idx][local_idx]

    def get_features_blocks(self, batch_idx: np.ndarray, node_mask: np.ndarray) -> np.ndarray:  # noqa: ARG002
        block_ids = np.unique(batch_idx // self.block_size)

        accumulator = Parallel(n_jobs=self.num_workers)(
            delayed(self.process_features_block)(b, batch_idx) for b in block_ids
        )

        return np.concatenate(accumulator, axis=0)  # type: ignore

    def get_edges(self, node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge_mask = node_mask[self.edges_src_mm] & node_mask[self.edges_dst_mm]
        edges_src = self.edges_src_mm[edge_mask]
        edges_dst = self.edges_dst_mm[edge_mask]

        return edges_src, edges_dst

    def process_edge_block(self, block_idx: int, node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges_src_mm = self.edges_blocks_src_mm[block_idx]
        edges_dst_mm = self.edges_blocks_dst_mm[block_idx]

        edge_src_mask = node_mask[edges_src_mm]

        filtered_src = edges_src_mm[edge_src_mask]
        filtered_dst = edges_dst_mm[edge_src_mask]

        edge_dst_mask = node_mask[filtered_dst]

        return filtered_src[edge_dst_mask], filtered_dst[edge_dst_mask]

    def get_edges_blocks(self, node_idx: np.ndarray, node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block_ids = np.unique(node_idx // self.block_size)

        results = Parallel(n_jobs=self.num_workers)(delayed(self.process_edge_block)(b, node_mask) for b in block_ids)

        edges_src, edges_dst = map(np.concatenate, zip(*results, strict=True))

        return edges_src, edges_dst

    def relabel_edges(
        self, batch_idx: np.ndarray, edges_src: np.ndarray, edges_dst: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        local_src = np.searchsorted(batch_idx, edges_src)
        local_dst = np.searchsorted(batch_idx, edges_dst)

        return local_src, local_dst

    def get_iterator(self) -> Iterator[Data]:
        permutations = np.random.permutation(self.num_nodes)

        for i in range(self.num_parts):
            start_idx = i * self.num_sample_per_batch
            end_idx = start_idx + self.num_sample_per_batch
            batch_idx = permutations[start_idx:end_idx]
            batch_idx = np.sort(batch_idx)

            node_mask = np.zeros(self.num_nodes, dtype=np.bool_)
            node_mask[batch_idx] = True

            if self.use_blocks:
                features = self.get_features_blocks(batch_idx, node_mask)
            else:
                features = self.features_mm[node_mask]

            if self.use_blocks:
                edges_src, edges_dst = self.get_edges_blocks(batch_idx, node_mask)
            else:
                edges_src, edges_dst = self.get_edges(node_mask)

            edges_src, edges_dst = self.relabel_edges(batch_idx, edges_src, edges_dst)

            labels = self.labels_mm[node_mask]

            split_mask = self.split_mask_mm[node_mask]

            subgraph = Data(
                x=torch.from_numpy(features),
                edge_index=torch.from_numpy(np.stack([edges_src, edges_dst], axis=0)),
                y=torch.from_numpy(labels),
                train_mask=torch.from_numpy(split_mask == 1),
                val_mask=torch.from_numpy(split_mask == 2),  # noqa: PLR2004
                test_mask=torch.from_numpy(split_mask == 3),  # noqa: PLR2004
            )

            subgraph = add_compute_mask(subgraph, self.split_type)

            subgraph = self.to_sparse_tensor(subgraph)
            subgraph.edge_index = subgraph.adj_t  # respect the common storage format for all models

            yield subgraph


class LinkLoader:
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        mask_loss_edges: bool = True,
        split_type: SplitType = SplitType.TRAIN,
        max_iterations: int = 3,
        negative_sampling_ratio: float = 0.5,
    ):
        self.data = dataset.data
        self.num_nodes = dataset.num_nodes
        self.batch_size = batch_size
        self.mask_loss_edges = mask_loss_edges
        self.split_type = split_type
        self.max_iterations = max_iterations
        self.negative_per_batch = int(batch_size * negative_sampling_ratio)
        self.positive_per_batch = batch_size - self.negative_per_batch

        match split_type:
            case SplitType.TRAIN:
                self.positive_edges, self.non_negative_edges_ids = self.cannonize_positive_edges(
                    dataset, add_self_loops=True
                )

            case SplitType.VAL:
                splits = dataset.get_edge_split()
                positive_edges = splits["valid"]["edge"].T
                negative_edges = splits["valid"]["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                )
            case SplitType.TEST:
                splits = dataset.get_edge_split()
                positive_edges = splits["test"]["edge"].T
                negative_edges = splits["test"]["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                )
            case _:
                raise ValueError(f"Invalid split type: {split_type}")

    def cannonize_positive_edges(self, dataset: Any, add_self_loops: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        positive_edges = dataset.get_edge_split()["train"]["edge"].T  # [2, n]
        positive_edges = torch.sort(positive_edges, dim=0).values
        if add_self_loops:
            non_negative_edges = torch.cat([positive_edges, torch.arange(self.num_nodes).repeat(2, 1)], dim=1)
        else:
            non_negative_edges = positive_edges

        non_negative_edges_ids = non_negative_edges[0, :] * self.num_nodes + non_negative_edges[1, :]

        return positive_edges, non_negative_edges_ids.unique()

    def rejection_sampling_negative_edges(self) -> torch.Tensor:
        candidates = torch.empty((2, 0), dtype=torch.int64)

        for _ in range(self.max_iterations):
            negative_candidates = torch.randint(0, self.num_nodes, (2, self.negative_per_batch))
            src = torch.minimum(negative_candidates[0, :], negative_candidates[1, :])
            dst = torch.maximum(negative_candidates[0, :], negative_candidates[1, :])

            candidate_edges_ids = src * self.num_nodes + dst

            idx = torch.searchsorted(self.non_negative_edges_ids, candidate_edges_ids)
            is_positive = torch.logical_and(
                idx < len(self.non_negative_edges_ids),
                self.non_negative_edges_ids[idx] == candidate_edges_ids,
            )

            candidates = torch.cat([candidates, negative_candidates[:, ~is_positive]], dim=1)

            if candidates.shape[1] >= self.negative_per_batch:
                return candidates[:, : self.negative_per_batch]
        return candidates[:, : self.negative_per_batch]

    def __len__(self) -> int:
        if self.split_type == SplitType.TRAIN:
            return self.positive_edges.shape[1] // self.positive_per_batch

        return self.target_edges.shape[1] // self.batch_size

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        if self.split_type == SplitType.TRAIN:
            for start_idx in range(0, self.positive_edges.shape[1], self.positive_per_batch):
                end_idx = start_idx + self.positive_per_batch

                positive_edges = self.positive_edges[:, start_idx:end_idx]
                negative_edges = self.rejection_sampling_negative_edges()
                target_edges = torch.cat([positive_edges, negative_edges], dim=1)
                labels = torch.cat([torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0)
                if self.mask_loss_edges:
                    edge_index = torch.cat(
                        [self.positive_edges[:, :start_idx], self.positive_edges[:, end_idx:]], dim=1
                    )
                    data = Data(
                        x=self.data.x,
                        edge_index=to_undirected(edge_index),
                        target_edges=target_edges,
                        y=labels,
                    )
                else:
                    data = self.data.clone()
                    data.target_edges = target_edges
                    data.y = labels

                yield data

        else:
            for start_idx in range(0, self.target_edges.shape[1], self.batch_size):
                end_idx = start_idx + self.batch_size
                target_edges = self.target_edges[:, start_idx:end_idx]
                labels = self.labels[start_idx:end_idx]
                data = self.data.clone()
                data.target_edges = target_edges
                data.y = labels

                yield data
