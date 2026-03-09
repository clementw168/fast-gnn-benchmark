from collections.abc import Iterator
from math import ceil
from typing import Any

import torch
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor

from fast_gnn_benchmark.data.node_dataloaders import SplitType


def get_next_hop_neighbors(
    src_nodes: torch.Tensor,
    CSR_adjacency_matrix: SparseTensor,
    max_neighbors: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample up to max_neighbors (without replacement) per source from the CSR adjacency. Padded with -1 if the number of neighbors is less than max_neighbors."""
    col = CSR_adjacency_matrix.storage.col()
    rowptr = CSR_adjacency_matrix.storage.rowptr()
    B = src_nodes.shape[0]

    row_start = rowptr[src_nodes]
    row_end = rowptr[src_nodes + 1]
    degree = row_end - row_start
    max_deg_batch = int(degree.max().item())

    if max_deg_batch == 0:
        return torch.full((B, max_neighbors), -1, dtype=torch.long, device=device)

    # [B, max_deg_batch] linear indices into col
    indices = row_start.unsqueeze(1) + torch.arange(max_deg_batch, device=device).unsqueeze(0)
    # Clamp to avoid out-of-bounds for rows with degree < max_deg_batch
    indices = torch.minimum(indices, (row_end - 1).clamp(min=0).unsqueeze(1))

    all_neighbors = col[indices]
    mask = torch.arange(max_deg_batch, device=device).unsqueeze(0) < degree.unsqueeze(1)

    # Random permutation per row; put valid positions first
    perm = torch.argsort(torch.rand(B, max_deg_batch, device=device), dim=1)
    valid_first = mask[torch.arange(B, device=device)[:, None], perm]
    order = torch.argsort((~valid_first).long(), dim=1)
    perm_ordered = torch.gather(perm, 1, order)

    indices_out = perm_ordered[:, :max_neighbors]
    valid_out = mask[torch.arange(B, device=device)[:, None], indices_out]

    sampled_neighbors = torch.gather(all_neighbors, 1, indices_out)
    sampled_neighbors[~valid_out] = -1

    return sampled_neighbors


def flatten_hop_neighbors(hop_neighbors: torch.Tensor, src_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # hop_neighbors: (B, k), src_nodes: (B,) -> align so each neighbor has its source index
    if src_nodes.dim() == 1:
        src_nodes = src_nodes.unsqueeze(1)  # (B, 1)
    hop_src = src_nodes.expand(-1, hop_neighbors.shape[1])  # (B, k)

    filter_mask = hop_neighbors != -1
    hop_src = hop_src[filter_mask]
    hop_neighbors = hop_neighbors[filter_mask]

    return hop_neighbors, hop_src


def get_k_hop_neighbors(
    src_nodes: torch.Tensor,
    CSR_adjacency_matrix: SparseTensor,
    k_list: list[int],
    device: torch.device,
    sort_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    neighbors_accumulator = [src_nodes]
    src_accumulator = [src_nodes]

    hop_src = src_nodes
    for k in k_list:
        current_hop_neighbors = get_next_hop_neighbors(hop_src, CSR_adjacency_matrix, max_neighbors=k, device=device)
        hop_neighbors, hop_src = flatten_hop_neighbors(current_hop_neighbors, hop_src)

        neighbors_accumulator.append(hop_neighbors)
        src_accumulator.append(hop_src)

    neighbors = torch.cat(neighbors_accumulator, dim=0)
    src = torch.cat(src_accumulator, dim=0)
    if sort_output:
        indices = torch.argsort(src, dim=1)
        return neighbors[indices], src[indices]

    return neighbors, src


class KHopNeighborsLoader:
    """
    K-hop neighbors node data loader. This loader is faster than the NeighborLoaderWrapper.
    """

    def __init__(
        self,
        dataset: Any,
        num_neighbors: list[int],
        batch_size: int,
        on_device: bool = True,
        split_type: SplitType = SplitType.TRAIN,
    ):
        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.data = dataset[0].to(self.device)
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size

        self.to_sparse_tensor = ToSparseTensor()
        self.CSR_adjacency_matrix = self.to_sparse_tensor(self.data).adj_t
        self.split_type = split_type

        match self.split_type:
            case SplitType.TRAIN:
                mask = self.data.train_mask
            case SplitType.VAL:
                mask = self.data.val_mask
            case SplitType.TEST:
                mask = self.data.test_mask
            case _:
                raise ValueError(f"Invalid split type: {self.split_type}")

        self.candidate_src_nodes = torch.where(mask)[0].to(self.device)

    def __len__(self) -> int:
        return ceil(len(self.candidate_src_nodes) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        random_indices = torch.randperm(len(self.candidate_src_nodes), device=self.device)
        for start_idx in range(0, len(self.candidate_src_nodes), self.batch_size):
            src_nodes = self.candidate_src_nodes[random_indices[start_idx : start_idx + self.batch_size]]

            neighbors, neighbors_src = get_k_hop_neighbors(
                src_nodes, self.CSR_adjacency_matrix, self.num_neighbors, self.device
            )
            subgraph = self.data.subgraph(neighbors)
            subgraph = self.to_sparse_tensor(subgraph)
            subgraph.edge_index = subgraph.adj_t
            subgraph.compute_mask = neighbors == neighbors_src
            yield subgraph


class SealLoader:
    def __init__(self, dataset: Any, batch_size: int, on_device: bool = True, split_type: SplitType = SplitType.TRAIN):
        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.data = dataset[0].to(self.device)
        self.batch_size = batch_size
        self.split_type = split_type


if __name__ == "__main__":
    from tqdm import tqdm

    from fast_gnn_benchmark.data.dataset.ogbl import FixLinkPropPredDataset
    from fast_gnn_benchmark.data.dataset.ogbn import OGBNDataset
    from fast_gnn_benchmark.data.utils import to_undirected

    # device = torch.device("cuda")

    # dataset = FixLinkPropPredDataset(name="ogbl-ppa", root="./datasets/ogbl/")

    # data = dataset[0].to(device)  # type: ignore
    # data.edge_index = to_undirected(data.edge_index)  # type: ignore

    # to_sparse_tensor = ToSparseTensor()
    # data = to_sparse_tensor(data)
    # CSR_adjacency_matrix = data.adj_t

    # print(CSR_adjacency_matrix)
    # print(CSR_adjacency_matrix.nnz())
    # print(CSR_adjacency_matrix.storage.rowptr().shape)
    # print(CSR_adjacency_matrix.storage.col().shape)

    # src_nodes = torch.randint(0, 500000, (128,), device=device)

    # # sampled_neighbors = get_next_hop_neighbors(src_nodes, CSR_adjacency_matrix, 5, device)
    # # print(sampled_neighbors)

    # neighbors, neighbors_src = get_k_hop_neighbors(src_nodes, CSR_adjacency_matrix, k_list=[50, 10], device=device)

    # print(neighbors.shape)
    # print(neighbors_src.shape)

    dataset = OGBNDataset(name="ogbn-arxiv", root="./datasets/ogbn/")

    print(dataset[0])

    dataset[0].edge_index = to_undirected(dataset[0].edge_index)

    loader = KHopNeighborsLoader(dataset, num_neighbors=[50, 10], batch_size=128, on_device=True)

    # loader = NeighborLoaderWrapper(
    #     dataset, num_neighbors=[50, 10], batch_size=128, shuffle=True, on_device=True, num_workers=16
    # )

    from time import time

    start_time = time()
    for i in range(10):
        for subgraph in loader:
            pass

    print(f"Time taken: {time() - start_time} seconds")
