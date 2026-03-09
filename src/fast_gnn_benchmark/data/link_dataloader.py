from collections.abc import Iterator
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

from fast_gnn_benchmark.data.utils import to_undirected
from fast_gnn_benchmark.schemas.dataset_models import SplitType


def cannonize_positive_edges(
    dataset: Any,
    num_nodes: int,
    device: torch.device,
    remove_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_edges = dataset.split["train"]["edge"].T  # [2, n]
    positive_edges = torch.sort(positive_edges, dim=0).values
    if remove_self_loops:
        non_negative_edges = torch.cat([positive_edges, torch.arange(num_nodes).repeat(2, 1)], dim=1)
    else:
        non_negative_edges = positive_edges

    non_negative_edges_ids = non_negative_edges[0, :] * num_nodes + non_negative_edges[1, :]

    positive_edges = positive_edges.to(device)
    non_negative_edges_ids = non_negative_edges_ids.unique().to(device)

    return positive_edges, non_negative_edges_ids


def get_number_of_negative_candidates(
    iteration_index: int,
    number_of_samples: int,
    non_negative_edges_ids: torch.Tensor,
    num_nodes: int,
) -> int:
    """
    At the first iteration, we start with a large number of negative candidates and then decrease it as we get more negative edges.
    The expected number of negative candidates to get the desired number of negative edges is given by the formula:
    negative_per_batch / (1 - non_negative_proportion). A factor 4 is used to avoid doing a second iteration.
    The number 20 candidates for other iterations is an arbitrary number since there should not be many missing negative edges.
    """
    non_negative_proportion = non_negative_edges_ids.shape[0] / (num_nodes * num_nodes)
    if iteration_index == 0:
        return int(number_of_samples / (1 - non_negative_proportion * 4))

    return 20


def rejection_sampling_negative_edges(
    number_of_samples: int,
    non_negative_edges_ids: torch.Tensor,
    num_nodes: int,
    max_rejection_sampling_iterations: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Rejection sampling negative edges.
    Basically, we sample negative edges randomly and then reject the ones that are already in the positive edges.
    To do that, we first attribute a unique id function of each edge which is simply min(src, dst) * num_nodes + max(src, dst).

    We then sample a number of negative candidates randomly.
    We use a binary search to find the index of the candidate edges in the non_negative_edges_ids tensor.
    If the edge is in the non_negative_edges_ids tensor, it is a positive edge and we reject it.
    Otherwise, we add it to the candidates tensor.

    We repeat this process for a maximum of max_rejection_sampling_iterations times.

    Note that we do not ensure that the negative edges are unique since the check would be too expensive.

    """

    candidate_list: list[torch.Tensor] = []
    total_collected = 0

    for sampling_iteration_index in range(max_rejection_sampling_iterations):
        number_of_negative_candidates = get_number_of_negative_candidates(
            sampling_iteration_index, number_of_samples, non_negative_edges_ids, num_nodes
        )
        potential_negative_candidates = torch.randint(0, num_nodes, (2, number_of_negative_candidates), device=device)
        src = torch.minimum(potential_negative_candidates[0, :], potential_negative_candidates[1, :])
        dst = torch.maximum(potential_negative_candidates[0, :], potential_negative_candidates[1, :])

        candidate_edges_ids = src * num_nodes + dst

        non_negative_edges_ids_index = torch.searchsorted(non_negative_edges_ids, candidate_edges_ids)
        is_positive_edge = torch.logical_and(
            non_negative_edges_ids_index < len(non_negative_edges_ids),
            non_negative_edges_ids[non_negative_edges_ids_index] == candidate_edges_ids,
        )

        neg = potential_negative_candidates[:, ~is_positive_edge]
        candidate_list.append(neg)
        total_collected += neg.shape[1]

        if total_collected >= number_of_samples:
            candidates = torch.cat(candidate_list, dim=1)
            return candidates[:, :number_of_samples]

    candidates = torch.cat(candidate_list, dim=1)
    return candidates[:, :number_of_samples]


class LinkLoader:
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        mask_loss_edges: bool = True,
        max_rejection_sampling_iterations: int = 3,
        negative_sampling_ratio: float = 0.5,
        on_device=True,
        split_type: SplitType = SplitType.TRAIN,
    ):
        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.data = dataset.data.to(self.device)
        self.num_nodes = dataset.num_nodes
        self.batch_size = batch_size
        self.mask_loss_edges = mask_loss_edges
        self.split_type = split_type
        self.max_rejection_sampling_iterations = max_rejection_sampling_iterations
        self.positive_per_batch = batch_size - int(batch_size * negative_sampling_ratio)

        self.to_sparse_tensor = ToSparseTensor()

        match split_type:
            case SplitType.TRAIN:
                self.positive_edges, self.non_negative_edges_ids = cannonize_positive_edges(
                    dataset, self.num_nodes, self.device, remove_self_loops=True
                )

            case SplitType.VAL:
                splits = dataset.split["valid"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                ).to(self.device)
            case SplitType.TEST:
                splits = dataset.split["test"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                ).to(self.device)
            case _:
                raise ValueError(f"Invalid split type: {split_type}")

    def __len__(self) -> int:
        if self.split_type == SplitType.TRAIN:
            return max(self.positive_edges.shape[1] // self.positive_per_batch, 1)

        return max(self.target_edges.shape[1] // self.batch_size, 1)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        if self.split_type == SplitType.TRAIN:
            random_shuffle = torch.randperm(self.positive_edges.shape[1])
            self.positive_edges = self.positive_edges[:, random_shuffle]

            for start_idx in range(0, self.positive_edges.shape[1], self.positive_per_batch):
                end_idx = start_idx + self.positive_per_batch

                positive_edges = self.positive_edges[:, start_idx:end_idx]
                negative_edges = rejection_sampling_negative_edges(
                    positive_edges.shape[1],
                    self.non_negative_edges_ids,
                    self.num_nodes,
                    self.max_rejection_sampling_iterations,
                    self.device,
                )
                target_edges = torch.cat([positive_edges, negative_edges], dim=1)
                labels = torch.cat(
                    [
                        torch.ones(positive_edges.shape[1], device=self.device),
                        torch.zeros(negative_edges.shape[1], device=self.device),
                    ],
                    dim=0,
                )
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
                    data = Data(
                        x=self.data.x,
                        edge_index=self.data.edge_index,
                        target_edges=target_edges,
                        y=labels,
                    )

                data = self.to_sparse_tensor(data)
                data.edge_index = data.adj_t

                yield data

        else:
            for start_idx in range(0, self.target_edges.shape[1], self.batch_size):
                end_idx = start_idx + self.batch_size
                target_edges = self.target_edges[:, start_idx:end_idx]
                labels = self.labels[start_idx:end_idx]
                data = Data(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    target_edges=target_edges,
                    y=labels,
                )

                data = self.to_sparse_tensor(data)
                data.edge_index = data.adj_t

                yield data


def get_khop_neighbors(
    nodes: torch.Tensor, edge_index: torch.Tensor, k_list: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the k-hop neighbors of the nodes.
    Returns the k-hop neighbors for each node in the nodes tensor.
    """
    separator = []
    khop_neighbors = []

    for node in nodes:
        previous_neighbors = torch.tensor([node])
        for k in k_list:
            current_neighbors = torch.unique(
                torch.cat(
                    [
                        edge_index[1, torch.any(edge_index[0] == previous_neighbors, dim=0)],
                        edge_index[0, torch.any(edge_index[1] == previous_neighbors, dim=0)],
                    ],
                    dim=0,
                )
            )
            previous_neighbors = current_neighbors
        khop_neighbors.append(current_neighbors)
        separator.append(node)

    neighbors = []
    for node in nodes:
        neighbors.append(torch.unique(edge_index[1][edge_index[0] == node]))
    return neighbors


class SealLoader:
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        num_neighbors: list[int],
        max_rejection_sampling_iterations: int = 3,
        negative_sampling_ratio: float = 0.5,
        on_device: bool = True,
        split_type: SplitType = SplitType.TRAIN,
    ):
        if on_device:
            self.device = torch.accelerator.current_accelerator() or torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.data = dataset.data.to(self.device)
        self.num_nodes = dataset.num_nodes
        self.batch_size = batch_size
        self.split_type = split_type
        self.num_neighbors = num_neighbors
        self.max_rejection_sampling_iterations = max_rejection_sampling_iterations
        self.positive_per_batch = batch_size - int(batch_size * negative_sampling_ratio)

        self.to_sparse_tensor = ToSparseTensor()

        match split_type:
            case SplitType.TRAIN:
                self.positive_edges, self.non_negative_edges_ids = cannonize_positive_edges(
                    dataset, self.num_nodes, self.device, remove_self_loops=True
                )

            case SplitType.VAL:
                splits = dataset.split["valid"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                ).to(self.device)

            case SplitType.TEST:
                splits = dataset.split["test"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                ).to(self.device)
            case _:
                raise ValueError(f"Invalid split type: {split_type}")

    def __len__(self) -> int:
        return max(self.target_edges.shape[1] // self.batch_size, 1)

    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()

    def get_iterator(self) -> Iterator[Data]:
        if self.split_type == SplitType.TRAIN:
            for start_idx in range(0, self.positive_edges.shape[1], self.positive_per_batch):
                end_idx = start_idx + self.positive_per_batch

                positive_edges = self.positive_edges[:, start_idx:end_idx]
                negative_edges = rejection_sampling_negative_edges(
                    positive_edges.shape[1],
                    self.non_negative_edges_ids,
                    self.num_nodes,
                    self.max_rejection_sampling_iterations,
                    self.device,
                )
                target_edges = torch.cat([positive_edges, negative_edges], dim=1)
                labels = torch.cat(
                    [
                        torch.ones(positive_edges.shape[1], device=self.device),
                        torch.zeros(negative_edges.shape[1], device=self.device),
                    ],
                    dim=0,
                )

                positive_src = positive_edges[0, :]
                positive_dst = positive_edges[1, :]
                negative_src = negative_edges[0, :]
                negative_dst = negative_edges[1, :]

                positive_src_neighbors = get_khop_neighbors(positive_src, self.data.edge_index, self.num_neighbors)
                positive_dst_neighbors = get_khop_neighbors(positive_dst, self.data.edge_index, self.num_neighbors)
                negative_src_neighbors = get_khop_neighbors(negative_src, self.data.edge_index, self.num_neighbors)
                negative_dst_neighbors = get_khop_neighbors(negative_dst, self.data.edge_index, self.num_neighbors)

                data = Data(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    target_edges=target_edges,
                    y=labels,
                )

                data = self.to_sparse_tensor(data)
                data.edge_index = data.adj_t

                yield data

        else:
            for start_idx in range(0, self.target_edges.shape[1], self.batch_size):
                end_idx = start_idx + self.batch_size
                target_edges = self.target_edges[:, start_idx:end_idx]
                labels = self.labels[start_idx:end_idx]
                data = Data(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    target_edges=target_edges,
                    y=labels,
                )

                data = self.to_sparse_tensor(data)
                data.edge_index = data.adj_t

                yield data

    def get_enclosing_graphs_features(self, k_hop_neighbors: list[torch.Tensor]) -> torch.Tensor:
        pass

    def add_positional_encoding(self, enclosing_features: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        pass

    def create_pyg_data(self, enclosing_features: torch.Tensor) -> Data:
        pass
