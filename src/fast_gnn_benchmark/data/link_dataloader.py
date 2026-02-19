from collections.abc import Iterator
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

from fast_gnn_benchmark.data.utils import to_undirected
from fast_gnn_benchmark.schemas.dataset_models import SplitType


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
        self.negative_per_batch = int(batch_size * negative_sampling_ratio)
        self.positive_per_batch = batch_size - self.negative_per_batch

        self.to_sparse_tensor = ToSparseTensor()

        match split_type:
            case SplitType.TRAIN:
                self.positive_edges, self.non_negative_edges_ids = self.cannonize_positive_edges(
                    dataset, remove_self_loops=True
                )

            case SplitType.VAL:
                splits = dataset.get_edge_split()
                positive_edges = splits["valid"]["edge"].T
                negative_edges = splits["valid"]["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                ).to(self.device)
            case SplitType.TEST:
                splits = dataset.get_edge_split()
                positive_edges = splits["test"]["edge"].T
                negative_edges = splits["test"]["edge_neg"].T
                self.target_edges = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
                self.labels = torch.cat(
                    [torch.ones(positive_edges.shape[1]), torch.zeros(negative_edges.shape[1])], dim=0
                ).to(self.device)
            case _:
                raise ValueError(f"Invalid split type: {split_type}")

    def cannonize_positive_edges(
        self, dataset: Any, remove_self_loops: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_edges = dataset.get_edge_split()["train"]["edge"].T  # [2, n]
        positive_edges = torch.sort(positive_edges, dim=0).values
        if remove_self_loops:
            non_negative_edges = torch.cat([positive_edges, torch.arange(self.num_nodes).repeat(2, 1)], dim=1)
        else:
            non_negative_edges = positive_edges

        non_negative_edges_ids = non_negative_edges[0, :] * self.num_nodes + non_negative_edges[1, :]

        positive_edges = positive_edges.to(self.device)
        non_negative_edges_ids = non_negative_edges_ids.unique().to(self.device)

        return positive_edges, non_negative_edges_ids

    def rejection_sampling_negative_edges(self) -> torch.Tensor:
        candidates = torch.empty((2, 0), dtype=torch.int64, device=self.device)

        for _ in range(self.max_rejection_sampling_iterations):
            negative_candidates = torch.randint(0, self.num_nodes, (2, self.negative_per_batch), device=self.device)
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
                labels = torch.cat(
                    [
                        torch.ones(positive_edges.shape[1]),
                        torch.zeros(negative_edges.shape[1]),
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
                    data = self.data.clone()
                    data.target_edges = target_edges
                    data.y = labels

                data = self.to_sparse_tensor(data)
                data.edge_index = data.adj_t

                yield data

        else:
            for start_idx in range(0, self.target_edges.shape[1], self.batch_size):
                end_idx = start_idx + self.batch_size
                target_edges = self.target_edges[:, start_idx:end_idx]
                labels = self.labels[start_idx:end_idx]
                data = self.data.clone()
                data.target_edges = target_edges
                data.y = labels

                data = self.to_sparse_tensor(data)
                data.edge_index = data.adj_t

                yield data
