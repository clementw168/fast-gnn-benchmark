import random
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_sparse import coalesce

from fast_gnn_benchmark.data.link_dataloader import cannonize_positive_edges, rejection_sampling_negative_edges
from fast_gnn_benchmark.schemas.dataset_models import SplitType


def get_positive_edges(dataset: Any, split_type: SplitType) -> torch.Tensor:
    if split_type == SplitType.TRAIN:
        return dataset.split["train"]["edge"].T

    if split_type == SplitType.VAL:
        return dataset.split["valid"]["edge"].T

    if split_type == SplitType.TEST:
        return dataset.split["test"]["edge"].T

    raise ValueError(f"Invalid split type: {split_type}")


def get_negative_edges(dataset: Any, split_type: SplitType) -> torch.Tensor:
    device = torch.device("cpu")
    if split_type == SplitType.TRAIN:
        positive_edges, non_negative_edges_ids = cannonize_positive_edges(
            dataset, dataset.num_nodes, device, remove_self_loops=True
        )
        return rejection_sampling_negative_edges(
            positive_edges.shape[1],
            non_negative_edges_ids,
            dataset.num_nodes,
            2,
            device,
        )

    if split_type == SplitType.VAL:
        return dataset.split["valid"]["edge_neg"].T

    if split_type == SplitType.TEST:
        return dataset.split["test"]["edge_neg"].T

    raise ValueError(f"Invalid split type: {split_type}")


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A,
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, max_nodes_per_hop=None, node_features=None, y=1):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops + 1):
        fringe = neighbors(fringe, A)

        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if max_nodes_per_hop is not None and max_nodes_per_hop < len(fringe):
            fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label="drnl"):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = sparse.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == "drnl":  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == "hop":  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == "zo":  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == "de":  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == "de+":
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == "degree":  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, node_id=node_ids, num_nodes=num_nodes)
    return data


class SealDatasetScipy(Dataset):
    def __init__(
        self,
        dataset: Any,
        num_hops: int,
        max_rejection_sampling_iterations: int = 3,
        negative_sampling_ratio: float = 0.5,
        split_type: SplitType = SplitType.TRAIN,
    ):
        self.data = dataset.data
        self.num_hops = num_hops
        self.max_rejection_sampling_iterations = max_rejection_sampling_iterations
        self.negative_sampling_ratio = negative_sampling_ratio
        self.split_type = split_type

        positive_edges = get_positive_edges(dataset, split_type)
        negative_edges = get_negative_edges(dataset, split_type)

        self.target_edges = torch.cat([positive_edges, negative_edges], dim=1)
        self.labels = torch.cat(
            [
                torch.ones(positive_edges.shape[1]),
                torch.zeros(negative_edges.shape[1]),
            ],
            dim=0,
        )

        edge_weights = torch.ones(self.data.edge_index.size(1), dtype=torch.int32)

        self.scipy_adjacency_matrix = sparse.csr_matrix(
            (edge_weights, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes),
        )

    def __len__(self) -> int:
        return self.target_edges.shape[1]

    def __getitem__(self, index: int) -> Data:  # type: ignore
        src, dst = self.target_edges[:, index]
        label = self.labels[index]

        nodes, subgraph, dists, node_features, y = k_hop_subgraph(src, dst, self.num_hops, self.scipy_adjacency_matrix)

        return construct_pyg_graph(nodes, subgraph, dists, node_features, label)


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    from fast_gnn_benchmark.data.dataset.ogbl import FixLinkPropPredDataset
    from fast_gnn_benchmark.data.dataset.ogbn import OGBNDataset
    from fast_gnn_benchmark.data.utils import to_undirected

    device = torch.device("cuda")

    dataset = FixLinkPropPredDataset(name="ogbl-ppa", root="./datasets/ogbl/")

    data = dataset[0].to(device)  # type: ignore
    data.edge_index = to_undirected(data.edge_index)  # type: ignore

    seal_dataset = SealDatasetScipy(dataset, num_hops=2)

    data_loader = DataLoader(seal_dataset, batch_size=64, shuffle=True, num_workers=16)

    for batch in tqdm(data_loader):
        # print(batch)
        pass
