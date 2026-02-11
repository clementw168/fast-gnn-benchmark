import torch
from torch_geometric import utils
from torch_geometric.data import Data


def random_subsample(data: Data, proportion: float) -> None:
    """
    Modify the dataset in place to subsample a proportion of the nodes.
    """
    assert data.num_nodes is not None, "Data must have num_nodes"
    assert data.num_nodes > 0, "Data must have num_nodes"
    assert proportion > 0 and proportion < 1, "Proportion must be between 0 and 1"
    assert data.x is not None, "Data must have x"
    assert isinstance(data.y, torch.Tensor), "Data must have y"

    nodes_to_keep = torch.randperm(data.num_nodes)[: int(data.num_nodes * proportion)]

    mapping = torch.zeros(data.num_nodes, dtype=torch.long) - 1

    data.x = data.x[nodes_to_keep]
    data.y = data.y[nodes_to_keep]

    mapping[nodes_to_keep] = torch.arange(nodes_to_keep.size(0))

    edge_index = mapping[data.edge_index]
    data.edge_index = edge_index[:, (edge_index != -1).all(dim=0)]

    if hasattr(data, "train_mask"):
        data.train_mask = data.train_mask[nodes_to_keep]
    if hasattr(data, "val_mask"):
        data.val_mask = data.val_mask[nodes_to_keep]
    if hasattr(data, "test_mask"):
        data.test_mask = data.test_mask[nodes_to_keep]


def remove_isolated_nodes(data: Data) -> None:
    """
    Modify the dataset in place to remove isolated nodes.
    """
    assert data.num_nodes is not None, "Data must have num_nodes"
    assert data.num_nodes > 0, "Data must have num_nodes"
    assert data.edge_index is not None, "Data must have edge_index"
    assert data.x is not None, "Data must have x"
    assert isinstance(data.y, torch.Tensor), "Data must have y"

    edge_index, _, mask = utils.remove_isolated_nodes(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index
    data.x = data.x[mask]
    data.y = data.y[mask]

    # If you have masks:
    if hasattr(data, "train_mask"):
        data.train_mask = data.train_mask[mask]
    if hasattr(data, "val_mask"):
        data.val_mask = data.val_mask[mask]
    if hasattr(data, "test_mask"):
        data.test_mask = data.test_mask[mask]


def print_data_properties(data: Data, show_all: bool = False) -> None:
    """
    Print the properties of the dataset.
    """
    assert data.edge_index is not None, "Data must have edge_index"
    assert isinstance(data.y, torch.Tensor), "Data must have y"
    assert isinstance(data.x, torch.Tensor), "Data must have x"

    print("Splits:")
    print("n_train:", data.train_mask.sum().item())  # type: ignore
    print("n_val:", data.val_mask.sum().item())  # type: ignore
    print("n_test:", data.test_mask.sum().item())  # type: ignore
    print("n_total:", data.y.shape[0])

    print(f"Number of nodes: {data.y.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Number of classes: {data.y.unique().shape[0]}")
    print(f"Number of features: {data.x.shape[1]}")

    print(f"Average node degree: {data.edge_index.shape[1] / data.y.shape[0]:.2f}")
    if show_all:
        # These are slower to compute
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")


def remove_duplicate_edges(edges: torch.Tensor) -> torch.Tensor:
    """
    Remove duplicate edges from the edge index tensor.

    Args:
        edges (torch.Tensor): The edge index tensor of shape (2, num_edges).

    Returns:
        torch.Tensor: A Tensor containing the unique edges.
    """
    return torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), dtype=torch.int32).coalesce().indices()


def remove_self_loops(edges: torch.Tensor) -> torch.Tensor:
    """
    Remove self-loops from the edge index tensor.

    Args:
        edges (torch.Tensor): The edge index tensor of shape (2, num_edges).

    Returns:
        torch.Tensor: The edge index tensor with self-loops removed.
    """

    sparse_coo = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), dtype=torch.int32).coalesce()
    self_loops_mask = sparse_coo.indices()[0] == sparse_coo.indices()[1]
    sparse_coo_without_self_loops = torch.sparse_coo_tensor(
        sparse_coo.indices()[:, ~self_loops_mask], sparse_coo.values()[~self_loops_mask], dtype=torch.int32
    ).coalesce()

    return sparse_coo_without_self_loops.indices()


def add_self_loops_and_remove_duplicate_edges(edges: torch.Tensor) -> torch.Tensor:
    """
    Add self-loops to the edge index tensor. It also removes duplicate edges if any.

    Args:
        edges (torch.Tensor): The edge index tensor of shape (2, num_edges).

    Returns:
        torch.Tensor: The edge index tensor with self-loops added.
    """
    sparse_coo = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), dtype=torch.int32).coalesce()
    size = sparse_coo.size()

    indices = torch.arange(size[0]).expand(2, -1)
    sparse_coo_with_self_loops = torch.sparse_coo_tensor(
        torch.cat([indices, sparse_coo.indices()], dim=1),
        torch.cat([torch.ones(size[0]), sparse_coo.values()], dim=0),
        dtype=torch.int32,
    ).coalesce()

    return sparse_coo_with_self_loops.indices()


def to_undirected(edges: torch.Tensor) -> torch.Tensor:
    """
    Convert the edge index tensor to an undirected graph.
    """
    edges = torch.cat([edges, edges.flip(0)], dim=1)

    return edges
