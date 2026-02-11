from typing import Any

import torch
from torch_geometric.datasets import Planetoid


def resplit_planetoid_dataset(dataset: Planetoid) -> Planetoid:
    """
    We reimplement the classic splits used for Planetoid from Semi-Supervised Classification
    with Graph Convolutional Networks:
    - For each class, we randomly select 20 nodes to create ``train_mask``.
    - We randomly select 500 nodes from the remaining nodes to create ``val_mask``.
    - We randomly select 1000 nodes from the remaining nodes to create ``test_mask``.
    - The remaining nodes are not used.
    """
    print("Re-splitting the Planetoid dataset")

    y: torch.Tensor = dataset.y  # type: ignore

    train_per_class = 20  # 20, 350
    number_val = 500  # 500, 1000
    number_test = 1000

    indices = torch.arange(y.shape[0])

    train_mask = torch.zeros(y.shape[0], dtype=torch.bool)

    for class_ in torch.unique(y):
        class_mask = y == class_
        class_indices = indices[class_mask]
        class_indices = class_indices[torch.randperm(len(class_indices))]
        train_mask[class_indices[:train_per_class]] = True

    non_train_indices = indices[~train_mask]
    non_train_indices = non_train_indices[torch.randperm(len(non_train_indices))]

    val_mask = torch.zeros(y.shape[0], dtype=torch.bool)
    val_mask[non_train_indices[:number_val]] = True
    test_mask = torch.zeros(y.shape[0], dtype=torch.bool)
    test_mask[non_train_indices[number_val : number_val + number_test]] = True

    dataset.data.train_mask = train_mask  # type: ignore
    dataset.data.val_mask = val_mask  # type: ignore
    dataset.data.test_mask = test_mask  # type: ignore

    return dataset


def random_split_dataset(dataset: Any, train_per_class: int = 20, val_per_class: int = 30) -> Any:
    """
    We reimplement the split strategy described in Pitfalls of Graph Neural Network Evaluation:
    - For each class, we randomly select ``train_per_class`` nodes to create ``train_mask``.
    - We randomly select ``val_per_class`` nodes from the remaining nodes to create ``val_mask``.
    - The remaining nodes are used as test nodes.
    """

    y: torch.Tensor = dataset.y  # type: ignore

    indices = torch.arange(y.shape[0])

    train_mask = torch.zeros(y.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(y.shape[0], dtype=torch.bool)

    for class_ in torch.unique(y):
        class_mask = y == class_
        class_indices = indices[class_mask]
        class_indices = class_indices[torch.randperm(len(class_indices))]
        train_mask[class_indices[:train_per_class]] = True
        val_mask[class_indices[train_per_class : train_per_class + val_per_class]] = True

    test_mask = (~train_mask) & (~val_mask)

    dataset.data.train_mask = train_mask  # type: ignore
    dataset.data.val_mask = val_mask  # type: ignore
    dataset.data.test_mask = test_mask  # type: ignore

    return dataset
