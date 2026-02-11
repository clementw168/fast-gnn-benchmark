from pathlib import Path
from typing import Callable

import numpy as np
import torch
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data


class FixNodePropPredDataset(NodePropPredDataset):
    """Fix the serialization error of NodePropPredDataset"""

    def pre_process(self):
        processed_dir = Path(self.root) / "processed"
        pre_processed_file_path = processed_dir / "data_processed"

        if pre_processed_file_path.exists():
            loaded_dict = torch.load(pre_processed_file_path, weights_only=False)
            self.graph, self.labels = loaded_dict["graph"], loaded_dict["labels"]

        else:
            super().pre_process()


class OGBNDataset:
    def __init__(self, name, root, transform: Callable | None = None):
        self.name = name
        self.root = root
        self.transform = transform
        self.data = self.process(transform)

    def process(self, transform: Callable | None = None):
        dataset = FixNodePropPredDataset(name=self.name, root=self.root)

        graph, labels = dataset[0]
        splits: dict[str, np.ndarray] = dataset.get_idx_split()  # type: ignore
        train_mask = np.zeros(graph["num_nodes"], dtype=np.bool)
        val_mask = np.zeros_like(train_mask, dtype=np.bool)
        test_mask = np.zeros_like(train_mask, dtype=np.bool)

        train_mask[splits["train"]] = True
        val_mask[splits["valid"]] = True
        test_mask[splits["test"]] = True

        x = torch.from_numpy(graph["node_feat"])
        if transform:
            x = transform(x)

        data = Data(
            x=x,
            edge_index=torch.from_numpy(graph["edge_index"]),
            y=torch.from_numpy(labels).squeeze(),
            train_mask=torch.from_numpy(train_mask),
            val_mask=torch.from_numpy(val_mask),
            test_mask=torch.from_numpy(test_mask),
        )

        return data

    def __len__(self):
        return 1

    def __getitem__(self, index: int) -> Data:
        assert index == 0, "Index must be 0"
        return self.data
