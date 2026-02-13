import os

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage


class FixLinkPropPredDataset(PygLinkPropPredDataset):
    """Fix the serialization error of PygLinkPropPredDataset"""

    def __init__(self, name, root="dataset", transform=None, pre_transform=None, meta_dict=None):
        with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            super().__init__(name, root, transform, pre_transform, meta_dict)

    def get_edge_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = os.path.join(self.root, "split", split_type)  # type: ignore

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"), weights_only=False)

        train = replace_numpy_with_torchtensor(torch.load(os.path.join(path, "train.pt"), weights_only=False))
        valid = replace_numpy_with_torchtensor(torch.load(os.path.join(path, "valid.pt"), weights_only=False))
        test = replace_numpy_with_torchtensor(torch.load(os.path.join(path, "test.pt"), weights_only=False))

        return {"train": train, "valid": valid, "test": test}


if __name__ == "__main__":
    dataset = FixLinkPropPredDataset(name="ogbl-ppa")
    print(dataset.get_edge_split())
