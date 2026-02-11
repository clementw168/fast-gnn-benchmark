import pathlib
from typing import Callable

import gdown
import scipy
import torch
from torch_geometric.data import Data


class PokecDataset:
    DRIVE_ID = "1575QYJwJlj7AWuOKMlwVmMz8FcslUncu"

    def __init__(
        self, root: str = "./datasets/pokec", transform: Callable | None = None, normalize_features: bool = True
    ):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.data = self.process(transform, normalize_features)

    def download(self):
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        raw_path = self.root / "raw" / "pokec.mat"
        if not raw_path.exists():
            gdown.download(id=self.DRIVE_ID, output=str(raw_path))
            print(f"Pokec dataset downloaded to {raw_path}")

        else:
            print("Pokec dataset already downloaded")

    def load_data(self):
        raw_data = scipy.io.loadmat(str(self.root / "raw" / "pokec.mat"))

        features = torch.tensor(raw_data["node_feat"], dtype=torch.float)

        labels = torch.tensor(raw_data["label"].flatten(), dtype=torch.long)

        # Follow 1:1:8 ratio for train:val:test from the paper
        random_split = torch.rand(labels.shape[0])

        train_mask = (random_split < 0.1) & (labels != -1)
        val_mask = (random_split >= 0.1) & (random_split < 0.2) & (labels != -1)
        test_mask = (random_split >= 0.2) & (labels != -1)

        return Data(
            x=features,
            edge_index=torch.tensor(raw_data["edge_index"], dtype=torch.long),
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def process(self, transform: Callable | None = None, normalize_features: bool = True):
        self.download()
        data = self.load_data()
        if transform:
            data = transform(data)
        if normalize_features:
            assert data.x is not None, "Data must have x"
            data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-10)  # type: ignore
        return data

    def __len__(self):
        return 1

    def __getitem__(self, index: int) -> Data:
        assert index == 0, "Index must be 0"
        return self.data
