import os
import shutil
from typing import Callable

import numpy as np
import pandas as pd
import torch
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import Data


def download_dataset_if_not_exists(root, name, original_root, download_link):
    paths = [
        os.path.join(original_root, "raw", "data.npz"),
        os.path.join(original_root, "raw", "node-label.npz"),
        os.path.join(original_root, "split", "time", "train.csv.gz"),
        os.path.join(original_root, "split", "time", "valid.csv.gz"),
        os.path.join(original_root, "split", "time", "test.csv.gz"),
    ]
    if all(os.path.exists(path) for path in paths):
        print(f"Dataset {name} already downloaded at {root}")
        return
    else:
        print(f"Dataset {name} not found at {root}, downloading...")
        path = download_url(download_link, original_root)
        extract_zip(path, original_root)
        os.unlink(path)

        extracted_folder = os.path.join(original_root, os.listdir(original_root)[0])

        for file in os.listdir(extracted_folder):
            shutil.move(os.path.join(extracted_folder, file), original_root)

        os.remove(extracted_folder)


class OGBNDatasetOnDisk:
    def __init__(self, name, root, transform: Callable | None = None):
        self.name = name
        self.root = root
        self.transform = transform

        self.set_metadata(name)

        self.original_root = os.path.join(root, "_".join(name.split("-")))

        download_dataset_if_not_exists(self.root, self.name, self.original_root, self.download_link)

    def set_metadata(self, name: str):
        match name:
            case "ogbn-papers100M":
                self.num_nodes = 111_059_956
                self.num_edges = 1_615_685_872
                self.dim_features = 128
                self.download_link = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
            case _:
                raise ValueError(f"Metadata not set for dataset {name}")

    def __len__(self):
        return 1


class OGBNDatasetOnRAM:
    def __init__(self, name, root, transform: Callable | None = None):
        self.name = name
        self.root = root
        self.transform = transform

        self.set_metadata(name)

        self.original_root = os.path.join(root, "_".join(name.split("-")))
        self.unzipped_folder = os.path.join(self.original_root, "decompressed")

        download_dataset_if_not_exists(self.root, self.name, self.original_root, self.download_link)

        self.process()

        self.data = self.load_all_data_in_ram()

    def set_metadata(self, name: str):
        match name:
            case "ogbn-papers100M":
                self.num_nodes = 111_059_956
                self.num_edges = 1_615_685_872
                self.dim_features = 128
                self.download_link = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
            case _:
                raise ValueError(f"Metadata not set for dataset {name}")

    def process(self) -> None:
        self.create_features()
        self.create_edges()
        self.create_labels()
        self.create_splits()

    def create_features(self):
        self.features_path = os.path.join(self.unzipped_folder, "features.npy")
        if os.path.exists(self.features_path):
            print("Features found")
            return

        print("Features not found, creating...")
        features = np.load(os.path.join(self.original_root, "raw", "data.npz"))["node_feat"]
        np.save(self.features_path, features)

    def create_edges(self):
        self.edges_path = os.path.join(self.unzipped_folder, "edges.npy")
        if os.path.exists(self.edges_path):
            print("Edges found")
            return

        print("Edges not found, creating...")
        edges = np.load(os.path.join(self.original_root, "raw", "data.npz"))["edge_index"]
        np.save(self.edges_path, edges)

    def create_labels(self):
        self.labels_path = os.path.join(self.unzipped_folder, "labels.npy")
        if os.path.exists(self.labels_path):
            print("Labels found")
            return

        print("Labels not found, creating...")
        labels = np.load(os.path.join(self.original_root, "raw", "node-label.npz"))["node_label"].flatten()
        labels[np.isnan(labels)] = -1
        labels = labels.astype(np.int64)
        np.save(self.labels_path, labels)

    def create_splits(self):
        self.train_mask_path = os.path.join(self.unzipped_folder, "train_mask.npy")
        self.val_mask_path = os.path.join(self.unzipped_folder, "val_mask.npy")
        self.test_mask_path = os.path.join(self.unzipped_folder, "test_mask.npy")
        if (
            os.path.exists(self.train_mask_path)
            and os.path.exists(self.val_mask_path)
            and os.path.exists(self.test_mask_path)
        ):
            print("Train, val and test masks found")
            return

        print("Split mask not found, creating...")

        split_folder = os.path.join(self.original_root, "split", "time")
        train_idx = pd.read_csv(os.path.join(split_folder, "train.csv.gz"), compression="gzip", header=None).values.T[0]
        valid_idx = pd.read_csv(os.path.join(split_folder, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
        test_idx = pd.read_csv(os.path.join(split_folder, "test.csv.gz"), compression="gzip", header=None).values.T[0]

        labels = np.load(self.labels_path)

        train_mask = np.zeros(labels.shape[0], dtype=np.bool)
        train_mask[train_idx] = True
        train_mask[np.isnan(labels)] = False

        val_mask = np.zeros(labels.shape[0], dtype=np.bool)
        val_mask[valid_idx] = True
        val_mask[np.isnan(labels)] = False

        test_mask = np.zeros(labels.shape[0], dtype=np.bool)
        test_mask[test_idx] = True
        test_mask[np.isnan(labels)] = False

        np.save(self.train_mask_path, train_mask)
        np.save(self.val_mask_path, val_mask)
        np.save(self.test_mask_path, test_mask)

    def load_all_data_in_ram(self) -> Data:
        print("Loading features...")
        features = np.load(self.features_path)
        print("Loading_edges...")
        edges = np.load(self.edges_path)
        print("Loading labels")
        labels = np.load(self.labels_path)
        print("Loading splits")
        train_mask = np.load(self.train_mask_path)
        val_mask = np.load(self.val_mask_path)
        test_mask = np.load(self.test_mask_path)

        return Data(
            x=torch.from_numpy(features),
            edge_index=torch.from_numpy(edges),
            y=torch.from_numpy(labels),
            train_mask=torch.from_numpy(train_mask),
            val_mask=torch.from_numpy(val_mask),
            test_mask=torch.from_numpy(test_mask),
        )

    def __getitem__(self, index: int) -> Data:
        assert index == 0, "Index must be 0"
        return self.data
