from pydantic import BaseModel, Field, field_validator
from torch_geometric.data import Dataset
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import Compose

from fast_gnn_benchmark.data.dataloaders import (
    BaseDataLoader,
    ClusterLoaderWrapper,
    DropEdgeLoader,
    NeighborLoaderWrapper,
    OnDiskMemmapsRandomNodeLoader,
    OptimizedRandomNodeLoader,
    PPRNodeLoader,
    RandomNodeLoaderWithReplacement,
    RandomNodeLoaderWrapper,
    RandomWalkLoaderWrapper,
)
from fast_gnn_benchmark.data.dataset.ogbl import FixLinkPropPredDataset
from fast_gnn_benchmark.data.dataset.ogbn import OGBNDataset
from fast_gnn_benchmark.data.dataset.ogbn_on_disk import OGBNDatasetOnDisk, OGBNDatasetOnRAM
from fast_gnn_benchmark.data.dataset.pokec import PokecDataset
from fast_gnn_benchmark.data.dataset.split_strategies import random_split_dataset, resplit_planetoid_dataset
from fast_gnn_benchmark.data.utils import (
    add_self_loops_and_remove_duplicate_edges,
    print_data_properties,
    remove_duplicate_edges,
    remove_self_loops,
    to_undirected,
)
from fast_gnn_benchmark.schemas.dataset_models import (
    DataLoaderParametersChoices,
    DataLoaderType,
    DatasetType,
    SplitType,
    TransformType,
)

DatasetTypeChoices = (
    Dataset
    | OGBNDataset
    | PokecDataset
    | OGBNDatasetOnDisk
    | OGBNDatasetOnRAM
    | Amazon
    | Coauthor
    | FixLinkPropPredDataset
)

DataLoaderTypeChoices = (
    BaseDataLoader
    | RandomNodeLoaderWrapper
    | OptimizedRandomNodeLoader
    | OnDiskMemmapsRandomNodeLoader
    | NeighborLoaderWrapper
    | RandomWalkLoaderWrapper
    | ClusterLoaderWrapper
    | PPRNodeLoader
    | LinkNeighborLoader
    | RandomNodeLoaderWithReplacement
    | DropEdgeLoader
)


class DataParameters(BaseModel):
    dataset_type: DatasetType
    transforms: list[TransformType] = Field(default_factory=list[TransformType])
    to_undirected: bool = False
    add_self_loops_and_remove_duplicate_edges: bool = False
    remove_duplicate_edges: bool = False
    remove_self_loops: bool = False

    train_data_loader_parameters: DataLoaderParametersChoices
    val_data_loader_parameters: DataLoaderParametersChoices
    test_data_loader_parameters: DataLoaderParametersChoices

    @field_validator("train_data_loader_parameters", mode="before")
    @classmethod
    def convert_train_data_loader_type(cls, v):
        if isinstance(v, dict) and "data_loader_type" in v:
            data_loader_type = v["data_loader_type"]
            if isinstance(data_loader_type, str):
                try:
                    v = v.copy()  # Don't modify the original
                    v["data_loader_type"] = DataLoaderType(data_loader_type)
                except ValueError:
                    raise ValueError(
                        f"Invalid data_loader_type: {data_loader_type}. Must be one of: {[e.value for e in DataLoaderType]}"
                    ) from None
        return v

    @field_validator("val_data_loader_parameters", mode="before")
    @classmethod
    def convert_val_data_loader_type(cls, v):
        if isinstance(v, dict) and "data_loader_type" in v:
            data_loader_type = v["data_loader_type"]
            if isinstance(data_loader_type, str):
                try:
                    v = v.copy()  # Don't modify the original
                    v["data_loader_type"] = DataLoaderType(data_loader_type)
                except ValueError:
                    raise ValueError(
                        f"Invalid data_loader_type: {data_loader_type}. Must be one of: {[e.value for e in DataLoaderType]}"
                    ) from None
        return v

    @field_validator("test_data_loader_parameters", mode="before")
    @classmethod
    def convert_test_data_loader_type(cls, v):
        if isinstance(v, dict) and "data_loader_type" in v:
            data_loader_type = v["data_loader_type"]
            if isinstance(data_loader_type, str):
                try:
                    v = v.copy()  # Don't modify the original
                    v["data_loader_type"] = DataLoaderType(data_loader_type)
                except ValueError:
                    raise ValueError(
                        f"Invalid data_loader_type: {data_loader_type}. Must be one of: {[e.value for e in DataLoaderType]}"
                    ) from None
        return v

    def get_dataset(self) -> DatasetTypeChoices:
        transforms = Compose([transform.get() for transform in self.transforms])

        match self.dataset_type:
            case DatasetType.CORA:
                dataset = Planetoid(root="./datasets/Planetoid", name="Cora", transform=transforms)
                dataset = resplit_planetoid_dataset(dataset)

            case DatasetType.CITESEER:
                dataset = Planetoid(root="./datasets/Planetoid", name="CiteSeer", transform=transforms)
                dataset = resplit_planetoid_dataset(dataset)

            case DatasetType.PUBMED:
                dataset = Planetoid(root="./datasets/Planetoid", name="PubMed", transform=transforms)
                dataset = resplit_planetoid_dataset(dataset)

            case DatasetType.AMAZON_COMPUTER:
                dataset = Amazon(root="./datasets/Amazon", name="Computers", transform=transforms)
                dataset = random_split_dataset(dataset)

            case DatasetType.CO_AUTHOR_CS:
                dataset = Coauthor(root="./datasets/Coauthor", name="CS", transform=transforms)
                dataset = random_split_dataset(dataset)

            case DatasetType.CO_AUTHOR_PHYSICS:
                dataset = Coauthor(root="./datasets/Coauthor", name="Physics", transform=transforms)
                dataset = random_split_dataset(dataset)

            case DatasetType.AMAZON_PHOTO:
                dataset = Amazon(root="./datasets/Amazon", name="Photo", transform=transforms)
                dataset = random_split_dataset(dataset)

            case DatasetType.OGBN_PRODUCTS:
                dataset = OGBNDataset(root="./datasets/ogb/", name="ogbn-products", transform=transforms)

            case DatasetType.OGBN_ARXIV:
                dataset = OGBNDataset(root="./datasets/ogb/", name="ogbn-arxiv", transform=transforms)

            case DatasetType.OGBN_PAPERS100M:
                dataset = OGBNDataset(root="./datasets/ogb/", name="ogbn-papers100M", transform=transforms)

            case DatasetType.OGBN_PAPERS100M_ON_DISK:
                dataset = OGBNDatasetOnDisk(root="./datasets/ogb/", name="ogbn-papers100M", transform=transforms)

            case DatasetType.OGBN_PAPERS100M_ON_RAM:
                dataset = OGBNDatasetOnRAM(root="./datasets/ogb/", name="ogbn-papers100M", transform=transforms)

            case DatasetType.POKEC:
                dataset = PokecDataset(root="./datasets/pokec", transform=transforms)

            case DatasetType.OGBL_PPA:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-ppa", transform=transforms)

            case DatasetType.OGBL_COLLAB:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-collab", transform=transforms)

            case DatasetType.OGBL_DDI:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-ddi", transform=transforms)

            case DatasetType.OGBL_CITATION2:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-citation2", transform=transforms)

            case DatasetType.OGBL_WIKIKG2:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-wikikg2", transform=transforms)

            case DatasetType.OGBL_BIOKG:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-biokg", transform=transforms)

            case DatasetType.OGBL_VESSEL:
                dataset = FixLinkPropPredDataset(root="./datasets/ogbl/", name="ogbl-vessel", transform=transforms)

            case _:
                raise ValueError(f"Invalid dataset type: {self}")

        if self.dataset_type == DatasetType.OGBN_PAPERS100M_ON_DISK:
            print("Skip processing for On Disk Dataset")
            return dataset

        assert not (
            self.add_self_loops_and_remove_duplicate_edges and self.remove_duplicate_edges
        ), "Cannot add self-loops and remove duplicate edges at the same time"

        if self.to_undirected:
            print(f"Converting to undirected graph for {self.dataset_type}")
            dataset[0].edge_index = to_undirected(dataset[0].edge_index)  # type: ignore

        if self.add_self_loops_and_remove_duplicate_edges:
            print(f"Adding self-loops and removing duplicate edges for {self.dataset_type}")
            dataset[0].edge_index = add_self_loops_and_remove_duplicate_edges(dataset[0].edge_index)  # type: ignore

        if self.remove_duplicate_edges:
            print(f"Removing duplicate edges for {self.dataset_type}")
            dataset[0].edge_index = remove_duplicate_edges(dataset[0].edge_index)  # type: ignore

        if self.remove_self_loops:
            print(f"Removing self-loops for {self.dataset_type}")
            dataset[0].edge_index = remove_self_loops(dataset[0].edge_index)  # type: ignore

        print_data_properties(dataset[0])  # type: ignore

        return dataset

    def get_data_loader(  # noqa: PLR0911
        self, dataset: DatasetTypeChoices, split_type: SplitType, data_loader_parameters: DataLoaderParametersChoices
    ) -> DataLoaderTypeChoices:
        match data_loader_parameters.data_loader_type:
            case DataLoaderType.BASE_DATA_LOADER:
                return BaseDataLoader(
                    dataset,
                    split_type=split_type,
                    num_workers=data_loader_parameters.num_workers,
                    pin_memory=data_loader_parameters.pin_memory,
                    persistent_workers=data_loader_parameters.persistent_workers,
                )
            case DataLoaderType.RANDOM_NODE_LOADER_WITH_REPLACEMENT:
                return RandomNodeLoaderWithReplacement(
                    dataset,
                    proportion=data_loader_parameters.proportion,
                    on_device=data_loader_parameters.on_device,
                    pin_memory=data_loader_parameters.pin_memory,
                    split_type=split_type,
                )

            case DataLoaderType.RANDOM_NODE_LOADER:
                return RandomNodeLoaderWrapper(
                    dataset,
                    num_workers=data_loader_parameters.num_workers,
                    num_parts=data_loader_parameters.num_parts,
                    shuffle=data_loader_parameters.shuffle,
                    pin_memory=data_loader_parameters.pin_memory,
                    persistent_workers=data_loader_parameters.persistent_workers,
                    split_type=split_type,
                )

            case DataLoaderType.OPTIMIZED_RANDOM_NODE_LOADER:
                return OptimizedRandomNodeLoader(
                    dataset,
                    data_loader_parameters.num_parts,
                    data_loader_parameters.on_device,
                    data_loader_parameters.pin_memory,
                    split_type,
                )

            case DataLoaderType.ON_DISK_MEMMAPS_RANDOM_NODE_LOADER:
                return OnDiskMemmapsRandomNodeLoader(
                    dataset,
                    data_loader_parameters.num_parts,
                    data_loader_parameters.use_blocks,
                    data_loader_parameters.block_size,
                    data_loader_parameters.num_workers,
                    split_type,
                )

            case DataLoaderType.NEIGHBOR_LOADER:
                return NeighborLoaderWrapper(
                    dataset,
                    num_neighbors=data_loader_parameters.num_neighbors,
                    batch_size=data_loader_parameters.batch_size,
                    shuffle=data_loader_parameters.shuffle,
                    num_workers=data_loader_parameters.num_workers,
                    pin_memory=data_loader_parameters.pin_memory,
                    persistent_workers=data_loader_parameters.persistent_workers,
                    split_type=split_type,
                    on_device=data_loader_parameters.on_device,
                )

            case DataLoaderType.RANDOM_WALK_LOADER:
                return RandomWalkLoaderWrapper(
                    dataset,
                    walk_length=data_loader_parameters.walk_length,
                    num_seeds=data_loader_parameters.num_seeds,
                    compute_normalization_stats=data_loader_parameters.compute_normalization_stats,
                    num_stats_samples=data_loader_parameters.num_stats_samples,
                    split_type=split_type,
                )

            case DataLoaderType.CLUSTER_LOADER:
                return ClusterLoaderWrapper(
                    dataset,
                    num_clusters=data_loader_parameters.num_clusters,
                    num_parts=data_loader_parameters.num_parts,
                    shuffle=data_loader_parameters.shuffle,
                    split_type=split_type,
                )

            case DataLoaderType.PPR_NODE_LOADER:
                return PPRNodeLoader(
                    dataset,
                    num_parts=data_loader_parameters.num_parts,
                    node_budget=data_loader_parameters.node_budget,
                    alpha=data_loader_parameters.alpha,
                    ppr_iterations=data_loader_parameters.ppr_iterations,
                    on_device=data_loader_parameters.on_device,
                    pin_memory=data_loader_parameters.pin_memory,
                    split_type=split_type,
                )

            case DataLoaderType.DROP_EDGE_LOADER:
                return DropEdgeLoader(
                    dataset,
                    drop_edge_ratio=data_loader_parameters.drop_edge_ratio,
                    on_device=data_loader_parameters.on_device,
                    pin_memory=data_loader_parameters.pin_memory,
                    split_type=split_type,
                )
            case DataLoaderType.LINK_NEIGHBOR_LOADER:
                raise ValueError("Link neighbor loader is not supported yet")

            case _:
                raise ValueError(f"Invalid data loader type: {data_loader_parameters.data_loader_type}")

    def get(
        self,
    ) -> tuple[DataLoaderTypeChoices, DataLoaderTypeChoices, DataLoaderTypeChoices]:
        dataset = self.get_dataset()

        train_data_loader = self.get_data_loader(dataset, SplitType.TRAIN, self.train_data_loader_parameters)
        val_data_loader = self.get_data_loader(dataset, SplitType.VAL, self.val_data_loader_parameters)
        test_data_loader = self.get_data_loader(dataset, SplitType.TEST, self.test_data_loader_parameters)

        return train_data_loader, val_data_loader, test_data_loader
