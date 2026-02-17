from collections.abc import Callable
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from torch_geometric.transforms import NormalizeFeatures, ToUndirected


class TransformType(Enum):
    NORMALIZE_FEATURES = "normalize_features"
    TO_UNDIRECTED = "to_undirected"

    def get(self) -> Callable:
        match self:
            case TransformType.NORMALIZE_FEATURES:
                return NormalizeFeatures()
            case TransformType.TO_UNDIRECTED:
                return ToUndirected()
            case _:
                raise ValueError(f"Invalid transform type: {self}")


class SplitType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetType(Enum):
    # Node classification
    CORA = "cora"
    CITESEER = "citeseer"
    PUBMED = "pubmed"
    AMAZON_COMPUTER = "amazon-computer"
    AMAZON_PHOTO = "amazon-photo"
    CO_AUTHOR_CS = "co-author-cs"
    CO_AUTHOR_PHYSICS = "co-author-physics"
    OGBN_PRODUCTS = "ogbn-products"
    OGBN_ARXIV = "ogbn-arxiv"
    OGBN_PAPERS100M = "ogbn-papers100M"
    OGBN_PAPERS100M_ON_DISK = "ogbn-papers100M-on-disk"
    OGBN_PAPERS100M_ON_RAM = "ogbn-papers100M-on-ram"
    POKEC = "pokec"

    # Link prediction
    OGBL_PPA = "ogbl-ppa"
    OGBL_COLLAB = "ogbl-collab"
    OGBL_DDI = "ogbl-ddi"
    OGBL_CITATION2 = "ogbl-citation2"
    OGBL_WIKIKG2 = "ogbl-wikikg2"
    OGBL_BIOKG = "ogbl-biokg"
    OGBL_VESSEL = "ogbl-vessel"


class DataLoaderType(Enum):
    BASE_DATA_LOADER = "base_data_loader"
    RANDOM_NODE_LOADER = "random_node_loader"
    OPTIMIZED_RANDOM_NODE_LOADER = "optimized_random_node_loader"
    RANDOM_NODE_LOADER_WITH_REPLACEMENT = "random_node_loader_with_replacement"
    ON_DISK_MEMMAPS_RANDOM_NODE_LOADER = "on_disk_memmaps_random_node_loader"
    NEIGHBOR_LOADER = "neighbor_loader"
    RANDOM_WALK_LOADER = "random_walk_loader"
    CLUSTER_LOADER = "cluster_loader"
    PPR_NODE_LOADER = "ppr_node_loader"
    DROP_EDGE_LOADER = "drop_edge_loader"
    LINK_NEIGHBOR_LOADER = "link_neighbor_loader"


class DataLoaderParameters(BaseModel):
    pass


class BaseDataLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.BASE_DATA_LOADER] = DataLoaderType.BASE_DATA_LOADER
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


class RandomNodeLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.RANDOM_NODE_LOADER] = DataLoaderType.RANDOM_NODE_LOADER
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = True
    num_parts: int


class RandomNodeLoaderWithReplacementParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.RANDOM_NODE_LOADER_WITH_REPLACEMENT] = (
        DataLoaderType.RANDOM_NODE_LOADER_WITH_REPLACEMENT
    )
    proportion: float
    on_device: bool = True
    pin_memory: bool = False


class OnDiskMemmapsRandomNodeLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.ON_DISK_MEMMAPS_RANDOM_NODE_LOADER] = (
        DataLoaderType.ON_DISK_MEMMAPS_RANDOM_NODE_LOADER
    )
    num_parts: int
    use_blocks: bool = True
    block_size: int = 1_000_000
    num_workers: int = -1


class OptimizedRandomNodeLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.OPTIMIZED_RANDOM_NODE_LOADER] = DataLoaderType.OPTIMIZED_RANDOM_NODE_LOADER
    pin_memory: bool = False
    on_device: bool = True
    num_parts: int


class NeighborLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.NEIGHBOR_LOADER] = DataLoaderType.NEIGHBOR_LOADER
    num_neighbors: list[int]
    batch_size: int
    num_workers: int = 0
    shuffle: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    on_device: bool = True


class RandomWalkLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.RANDOM_WALK_LOADER] = DataLoaderType.RANDOM_WALK_LOADER
    walk_length: int
    num_seeds: int
    compute_normalization_stats: bool = False
    num_stats_samples: int = 1000


class ClusterLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.CLUSTER_LOADER] = DataLoaderType.CLUSTER_LOADER
    num_clusters: int
    num_parts: int
    shuffle: bool = True


class PPRNodeLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.PPR_NODE_LOADER] = DataLoaderType.PPR_NODE_LOADER
    num_parts: int
    node_budget: int
    alpha: float = 0.85
    ppr_iterations: int = 4
    on_device: bool = True
    pin_memory: bool = False


class LinkNeighborLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.LINK_NEIGHBOR_LOADER] = DataLoaderType.LINK_NEIGHBOR_LOADER


class DropEdgeLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.DROP_EDGE_LOADER] = DataLoaderType.DROP_EDGE_LOADER
    drop_edge_ratio: float
    on_device: bool = True
    pin_memory: bool = False


DataLoaderParametersChoices = Annotated[
    BaseDataLoaderParameters
    | RandomNodeLoaderParameters
    | OptimizedRandomNodeLoaderParameters
    | OnDiskMemmapsRandomNodeLoaderParameters
    | NeighborLoaderParameters
    | RandomWalkLoaderParameters
    | ClusterLoaderParameters
    | PPRNodeLoaderParameters
    | LinkNeighborLoaderParameters
    | RandomNodeLoaderWithReplacementParameters
    | DropEdgeLoaderParameters,
    Field(discriminator="data_loader_type"),
]
