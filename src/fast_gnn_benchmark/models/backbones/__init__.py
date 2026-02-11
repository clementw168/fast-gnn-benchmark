import torch

from fast_gnn_benchmark.models.backbones.gcn2 import GCNII
from fast_gnn_benchmark.models.backbones.gnn import load_gnn
from fast_gnn_benchmark.models.backbones.mlp import MLP, PMLP, MLPAdjacency
from fast_gnn_benchmark.models.backbones.polynormer import PolyNormer
from fast_gnn_benchmark.models.backbones.sgformer import SGFormer
from fast_gnn_benchmark.schemas.model import (
    ArchitectureParametersChoices,
    ArchitectureType,
    GCNIIParameters,
    GNNParameters,
    MLPAdjacencyParameters,
    MLPParameters,
    PMLPParameters,
    PolyNormerParameters,
    SGCParameters,
    SGFormerParameters,
)


def load_backbone(architecture_parameters: ArchitectureParametersChoices) -> torch.nn.Module:
    match architecture_parameters.architecture_type:
        case ArchitectureType.GCN | ArchitectureType.SAGE | ArchitectureType.GAT | ArchitectureType.SGC:
            assert isinstance(architecture_parameters, GNNParameters | SGCParameters)
            return load_gnn(architecture_parameters)

        case ArchitectureType.GCNII:
            assert isinstance(architecture_parameters, GCNIIParameters)
            return GCNII(architecture_parameters)

        case ArchitectureType.SGFORMER:
            assert isinstance(architecture_parameters, SGFormerParameters)
            return SGFormer(architecture_parameters)

        case ArchitectureType.MLP:
            assert isinstance(architecture_parameters, MLPParameters)
            return MLP(architecture_parameters)

        case ArchitectureType.MLP_ADJACENCY:
            assert isinstance(architecture_parameters, MLPAdjacencyParameters)
            return MLPAdjacency(architecture_parameters)

        case ArchitectureType.PMLP:
            assert isinstance(architecture_parameters, PMLPParameters)
            return PMLP(architecture_parameters)

        case ArchitectureType.POLYNORMER:
            assert isinstance(architecture_parameters, PolyNormerParameters)
            return PolyNormer(architecture_parameters)

        case _:
            raise ValueError(f"Invalid architecture type: {architecture_parameters.architecture_type}")
