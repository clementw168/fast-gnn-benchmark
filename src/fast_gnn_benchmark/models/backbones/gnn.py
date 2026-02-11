import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv

from fast_gnn_benchmark.schemas.model import (
    ArchitectureParametersChoices,
    ArchitectureType,
    GNNParameters,
    SGCParameters,
)


class GNNStack(torch.nn.Module):
    def __init__(self, architecture_parameters: GNNParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        if architecture_parameters.use_input_projection:
            self.input_projection = torch.nn.Linear(
                architecture_parameters.input_dim, architecture_parameters.hidden_dim
            )

        layer_class = self.get_layer_type(ArchitectureType(architecture_parameters.architecture_type))

        self.conv_layers = torch.nn.ModuleList()

        if architecture_parameters.use_residual:
            self.residual_layers = torch.nn.ModuleList()

        assert not (architecture_parameters.use_batch_norm and architecture_parameters.use_layer_norm), (
            "Cannot use both batch norm and layer norm"
        )
        if architecture_parameters.use_batch_norm:
            self.batch_norms = torch.nn.ModuleList()
        if architecture_parameters.use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        for layer_index in range(architecture_parameters.num_layers):
            if layer_index == 0 and (not architecture_parameters.use_input_projection):
                input_dim = architecture_parameters.input_dim
            else:
                input_dim = architecture_parameters.hidden_dim

            if layer_index == architecture_parameters.num_layers - 1 and (
                not architecture_parameters.use_output_projection
            ):
                output_dim = architecture_parameters.output_dim
            else:
                output_dim = architecture_parameters.hidden_dim

            self.conv_layers.append(layer_class(input_dim, output_dim, **architecture_parameters.conv_parameters))

            if architecture_parameters.use_residual:
                self.residual_layers.append(torch.nn.Linear(input_dim, output_dim))

            if architecture_parameters.use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(output_dim))

            if architecture_parameters.use_batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))

        if architecture_parameters.use_output_projection:
            self.output_projection = torch.nn.Linear(
                architecture_parameters.hidden_dim, architecture_parameters.output_dim
            )

    def get_layer_type(self, architecture_type: ArchitectureType) -> type[torch.nn.Module]:
        match architecture_type:
            case ArchitectureType.GCN:
                return GCNConv
            case ArchitectureType.SAGE:
                return SAGEConv

            case _:
                raise ValueError(f"Invalid architecture type: {architecture_type}")

    def forward(self, x, edge_index):
        if self.architecture_parameters.use_input_projection:
            x = self.input_projection(x)
            x = F.dropout(x, p=self.architecture_parameters.dropout, training=self.training)

        for layer_index in range(len(self.conv_layers)):
            if self.architecture_parameters.use_residual:
                x = self.conv_layers[layer_index](x, edge_index) + self.residual_layers[layer_index](x)
            else:
                x = self.conv_layers[layer_index](x, edge_index)

            if self.architecture_parameters.use_layer_norm:
                x = self.layer_norms[layer_index](x)

            if self.architecture_parameters.use_batch_norm:
                x = self.batch_norms[layer_index](x)

            if (self.architecture_parameters.use_output_projection) or (layer_index != len(self.conv_layers) - 1):
                x = F.relu(x)
                x = F.dropout(x, p=self.architecture_parameters.dropout, training=self.training)

        if self.architecture_parameters.use_output_projection:
            x = self.output_projection(x)

        return x


class SGC(torch.nn.Module):
    def __init__(self, architecture_parameters: SGCParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.conv_layers = SGConv(
            in_channels=architecture_parameters.input_dim,
            out_channels=architecture_parameters.output_dim,
            K=architecture_parameters.num_layers,
        )

    def forward(self, x, edge_index):
        x = self.conv_layers(x, edge_index)

        return x


class GAT(torch.nn.Module):
    """
    Separate class because multihead is handled differently
    """

    def __init__(self, architecture_parameters: GNNParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters
        self.heads = architecture_parameters.conv_parameters.get("heads", 1)

        if architecture_parameters.use_input_projection:
            self.input_projection = torch.nn.Linear(
                architecture_parameters.input_dim, architecture_parameters.hidden_dim
            )

        self.conv_layers = torch.nn.ModuleList()

        if architecture_parameters.use_residual:
            self.residual_layers = torch.nn.ModuleList()

        assert not (architecture_parameters.use_batch_norm and architecture_parameters.use_layer_norm), (
            "Cannot use both batch norm and layer norm"
        )
        if architecture_parameters.use_batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        if architecture_parameters.use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        for layer_index in range(architecture_parameters.num_layers):
            if layer_index == 0 and (not architecture_parameters.use_input_projection):
                input_dim = architecture_parameters.input_dim
            else:
                input_dim = architecture_parameters.hidden_dim

            if layer_index == architecture_parameters.num_layers - 1 and (
                not architecture_parameters.use_output_projection
            ):
                output_dim = architecture_parameters.output_dim
                gat_output_dim = output_dim
                layer_heads = 1

            else:
                output_dim = architecture_parameters.hidden_dim
                gat_output_dim = output_dim // self.heads
                layer_heads = self.heads

            self.conv_layers.append(GATConv(input_dim, gat_output_dim, heads=layer_heads))

            if architecture_parameters.use_residual:
                self.residual_layers.append(torch.nn.Linear(input_dim, output_dim))

            if architecture_parameters.use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(output_dim))

            if architecture_parameters.use_batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))

        if architecture_parameters.use_output_projection:
            self.output_projection = torch.nn.Linear(
                architecture_parameters.hidden_dim, architecture_parameters.output_dim
            )

    def forward(self, x, edge_index):
        if self.architecture_parameters.use_input_projection:
            x = self.input_projection(x)
            x = F.dropout(x, p=self.architecture_parameters.dropout, training=self.training)

        for layer_index in range(len(self.conv_layers)):
            if self.architecture_parameters.use_residual:
                x = self.conv_layers[layer_index](x, edge_index) + self.residual_layers[layer_index](x)
            else:
                x = self.conv_layers[layer_index](x, edge_index)

            if self.architecture_parameters.use_layer_norm:
                x = self.layer_norms[layer_index](x)

            if self.architecture_parameters.use_batch_norm:
                x = self.batch_norms[layer_index](x)

            if (self.architecture_parameters.use_output_projection) or (layer_index != len(self.conv_layers) - 1):
                x = F.relu(x)
                x = F.dropout(x, p=self.architecture_parameters.dropout, training=self.training)

        if self.architecture_parameters.use_output_projection:
            x = self.output_projection(x)

        return x


def load_gnn(architecture_parameters: ArchitectureParametersChoices) -> torch.nn.Module:
    match architecture_parameters.architecture_type:
        case ArchitectureType.GCN | ArchitectureType.SAGE:
            return GNNStack(architecture_parameters)
        case ArchitectureType.GAT:
            return GAT(architecture_parameters)
        case ArchitectureType.SGC:
            return SGC(architecture_parameters)
        case _:
            raise ValueError(f"Invalid architecture type: {architecture_parameters.architecture_type}")
