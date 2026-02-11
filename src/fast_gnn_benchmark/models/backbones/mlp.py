import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from fast_gnn_benchmark.schemas.model import MLPAdjacencyParameters, MLPParameters, PMLPParameters


class MLP(torch.nn.Module):
    def __init__(self, architecture_parameters: MLPParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.projection_layers = torch.nn.ModuleList()
        if architecture_parameters.use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        for layer_index in range(architecture_parameters.num_layers):
            if layer_index == 0:
                input_dim = architecture_parameters.input_dim
            else:
                input_dim = architecture_parameters.hidden_dim

            if layer_index == architecture_parameters.num_layers - 1:
                output_dim = architecture_parameters.output_dim
            else:
                output_dim = architecture_parameters.hidden_dim

            self.projection_layers.append(torch.nn.Linear(input_dim, output_dim))
            if architecture_parameters.use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(output_dim))

    def forward(self, x, edge_index):
        for layer_index in range(self.architecture_parameters.num_layers):
            projected_x = self.projection_layers[layer_index](x)

            if layer_index != self.architecture_parameters.num_layers - 1:
                if self.architecture_parameters.use_layer_norm:
                    projected_x = self.layer_norms[layer_index](projected_x)
                projected_x = F.relu(projected_x)
                projected_x = F.dropout(projected_x, p=self.architecture_parameters.dropout, training=self.training)

            if (
                self.architecture_parameters.use_residual
                and layer_index != 0
                and layer_index != self.architecture_parameters.num_layers - 1
            ):
                x = x + projected_x
            else:
                x = projected_x

        return x


class MLPAdjacency(torch.nn.Module):
    """MLP using adjacency matrix as features"""

    def __init__(self, architecture_parameters: MLPAdjacencyParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.projection_layers = torch.nn.ModuleList()
        if architecture_parameters.use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        for layer_index in range(architecture_parameters.num_layers):
            if layer_index == 0:
                input_dim = architecture_parameters.input_dim
            else:
                input_dim = architecture_parameters.hidden_dim

            if layer_index == architecture_parameters.num_layers - 1:
                output_dim = architecture_parameters.output_dim
            else:
                output_dim = architecture_parameters.hidden_dim

            self.projection_layers.append(torch.nn.Linear(input_dim, output_dim))
            if architecture_parameters.use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(output_dim))

    def forward(self, x, edge_index):
        adj = to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None).squeeze(0)
        x = torch.cat([x, adj], dim=-1)

        for layer_index in range(self.architecture_parameters.num_layers):
            projected_x = self.projection_layers[layer_index](x)

            if layer_index != self.architecture_parameters.num_layers - 1:
                if self.architecture_parameters.use_layer_norm:
                    projected_x = self.layer_norms[layer_index](projected_x)
                projected_x = F.relu(projected_x)
                projected_x = F.dropout(projected_x, p=self.architecture_parameters.dropout, training=self.training)

            if (
                self.architecture_parameters.use_residual
                and layer_index != 0
                and layer_index != self.architecture_parameters.num_layers - 1
            ):
                x = x + projected_x
            else:
                x = projected_x

        return x


class PropagationLayer(MessagePassing):
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):  # type: ignore
        return norm.view(-1, 1) * x_j


class PMLP(torch.nn.Module):
    def __init__(self, architecture_parameters: PMLPParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.projection_layers = torch.nn.ModuleList()
        self.propagation_layers = torch.nn.ModuleList()

        for layer_index in range(architecture_parameters.num_layers):
            if layer_index == 0:
                input_dim = architecture_parameters.input_dim
            else:
                input_dim = architecture_parameters.hidden_dim

            if layer_index == architecture_parameters.num_layers - 1:
                output_dim = architecture_parameters.output_dim
            else:
                output_dim = architecture_parameters.hidden_dim

            self.projection_layers.append(torch.nn.Linear(input_dim, output_dim))
            self.propagation_layers.append(PropagationLayer())

    def forward(self, x, edge_index):
        for layer_index in range(self.architecture_parameters.num_layers):
            x = x.matmul(self.projection_layers[layer_index].weight.T)  # type: ignore

            if not self.training:
                x = self.propagation_layers[layer_index](x, edge_index)

            x = x + self.projection_layers[layer_index].bias  # type: ignore

            if layer_index != self.architecture_parameters.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.architecture_parameters.dropout, training=self.training)

        return x
