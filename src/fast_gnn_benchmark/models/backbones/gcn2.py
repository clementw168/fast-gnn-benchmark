import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCN2Conv

from fast_gnn_benchmark.schemas.model import GCNIIParameters


class GCNII(torch.nn.Module):
    def __init__(self, architecture_parameters: GCNIIParameters):
        super().__init__()

        self.architecture_parameters = architecture_parameters

        self.input_projection = torch.nn.Linear(architecture_parameters.input_dim, architecture_parameters.hidden_dim)
        self.convs = torch.nn.ModuleList(
            GCN2Conv(
                architecture_parameters.hidden_dim,
                alpha=architecture_parameters.alpha,
                theta=architecture_parameters.theta,
                layer=layer_index + 1,
            )
            for layer_index in range(architecture_parameters.num_layers)
        )

        self.output_projection = torch.nn.Linear(architecture_parameters.hidden_dim, architecture_parameters.output_dim)

    def forward(self, x, edge_index):
        x_init = self.input_projection(x)
        x_init = F.relu(x_init)
        x_init = F.dropout(x_init, self.architecture_parameters.dropout, training=self.training)

        x = x_init

        for conv in self.convs:
            x = conv(x, x_init, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.architecture_parameters.dropout, training=self.training)

        x = self.output_projection(x)

        return x
