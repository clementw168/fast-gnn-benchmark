import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from fast_gnn_benchmark.schemas.model import PolyNormerParameters


class PolyNormerLocalStack(torch.nn.Module):
    def __init__(
        self,
        beta: float,
        local_layers: int,
        hidden_dim: int,
        num_heads: int,
        pre_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.local_layers = local_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.dropout = dropout

        if self.beta < 0:
            self.betas_weights = torch.nn.Parameter(torch.zeros(local_layers, num_heads * hidden_dim))
        else:
            self.betas_weights = torch.nn.Parameter(torch.ones(local_layers, num_heads * hidden_dim) * self.beta)

        if pre_norm:
            self.pre_norm_layers = torch.nn.ModuleList(
                [torch.nn.LayerNorm(hidden_dim * num_heads) for _ in range(local_layers)]
            )

        self.h_projections = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads) for _ in range(local_layers)]
        )

        self.local_convs = torch.nn.ModuleList(
            [
                GATConv(
                    hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, add_self_loops=False, bias=False
                )
                for _ in range(local_layers)
            ]
        )
        self.residual_projections = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads) for _ in range(local_layers)]
        )
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_dim * num_heads) for _ in range(local_layers)]
        )

    def forward(self, x, edge_index):
        x_out = torch.zeros_like(x)

        for layer_index in range(self.local_layers):
            if self.pre_norm:
                x = self.pre_norm_layers[layer_index](x)

            h = F.relu(self.h_projections[layer_index](x))

            av = self.local_convs[layer_index](x, edge_index) + self.residual_projections[layer_index](x)
            av = F.relu(av)
            av = F.dropout(av, p=self.dropout, training=self.training)

            beta = self.betas_weights[layer_index]
            if self.beta < 0:
                beta = torch.sigmoid(beta)
            beta = beta.unsqueeze(0)

            x = (1 - beta) * self.layer_norms[layer_index](av * h) + beta * av

            x_out = x_out + x

        return x_out


class PolyNormerGlobalStack(torch.nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int, qk_shared: bool, beta: float, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qk_shared = qk_shared
        self.beta = beta
        self.dropout = dropout

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, num_heads * hidden_dim))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, num_heads * hidden_dim) * self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(num_heads * hidden_dim, num_heads * hidden_dim))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(num_heads * hidden_dim, num_heads * hidden_dim))
            self.k_lins.append(torch.nn.Linear(num_heads * hidden_dim, num_heads * hidden_dim))
            self.v_lins.append(torch.nn.Linear(num_heads * hidden_dim, num_heads * hidden_dim))
            self.lns.append(torch.nn.LayerNorm(num_heads * hidden_dim))
        self.lin_out = torch.nn.Linear(num_heads * hidden_dim, num_heads * hidden_dim)

    def forward(self, x, edge_index):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_dim, self.num_heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_dim, self.num_heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_dim, self.num_heads)

            # numerator
            kv = torch.einsum("ndh, nmh -> dmh", k, v)
            num = torch.einsum("ndh, dmh -> nmh", q, kv)

            # denominator
            k_sum = torch.einsum("ndh -> dh", k)
            den = torch.einsum("ndh, dh -> nh", q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num / den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h + beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class PolyNormer(torch.nn.Module):
    def __init__(self, architecture_parameters: PolyNormerParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.input_projection = torch.nn.Linear(
            architecture_parameters.input_dim, architecture_parameters.hidden_dim * architecture_parameters.num_heads
        )

        self.local_stack = PolyNormerLocalStack(
            beta=architecture_parameters.beta,
            local_layers=architecture_parameters.local_layers,
            hidden_dim=architecture_parameters.hidden_dim,
            num_heads=architecture_parameters.num_heads,
            pre_norm=architecture_parameters.pre_norm,
            dropout=architecture_parameters.local_dropout,
        )

        self.global_stack = PolyNormerGlobalStack(
            num_layers=architecture_parameters.global_layers,
            hidden_dim=architecture_parameters.hidden_dim,
            num_heads=architecture_parameters.num_heads,
            qk_shared=architecture_parameters.qk_shared,
            beta=architecture_parameters.beta,
            dropout=architecture_parameters.global_dropout,
        )

        self.output_projection = torch.nn.Linear(
            architecture_parameters.hidden_dim * architecture_parameters.num_heads, architecture_parameters.output_dim
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.architecture_parameters.in_dropout, training=self.training)

        x = self.input_projection(x)
        x = F.dropout(x, p=self.architecture_parameters.local_dropout, training=self.training)

        x = self.local_stack(x, edge_index)

        x = self.global_stack(x, edge_index)

        x = self.output_projection(x)

        return x
