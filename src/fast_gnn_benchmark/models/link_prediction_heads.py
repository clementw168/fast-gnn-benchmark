import torch
import torch.nn.functional as F


class CosineSimilarityClassifier(torch.nn.Module):
    def forward(
        self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        return (embedding_1[edge_label_index[0]] * embedding_2[edge_label_index[1]]).sum(dim=-1)


class Hadamard_MLPPredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        num_layers: int = 2,
        use_residual: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.linear_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers.append(torch.nn.Linear(hidden_dim, 1))

        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, edge_label_index: torch.Tensor):
        x = embedding_1[edge_label_index[0]] * embedding_2[edge_label_index[1]]

        ori = x
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            if self.use_residual:
                x += ori
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.linear_layers[-1](x).squeeze(-1)
