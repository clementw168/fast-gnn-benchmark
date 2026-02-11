import torch
import torch.nn.functional as F

from fast_gnn_benchmark.models.backbones.gnn import load_gnn
from fast_gnn_benchmark.schemas.model import SGFormerParameters


def full_attention_conv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, compute_attention: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Implements the full attention convolution mechanism used in SGFormer.

    This function computes the attention-weighted aggregation of values using the formula:

    Output = D^{-1} * [V + (1/N) * Q * (K^T * V)]

    Where:
    - D = diag(1 + (1/N) * Q * (K^T * 1)) is the normalization diagonal matrix
    - Q, K, V are the query, key, and value tensors respectively
    - N is the number of nodes
    - 1 represents a vector of ones

    Args:
        query: Query tensor of shape [N, H, M] where N=nodes, H=heads, M=features
        key: Key tensor of shape [L, H, M] where L=keys, H=heads, M=features
        value: Value tensor of shape [L, H, D] where L=values, H=heads, D=output_dim
        output_attn: Whether to return attention weights for visualization

    Returns:
        Tuple of (attention_output, attention_weights) where attention_weights is None
        if output_attn=False, otherwise contains the computed attention matrix
    """
    # Normalize input tensors using L2 norm
    query = query / torch.norm(query, p=2)  # [N, H, M]
    key = key / torch.norm(key, p=2)  # [L, H, M]
    nodes_num = query.shape[0]
    # assert isinstance(nodes_num, torch.Tensor)

    # Compute numerator: V + (1/N) * Q * (K^T * V)
    # First compute K^T * V using einsum
    key_transpose_value = torch.einsum("lhm,lhd->hmd", key, value)
    # Then compute Q * (K^T * V)
    attention_num = torch.einsum("nhm,hmd->nhd", query, key_transpose_value)  # [N, H, D]
    # Add the V term (scaled by number of nodes)
    attention_num += nodes_num * value

    # Compute denominator: diag(1 + (1/N) * Q * (K^T * 1))
    # Create vector of ones for the K^T * 1 operation
    all_ones = torch.ones([key.shape[0]]).to(key.device)
    # Compute K^T * 1
    key_sum = torch.einsum("lhm,l->hm", key, all_ones)
    # Compute Q * (K^T * 1)
    attention_normalizer = torch.einsum("nhm,hm->nh", query, key_sum)  # [N, H]

    # Apply normalization and compute final output
    # Add the identity term (1) to the normalizer
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * nodes_num
    # Final division: numerator / denominator
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # Compute attention weights for visualization if requested
    if compute_attention:
        attention = torch.einsum("nhm,lhm->nlh", query, key).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdim=True)  # [N,1]
        attention = attention / normalizer

        return attn_output, attention

    return attn_output, None


class FullAttentionConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int, use_weights: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.use_weights = use_weights
        self.heads = heads
        self.head_dim = out_channels // heads

        self.key_projection = torch.nn.Linear(in_channels, self.out_channels)
        self.query_projection = torch.nn.Linear(in_channels, self.out_channels)

        if use_weights:
            self.value_projection = torch.nn.Linear(in_channels, self.out_channels)
        else:
            self.value_projection = None

    def forward(self, x, edge_index, compute_attention: bool = False):
        # [N, M] -> [N, H, M]
        query = self.query_projection(x).reshape(-1, self.heads, self.head_dim)
        key = self.key_projection(x).reshape(-1, self.heads, self.head_dim)
        value = (
            self.value_projection(x).reshape(-1, self.heads, self.head_dim)
            if self.value_projection is not None
            else x.reshape(-1, self.heads, self.head_dim)
        )
        attention_output, attention_weights = full_attention_conv(query, key, value, compute_attention)

        attention_output = attention_output.reshape(-1, self.out_channels)

        return attention_output, attention_weights


class FullAttentionStack(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 1,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
        use_weights: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.input_projection = torch.nn.Linear(in_channels, hidden_dim)

        self.attention_convs = torch.nn.ModuleList(
            [FullAttentionConv(hidden_dim, hidden_dim, self.num_heads, use_weights) for _ in range(self.num_layers)]
        )

        if use_layer_norm:
            self.input_layer_norm = torch.nn.LayerNorm(hidden_dim)
            self.layer_norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        x = self.input_projection(x)
        if self.use_layer_norm:
            x = self.input_layer_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        prev_x = x

        for layer_index in range(self.num_layers):
            x, _ = self.attention_convs[layer_index](x, edge_index)

            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * prev_x

            if self.use_layer_norm:
                x = self.layer_norms[layer_index](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            prev_x = x

        return x


class SGFormer(torch.nn.Module):
    """
    SGFormer (Simple Graph Former) model implementation.

    This is a placeholder implementation for the SGFormer architecture.
    The model uses full attention convolution mechanism for graph neural networks.

    Args:
        architecture_parameters: Configuration parameters for the SGFormer model
    """

    def __init__(self, architecture_parameters: SGFormerParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.gnn = load_gnn(architecture_parameters.gnn_parameters)

        self.attention_convs = self.load_attention_convs()
        self.output_projection = torch.nn.Linear(
            self.architecture_parameters.hidden_dim,
            self.architecture_parameters.output_dim,
        )

    def load_attention_convs(self):
        return FullAttentionStack(
            in_channels=self.architecture_parameters.input_dim,
            hidden_dim=self.architecture_parameters.hidden_dim,
            num_layers=self.architecture_parameters.num_layers,
            num_heads=self.architecture_parameters.num_heads,
            alpha=self.architecture_parameters.alpha,
            dropout=self.architecture_parameters.dropout,
            use_layer_norm=self.architecture_parameters.use_layer_norm,
            use_residual=self.architecture_parameters.use_residual,
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the SGFormer model.

        Args:
            x: Node feature tensor of shape [N, input_dim]
            edge_index: Edge connectivity tensor of shape [2, E]

        Returns:
            Node embeddings of shape [N, output_dim]
        """
        transformer_output = self.attention_convs(x, edge_index)

        if self.architecture_parameters.use_graph:
            gnn_output = self.gnn(x, edge_index)
            x = (
                self.architecture_parameters.graph_weight * gnn_output
                + (1 - self.architecture_parameters.graph_weight) * transformer_output
            )
        else:
            x = transformer_output

        x = self.output_projection(x)

        return x
