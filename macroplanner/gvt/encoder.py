import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryPositionEmbedding, apply_rotary_pos_emb

class GVTAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin, mask=None):
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)

class GVTEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GVTAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ff(self.norm2(x))
        return x

class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, num_layers: int = 4, heads: int = 4, max_seq_len: int = 1024):
        """
        Uses a pure Graph VQ-Transformer (GVT) to encode continuous input features.
        Replaces GAT and integrates Rotary Position Embeddings (RoPE).
        """
        super().__init__()
        self.num_layers = num_layers
        
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.rope = RotaryPositionEmbedding(hidden_channels // heads, max_seq_len)
        
        self.layers = nn.ModuleList([
            GVTEncoderLayer(hidden_channels, heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges] (structure implicitly encoded by RoPE)
            batch: Batch vector [num_nodes]
        Returns:
            node_embeddings: Latent representations of nodes [num_nodes, hidden_channels]
            graph_embeddings: Pooled representation of the entire graph [num_graphs, hidden_channels]
        """
        N, C = x.shape
        h = self.in_proj(x)
        h = h.unsqueeze(0) # [1, N, hidden_channels]
        
        cos, sin = self.rope(h, seq_len=N)
        
        for layer in self.layers:
            h = layer(h, cos, sin, mask=None)
            
        h = self.norm(h)
        h = h.squeeze(0) # [N, hidden_channels]
        
        if batch is not None:
            try:
                from torch_geometric.nn import global_mean_pool
                graph_emb = global_mean_pool(h, batch)
            except ImportError:
                graph_emb = h.mean(dim=0, keepdim=True)
        else:
            graph_emb = h.mean(dim=0, keepdim=True)
            
        return h, graph_emb
