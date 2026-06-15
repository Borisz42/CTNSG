import torch
import torch.nn as nn

class PerceiverIOPooling(nn.Module):
    def __init__(self, in_dim: int = 256, num_latents: int = 8, latent_dim: int = 256, n_heads: int = 8):
        """
        Compresses a variable-length graph into exactly K dense tokens using Cross-Attention.
        Strictly processes the graph nodes as a permutation-invariant set (no self-attention
        or GNN message passing) to maintain a pure Information Bottleneck.
        """
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        # Learnable virtual node queries [num_latents, latent_dim]
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        # Standard multi-head cross-attention. 
        # By only applying cross-attention between queries and the input set,
        # we strictly enforce the Information Bottleneck condition.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=n_heads, 
            kdim=in_dim, 
            vdim=in_dim,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Decoded continuous graph nodes [batch_size, num_nodes, in_dim]
        Returns:
            pooled: Compressed graph tokens [batch_size, num_latents, latent_dim]
        """
        batch_size = x.size(0)
        
        # Expand latent queries for the batch
        # [batch_size, num_latents, latent_dim]
        q = self.latent_queries.expand(batch_size, -1, -1)
        
        # Perform cross-attention
        # Queries = Latents
        # Keys/Values = Decoded Graph Nodes (Permutation Invariant Set)
        attn_out, _ = self.cross_attention(query=q, key=x, value=x)
        
        # Add & Norm
        pooled = self.layer_norm(q + attn_out)
        
        return pooled
