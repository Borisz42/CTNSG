import torch
import torch.nn as nn

class RVQDecoder(nn.Module):
    def __init__(self, num_embeddings: int = 64, embedding_dim: int = 256, num_quantizers: int = 4):
        """
        Decodes Residual Vector Quantization (RVQ) discrete tokens back into continuous space.
        Instantiates N separate embedding tables and sums their outputs to preserve hierarchy.
        """
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        
        # Instantiate an embedding table for each level of the RVQ hierarchy
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) for _ in range(num_quantizers)
        ])

    def forward(self, indices: torch.Tensor):
        """
        Args:
            indices: Discrete hierarchical tokens [batch_size, num_nodes, num_quantizers]
        Returns:
            z_q: Continuous representations [batch_size, num_nodes, embedding_dim]
        """
        batch_size, num_nodes, n_q = indices.shape
        assert n_q == self.num_quantizers, f"Expected {self.num_quantizers} quantizers, got {n_q}"
        
        z_q = torch.zeros((batch_size, num_nodes, self.embedding_dim), device=indices.device)
        
        # Look up and sum the continuous vectors for each residual level
        for i, embedding_layer in enumerate(self.embeddings):
            z_q += embedding_layer(indices[:, :, i])
            
        return z_q
