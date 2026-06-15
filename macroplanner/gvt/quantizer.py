import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, temperature: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # The codebook (K x D)
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize codebook weights uniformly to prevent early collapse
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: Continuous latents from encoder, shape [..., D]
        Returns:
            z_q: Quantized latents, shape [..., D]
            indices: Codebook indices, shape [...]
            commit_loss: Commitment loss scalar
        """
        # Flatten z to [N, D]
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Calculate distances between z and codebook
        # (z - c)^2 = z^2 + c^2 - 2zc
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.codebook.weight.t())
        ) # [N, K]
        
        # Gumbel-Softmax reparameterization to ensure differentiability
        # and significantly mitigate codebook collapse by injecting noise
        logits = -d
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1) # [N, K]
        
        # Multiply with codebook to get quantized vectors
        z_q = torch.matmul(soft_one_hot, self.codebook.weight) # [N, D]
        z_q = z_q.view(z.shape)
        
        # Get discrete indices
        indices = torch.argmax(soft_one_hot, dim=-1).view(z.shape[:-1])
        
        # Commitment loss: ||z_q.detach() - z||^2 + beta * ||z_q - z.detach()||^2
        # Beta is traditionally set to 0.25
        beta = 0.25
        commit_loss = F.mse_loss(z_q.detach(), z) + beta * F.mse_loss(z_q, z.detach())
        
        # Straight-through estimator for gradients
        z_q = z + (z_q - z).detach()
        
        return z_q, indices, commit_loss

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 64, embedding_dim: int = 256, num_quantizers: int = 4):
        """
        Implements Residual Vector Quantization (RVQ) to create a hierarchical 
        discrete representation space, scaling combinations to (K^N).
        """
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim) 
            for _ in range(num_quantizers)
        ])

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: Continuous latents, shape [..., D]
        Returns:
            z_q: Sum of quantized latents, shape [..., D]
            all_indices: Tensor of shape [..., num_quantizers]
            total_commit_loss: Sum of commitment losses across all stages
        """
        z_q_total = torch.zeros_like(z)
        residual = z
        all_indices = []
        total_commit_loss = 0.0
        
        for quantizer in self.quantizers:
            z_q, indices, commit_loss = quantizer(residual)
            
            z_q_total = z_q_total + z_q
            # The residual passed to the next stage is the error from this stage
            residual = residual - z_q
            
            all_indices.append(indices)
            total_commit_loss = total_commit_loss + commit_loss
            
        # Stack indices: shape [..., num_quantizers]
        all_indices = torch.stack(all_indices, dim=-1)
        
        return z_q_total, all_indices, total_commit_loss
