import torch
import torch.nn as nn

class LLMProjector(nn.Module):
    def __init__(self, in_dim: int = 256, out_dim: int = 4096):
        """
        Maps the pooled virtual nodes to the LLM embedding dimension.
        Validated Bottleneck Architecture: Linear -> ReLU -> Linear.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Pooled tokens [batch_size, K, in_dim]
        Returns:
            Projected tokens [batch_size, K, out_dim]
        """
        return self.proj(x)
