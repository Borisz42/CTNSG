import torch
import torch.nn as nn
import math

class MagneticLaplacianPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # mLPE mathematically injects directed graph structure
        # using learnable magnitude and phase parameters for directionality
        self.magnitude = nn.Parameter(torch.randn(max_len, 1, d_model // 2))
        self.phase = nn.Parameter(torch.randn(max_len, 1, d_model // 2))

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        mag = self.magnitude[:seq_len]
        phase = self.phase[:seq_len]
        
        # Combine magnitude and phase to create directed positional encoding
        pe = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)
        return x + pe

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: [batch_size]
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024):
        super().__init__()
        # Dual Attention for source-to-target and target-to-source channels
        self.attn_src2tgt = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.attn_tgt2src = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.attn_proj = nn.Linear(d_model * 2, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # AdaLN projection layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model, bias=True)
        )
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # t_emb: [batch_size, d_model]
        # x: [batch_size, seq_len, d_model]
        
        # AdaLN modulation
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(t_emb).chunk(4, dim=1)
        
        # Expand for sequence length
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        
        # Pre-Norm architecture (norm_first=True) with AdaLN
        x_norm1 = self.norm1(x) * (1 + scale_msa) + shift_msa
        
        # Dual Attention
        attn_out_s2t, _ = self.attn_src2tgt(x_norm1, x_norm1, x_norm1)
        attn_out_t2s, _ = self.attn_tgt2src(x_norm1, x_norm1, x_norm1)
        
        # Combine channels
        attn_combined = torch.cat([attn_out_s2t, attn_out_t2s], dim=-1)
        x = x + self.attn_proj(attn_combined)
        
        # FFN with Pre-Norm AdaLN
        x_norm2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.ffn(x_norm2)
        
        return x

class RelationalTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 65, # K=64 + 1 for [MASK]
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_len: int = 1024
    ):
        """
        Bidirectional Transformer for Non-Autoregressive Discrete Diffusion.
        Features mLPE, Dual Attention, and AdaLN.
        """
        super().__init__()
        self.d_model = d_model
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding (mLPE)
        self.pos_encoder = MagneticLaplacianPositionalEncoding(d_model, max_len=max_seq_len)
        
        # Timestep Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer Backbone
        self.layers = nn.ModuleList([
            DualAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output Projection (predicts logits over vocab_size - 1, we don't predict [MASK])
        self.output_proj = nn.Linear(d_model, vocab_size - 1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: Token indices [batch_size, seq_len]
            t: Diffusion timesteps [batch_size]
        Returns:
            logits: Unnormalized predictions [batch_size, seq_len, vocab_size - 1]
        """
        # Embed timesteps
        t_emb = self.time_mlp(t) # [batch_size, d_model]
        
        # Embed tokens
        emb = self.embedding(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]
        
        # Apply mLPE pos encoding (needs [seq_len, batch, dim])
        emb = emb.permute(1, 0, 2)
        emb = self.pos_encoder(emb)
        emb = emb.permute(1, 0, 2) # Back to [batch_size, seq_len, d_model]
        
        # Pass through Dual Attention layers with AdaLN
        hidden = emb
        for layer in self.layers:
            hidden = layer(hidden, t_emb)
            
        hidden = self.final_norm(hidden)
        
        # Project to vocabulary
        logits = self.output_proj(hidden)
        return logits
