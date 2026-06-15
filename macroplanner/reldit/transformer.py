import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
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
        """
        super().__init__()
        self.d_model = d_model
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True # [batch_size, seq_len, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Projection (predicts logits over vocab_size - 1, we don't predict [MASK])
        self.output_proj = nn.Linear(d_model, vocab_size - 1)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Token indices [batch_size, seq_len]
        Returns:
            logits: Unnormalized predictions [batch_size, seq_len, vocab_size - 1]
        """
        # Embed 
        emb = self.embedding(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]
        
        # Apply pos encoding (needs [seq_len, batch, dim])
        emb = emb.permute(1, 0, 2)
        emb = self.pos_encoder(emb)
        emb = emb.permute(1, 0, 2) # Back to [batch_size, seq_len, d_model]
        
        # Pass through bidirectional Transformer
        hidden = self.transformer_encoder(emb)
        
        # Project to vocabulary
        logits = self.output_proj(hidden)
        return logits
