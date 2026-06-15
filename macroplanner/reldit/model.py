import torch
import torch.nn as nn
from .diffusion import AbsorbingDiffusion
from .transformer import RelationalTransformer

class RelDiT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        num_timesteps: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size # MASK is the last token (K=64 -> MASK=64)
        self.num_timesteps = num_timesteps
        
        self.diffusion = AbsorbingDiffusion(
            num_timesteps=num_timesteps, 
            mask_token_id=self.mask_token_id
        )
        
        self.transformer = RelationalTransformer(
            vocab_size=vocab_size + 1, # +1 for MASK
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )
        
        # Critic module: evaluates the structural validity of generated tokens
        self.critic = nn.Sequential(
            nn.Linear(vocab_size, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Loss function for predicting the original tokens
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x0: torch.Tensor):
        """
        Training pass:
        1. Sample random timesteps.
        2. Apply forward masking (add_noise).
        3. Predict original tokens using the Transformer.
        4. Compute loss only on the masked tokens.
        
        Args:
            x0: Original tokens [batch_size, seq_len]
        Returns:
            loss: Unweighted CrossEntropy loss
        """
        batch_size = x0.size(0)
        device = x0.device
        
        # Sample random timesteps uniformly
        t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device)
        
        # Add noise (mask tokens)
        x_t, mask = self.diffusion.add_noise(x0, t)
        
        # Predict logits
        logits = self.transformer(x_t) # [batch_size, seq_len, vocab_size]
        
        # We only compute loss on the tokens that were masked.
        # Flatten for CrossEntropyLoss
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = x0.view(-1)
        mask_flat = mask.view(-1)
        
        # Calculate unweighted cross entropy on masked tokens
        loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
        
        return loss

    @torch.no_grad()
    def generate(self, batch_size: int, seq_len: int, device: torch.device):
        """
        Iterative unmasking (reverse diffusion process).
        Starts with a fully masked sequence and progressively decodes it.
        """
        # Start with all [MASK] tokens
        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        
        for t in reversed(range(1, self.num_timesteps + 1)):
            # Predict all tokens
            logits = self.transformer(x)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            pred_tokens = torch.argmax(probs, dim=-1)
            
            # Critic evaluates the token probabilities
            critic_scores = self.critic(probs).squeeze(-1) # [batch_size, seq_len]
            
            # Unmask the highest confidence predictions linearly
            unmasked_count = seq_len - (seq_len * (t - 1) // self.num_timesteps)
            
            # Get confidence of the predicted tokens
            pred_probs = torch.gather(probs, -1, pred_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Combine prediction confidence with Critic evaluation for selection
            combined_confidence = pred_probs * critic_scores
            
            # Only consider tokens that are currently masked
            is_masked = (x == self.mask_token_id)
            
            # Force unmasked tokens to have low confidence so we don't pick them again
            combined_confidence[~is_masked] = -1.0
            
            # Find the top-k most confident masked tokens to unmask
            num_to_unmask = max(1, seq_len // self.num_timesteps)
            
            # Safety check: if there are fewer masked tokens than we want to unmask
            num_to_unmask = min(num_to_unmask, is_masked.sum(dim=1).min().item())
            
            if num_to_unmask > 0:
                _, unmask_idx = torch.topk(combined_confidence, num_to_unmask, dim=-1)
                
                # Scatter the predictions into x
                x.scatter_(1, unmask_idx, pred_tokens.gather(1, unmask_idx))
                
            # Simple Iterative Denoising (SID): actively re-corrupt low-likelihood elements
            # If an already unmasked token falls below the Critic's quality threshold, mask it again
            if t < self.num_timesteps: # Don't re-mask on the very first unmasking step
                sid_threshold = 0.2
                poor_quality = (critic_scores < sid_threshold) & (~is_masked)
                x[poor_quality] = self.mask_token_id
            
        return x

if __name__ == '__main__':
    # --- Local Verification Test ---
    print("Executing RelDiT verification test...")
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 16
    vocab_size = 64
    
    # 1. Create dummy target tokens (from GVT)
    x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 2. Instantiate Model
    model = RelDiT(vocab_size=vocab_size, num_timesteps=50, d_model=128, nhead=4, num_layers=2)
    
    # 3. Training Forward Pass
    loss = model(x0)
    
    print("\n--- Training Verification ---")
    print(f"Target Sequence Shape: {x0.shape}")
    print(f"Computed Masked Cross-Entropy Loss: {loss.item():.4f}")
    
    # 4. Generation Pass
    print("\n--- Generation Verification ---")
    generated_tokens = model.generate(batch_size=1, seq_len=seq_len, device=x0.device)
    print(f"Generated Sequence Shape: {generated_tokens.shape}")
    print(f"Generated Tokens:\n{generated_tokens[0].tolist()}")
    
    print("\n[SUCCESS] RelDiT training and generation passes completed.")
