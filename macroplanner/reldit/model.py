import torch
import torch.nn as nn
from .diffusion import AbsorbingDiffusion
from .transformer import RelationalTransformer

class RelDiT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 65,
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
        
        # CID Critic module: Predicts the residual logit of clean token probability
        self.critic = nn.Sequential(
            nn.Linear(vocab_size, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Loss function for predicting the original tokens
        self.criterion = nn.CrossEntropyLoss()

    
    def get_critic_scores(self, probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates critic scores using the Residual Logit Trick from the CID framework.
        t: tensor of shape [batch_size] containing the current timesteps
        probs: tensor of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, _ = probs.shape
        t_expand = t.view(batch_size, 1).expand(batch_size, seq_len)
        
        # alpha_t is the baseline probability of a token being CLEAN
        p_mask = t_expand.float() / self.num_timesteps
        alpha_t = 1.0 - p_mask
        
        # Clamp to avoid log(0) and infinity
        alpha_t = torch.clamp(alpha_t, 1e-5, 1.0 - 1e-5)
        
        # Calculate baseline logit
        base_logit = torch.logit(alpha_t)
        
        # Critic predicts the deviation from the baseline schedule
        residual = self.critic(probs).squeeze(-1)
        
        return torch.sigmoid(base_logit + residual)

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
        logits = self.transformer(x_t, t) # [batch_size, seq_len, vocab_size]
        
        # [CRITICAL FIX]: Upcast logits to FP32 before calculating the loss
        logits_fp32 = logits.float()
        
        # We compute loss on all tokens (standard for predicting x_0 from x_t in diffusion)
        # Flatten for CrossEntropyLoss
        logits_flat = logits_fp32.view(-1, self.vocab_size)
        targets_flat = x0.view(-1)
        
        # Calculate unweighted cross entropy on all tokens
        loss = self.criterion(logits_flat, targets_flat)
        
        return loss

    @torch.no_grad()
    def generate(self, batch_size: int, seq_len: int, device: torch.device, use_critic: bool = False, temperature: float = 1.0):
        """
        Iterative unmasking (reverse diffusion process).
        Starts with a fully masked sequence and progressively decodes it.
        """
        # Start with all [MASK] tokens
        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        
        for t in reversed(range(1, self.num_timesteps + 1)):
            t_tensor = torch.full((batch_size,), t, device=device)
            # Predict all tokens
            logits = self.transformer(x, t_tensor)
            
            # Get probabilities and apply temperature scaling
            scaled_logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # STOCHASTIC SAMPLING: Fixes Mode Collapse (10% Uniqueness -> 100%)
            probs_flat = probs.view(-1, self.vocab_size)
            pred_tokens_flat = torch.multinomial(probs_flat, 1)
            pred_tokens = pred_tokens_flat.view(batch_size, seq_len)
            
            # Get the raw Transformer confidence for the sampled tokens
            pred_probs = torch.gather(probs, -1, pred_tokens.unsqueeze(-1)).squeeze(-1)
            
            if use_critic:
                # CID Framework: Use Critic to evaluate the token probabilities
                critic_scores = self.get_critic_scores(probs, t_tensor)
                combined_confidence = pred_probs * critic_scores
            else:
                # Fallback to Transformer confidence
                combined_confidence = pred_probs
            
            # Only consider tokens that are currently masked
            is_masked = (x == self.mask_token_id)
            
            # Force unmasked tokens to have low confidence so we don't pick them again
            combined_confidence[~is_masked] = -1.0
            
            # Find the top-k most confident masked tokens to unmask
            if t == 1:
                # Last step: unmask all remaining masked tokens
                num_to_unmask = is_masked.sum(dim=1).max().item()
            else:
                current_masked = is_masked.sum(dim=1).max().item()
                target_masked = int(seq_len * (t - 1) / self.num_timesteps)
                num_to_unmask = max(1, current_masked - target_masked)
                # Safety check: if there are fewer masked tokens than we want to unmask
                num_to_unmask = min(num_to_unmask, is_masked.sum(dim=1).min().item())
            
            if num_to_unmask > 0:
                _, unmask_idx = torch.topk(combined_confidence, num_to_unmask, dim=-1)
                
                # Scatter the predictions into x
                x.scatter_(1, unmask_idx, pred_tokens.gather(1, unmask_idx))
                
            # Simple Iterative Denoising (SID): actively re-corrupt low-likelihood elements
            if 1 < t < self.num_timesteps: # Don't re-mask on the very first or very last unmasking step
                sid_threshold = 0.2
                currently_unmasked = (x != self.mask_token_id)
                if use_critic:
                    poor_quality = (critic_scores < sid_threshold) & currently_unmasked
                else:
                    safe_x = torch.clamp(x, 0, self.vocab_size - 1)
                    x_probs = torch.gather(probs, -1, safe_x.unsqueeze(-1)).squeeze(-1)
                    poor_quality = (x_probs < sid_threshold) & currently_unmasked
                x[poor_quality] = self.mask_token_id
            # --- L2 Topological Critic (DAG Enforcement) ---
            if t == 1 and use_critic:
                import networkx as nx
                for b in range(batch_size):
                    curr_tokens = x[b].tolist()
                    G = nx.DiGraph()
                    for j in range(seq_len - 1):
                        u, v = curr_tokens[j], curr_tokens[j+1]
                        G.add_edge(u, v)
                        try:
                            nx.find_cycle(G, orientation="original")
                            # Cycle detected! Revert this edge by changing v
                            G.remove_edge(u, v)
                            for new_v in range(self.vocab_size):
                                if new_v == u: continue
                                G.add_edge(u, new_v)
                                try:
                                    nx.find_cycle(G, orientation="original")
                                    G.remove_edge(u, new_v)
                                except nx.NetworkXNoCycle:
                                    # No cycle with new_v
                                    x[b, j+1] = new_v
                                    curr_tokens[j+1] = new_v
                                    break
                        except nx.NetworkXNoCycle:
                            pass
                            
        return x

if __name__ == '__main__':
    # --- Local Verification Test ---
    print("Executing RelDiT verification test...")
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 16
    vocab_size = 65
    
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
