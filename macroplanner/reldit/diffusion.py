import torch

class AbsorbingDiffusion:
    def __init__(self, num_timesteps: int = 256, mask_token_id: int = 64):
        """
        Implements Discrete Absorbing State Diffusion.
        Tokens are progressively replaced by a [MASK] token.
        
        Args:
            num_timesteps: Total steps in the diffusion schedule.
            mask_token_id: The ID of the [MASK] token. (If K=64, indices are 0-63, mask is 64).
        """
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion process (q).
        Replaces a fraction of tokens in x0 with the mask_token_id based on timestep t.
        
        Args:
            x0: Original discrete tokens [batch_size, seq_len]
            t: Timesteps [batch_size] in range [0, num_timesteps]
        
        Returns:
            x_t: Noisy tokens [batch_size, seq_len]
            mask: Boolean tensor indicating which tokens were masked.
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Calculate the probability of masking at time t.
        # Linear schedule where p_mask = t / num_timesteps
        p_mask = t.float() / self.num_timesteps
        p_mask = p_mask.unsqueeze(-1).expand(batch_size, seq_len)
        
        # Sample uniform noise to determine which tokens to mask
        rand_probs = torch.rand((batch_size, seq_len), device=device)
        
        # Mask where rand_probs < p_mask
        mask = rand_probs < p_mask
        
        # Create x_t by applying the mask
        x_t = x0.clone()
        x_t[mask] = self.mask_token_id
        
        return x_t, mask
