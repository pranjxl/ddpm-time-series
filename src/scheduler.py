import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define variance schedule and transfer to target device immediately
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion process: computes x_t directly from x_0.
        Dimension-agnostic: works for [B, 2], [B, C, L], or [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Grab the precomputed terms for the specific timesteps in the batch
        sqrt_alpha_cumprod = self.alpha_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_cumprod = (1.0 - self.alpha_cumprod[t]) ** 0.5
        
        # Reshape to (batch, 1, 1, ...) matching x_0's total dimensions
        shape = [x_0.shape[0]] + [1] * (x_0.dim() - 1)
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(*shape)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(*shape)
        
        # The reparameterization math
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise