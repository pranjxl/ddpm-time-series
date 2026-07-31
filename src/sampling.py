import torch

@torch.no_grad()
def sample(model, scheduler, shape, device, save_every=None):
    """
    Reverse diffusion process: start from pure noise and iteratively denoise.
    
    Args:
        model: trained noise-prediction network (ToyMLP or unet_1d)
        scheduler: LinearNoiseScheduler instance
        shape: shape of the output tensor, e.g. (1024, 2) for 2D points
        device: torch device
        save_every: if set, save intermediate x_t every N steps for animation
        
    Returns:
        final denoised sample, and optionally a list of intermediate frames
    """
    model.eval()
    x_t = torch.randn(shape, device=device)
    
    frames = []
    num_steps = scheduler.num_timesteps
    
    for i in reversed(range(num_steps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        predicted_noise = model(x_t, t)
        
        alpha_t = scheduler.alphas[i]
        alpha_bar_t = scheduler.alpha_cumprod[i]
        beta_t = scheduler.betas[i]
        
        # Predict x_0 estimate implicitly via the reverse mean formula
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        if i > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(beta_t)
            x_t = mean + variance * noise
        else:
            # No noise added at the final step (t=0)
            x_t = mean
            
        if save_every is not None and (i % save_every == 0 or i == 0):
            frames.append(x_t.detach().cpu().clone())
            
    return x_t, frames