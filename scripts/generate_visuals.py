import os
import sys
from pathlib import Path

import matplotlib
# Must be set before importing pyplot for headless safety on Kaggle
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

# Resolve the absolute path to the repo root dynamically.
# If cloned to /kaggle/working/ddpm-time-series, this points exactly there.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.scheduler import LinearNoiseScheduler
from src.models.toy_mlp import ToyMLP
from src.sampling import sample

def main():
    # 1. Setup absolute paths and device
    assets_dir = ROOT_DIR / "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running generation on: {device}")

    # 2. Load the trained model
    model = ToyMLP().to(device)
    checkpoint_path = ROOT_DIR / "checkpoints" / "toy_mlp.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find model at {checkpoint_path}. Did you run train_toy.py first?")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    scheduler = LinearNoiseScheduler(device=device)

    # 3. Run reverse diffusion
    print("Running reverse diffusion (this may take a moment)...")
    num_points = 1000
    final_sample, frames = sample(
        model, scheduler, shape=(num_points, 2), device=device, save_every=50
    )
    
    # 4. Reconstruct the timesteps array (Fixed off-by-one bug)
    frame_timesteps = list(reversed(range(0, scheduler.num_timesteps, 50)))
    print(f"Captured {len(frames)} frames for animation.")

    # 5. Generate the static snapshot grid
    snapshot_png_path = assets_dir / "swiss_roll_snapshots.png"
    print(f"Generating static grid: {snapshot_png_path}")
    snapshot_targets = [999, 750, 500, 250, 0]
    fig, axes = plt.subplots(1, len(snapshot_targets), figsize=(20, 4))
    
    for ax, target_t in zip(axes, snapshot_targets):
        closest_idx = min(range(len(frame_timesteps)), key=lambda i: abs(frame_timesteps[i] - target_t))
        pts = frames[closest_idx].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6)
        ax.set_title(f"t ≈ {target_t}")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(snapshot_png_path, dpi=150)
    plt.close(fig)

    # 6. Generate the animated GIF
    gif_path = assets_dir / "swiss_roll_reverse.gif"
    print(f"Generating animation: {gif_path}")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    scatter = ax.scatter([], [], s=3, alpha=0.6, color='steelblue')
    title = ax.set_title("")

    def update(frame_idx):
        pts = frames[frame_idx].cpu().numpy()
        scatter.set_offsets(pts)
        t_label = frame_timesteps[frame_idx] if frame_idx < len(frame_timesteps) else 0
        title.set_text(f"Reverse Diffusion — t = {t_label}")
        return scatter, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    ani.save(gif_path, writer="pillow", fps=12)
    plt.close(fig)
    
    print(f"Success! Check the {assets_dir} folder for your outputs.")

if __name__ == "__main__":
    main()