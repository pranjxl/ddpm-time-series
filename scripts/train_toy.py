import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.scheduler import LinearNoiseScheduler
from src.dataset import get_swiss_roll_data
from src.models.toy_mlp import ToyMLP

def main():
    # Ensure the checkpoints directory exists before training
    os.makedirs("checkpoints", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = ToyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = LinearNoiseScheduler(device=device)

    epochs = 5000
    batch_size = 1024

    model.train()
    for epoch in range(1, epochs + 1):
        x_0 = get_swiss_roll_data(batch_size).to(device)
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        
        x_t, noise = scheduler.add_noise(x_0, t)
        predicted_noise = model(x_t, t)
        
        loss = nn.MSELoss()(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.5f}")

    torch.save(model.state_dict(), "checkpoints/toy_mlp.pth")
    print("Training complete. Model saved to checkpoints/toy_mlp.pth")

if __name__ == "__main__":
    main()