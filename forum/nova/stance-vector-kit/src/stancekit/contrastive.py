
import math, os
from typing import List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def pairwise_contrastive_loss(z1, z2, margin: float = 0.5, same: bool = True):
    # cosine similarity
    c = F.cosine_similarity(z1, z2)
    if same:
        # pull together: maximize cos -> minimize (1 - cos)
        return (1.0 - c).mean()
    else:
        # push apart: minimize max(0, cos - (1 - margin)) => encourage cos < 1 - margin
        return F.relu(c - (1.0 - margin)).mean()

def train_contrastive(X_pairs: List[Tuple[np.ndarray, np.ndarray, int]], in_dim: int, epochs: int = 10, lr: float = 1e-3, hidden: int = 256, out_dim: int = 64, margin: float = 0.5, device: str = None):
    assert _HAS_TORCH, "PyTorch not installed. Please install torch to use contrastive encoder."
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLP(in_dim, hidden=hidden, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    for ep in range(epochs):
        total = 0.0
        for a, b, y in X_pairs:
            za = model(to_tensor(a))
            zb = model(to_tensor(b))
            loss_same = pairwise_contrastive_loss(za, zb, margin=margin, same=(y==1))
            # for negatives, invert 'same' flag
            if y == 0:
                loss = pairwise_contrastive_loss(za, zb, margin=margin, same=False)
            else:
                loss = loss_same
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())
        if (ep+1) % max(1, epochs//5) == 0:
            print(f"epoch {ep+1}/{epochs} loss={total/len(X_pairs):.4f}")
    return model

def encode(model, X: np.ndarray, device: str = None) -> np.ndarray:
    assert _HAS_TORCH, "PyTorch not installed."
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        z = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return z
