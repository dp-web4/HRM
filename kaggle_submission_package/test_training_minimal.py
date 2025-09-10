#!/usr/bin/env python3
"""Minimal training test to debug GPU issue"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from pathlib import Path

print("Starting minimal training test...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("WARNING: Using CPU")

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(900, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 900 * 12)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).view(-1, 900, 12)

# Create model
model = SimpleModel().to(device)
print(f"Model on device: {next(model.parameters()).device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Create dummy data
batch_size = 16
dummy_input = torch.randn(batch_size, 900).to(device)
dummy_target = torch.randint(0, 12, (batch_size, 900)).to(device)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nStarting training loop...")
for epoch in range(3):
    start = time.time()
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output.transpose(1, 2), dummy_target)
    loss.backward()
    optimizer.step()
    
    elapsed = time.time() - start
    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Time={elapsed:.3f}s")
    
    # Check GPU usage
    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e6:.1f}MB")

print("\nTest complete!")