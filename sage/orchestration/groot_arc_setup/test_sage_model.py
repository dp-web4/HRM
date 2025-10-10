#!/usr/bin/env python3
"""Quick test of SAGE student model with smaller config."""

import torch
from sage_student_model import SAGEStudent, DistillationLoss

print("="*80)
print("SAGE Student Model Quick Test")
print("="*80)

# Create small model for testing
print("\n📦 Creating model...")
model = SAGEStudent(
    input_dim=2048,
    hidden_dim=256,      # Smaller for testing
    num_layers=2,        # Fewer layers for testing
    num_heads=4,
    ffn_dim=1024,
    max_seq_len=6144,    # Just above our 5164
    grid_size=30,
    num_classes=10,
    dropout=0.1,
)

# Parameter count
params = model.get_num_params()
print(f"✅ Model created")
print(f"   Total parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")

# Test forward pass
print(f"\n🔬 Testing forward pass...")
batch_size = 1
seq_len = 100  # Small for testing

groot_features = torch.randn(batch_size, seq_len, 2048)
attention_mask = torch.ones(batch_size, seq_len)

grid_logits, student_features = model(groot_features, attention_mask)

print(f"✅ Forward pass successful")
print(f"   Input: {list(groot_features.shape)}")
print(f"   Output: {list(grid_logits.shape)}")
print(f"   Features: {list(student_features.shape)}")

# Test prediction
predicted_grid = model.predict_grid(groot_features, attention_mask)
print(f"   Predicted grid: {list(predicted_grid.shape)}")

# Test loss
print(f"\n🔬 Testing loss...")
loss_fn = DistillationLoss(student_hidden_dim=256, teacher_hidden_dim=2048)
target_grid = torch.randint(0, 10, (batch_size, 30, 30))

total_loss, loss_dict = loss_fn(
    grid_logits, target_grid, student_features, groot_features, attention_mask
)

print(f"✅ Loss computation successful")
print(f"   Total loss: {loss_dict['total']:.4f}")
print(f"   Task loss: {loss_dict['task']:.4f}")
print(f"   Feature MSE: {loss_dict['feature_mse']:.4f}")

print(f"\n✅ All tests passed!")
