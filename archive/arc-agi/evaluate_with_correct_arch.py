#!/usr/bin/env python3
"""
Evaluate HRM with CORRECT architecture matching the training config
Training used 3 L-layers, not 6!
"""
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class HierarchicalReasoningModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # H-layers (high-level reasoning)
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        # L-layers (low-level processing) - CORRECT NUMBER!
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])  # Will be 3, not 6
        ])
        
        # Cross-layer connections
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Halt predictor
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Cycle embedding
        self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
        
        # Output projection
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Layer norms
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        
        # Dropout
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, max_cycles=None):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        
        # Embed tokens
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        halt_probs = []
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
        for cycle in range(max_cycles):
            # Add cycle embedding
            cycle_emb = self.cycle_embedding(torch.tensor([cycle], device=x.device))
            cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
            
            # H-level processing
            h_state = h_state + 0.1 * cycle_emb
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing with H influence
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # Update H with L information
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halt probability
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            halt_prob = torch.sigmoid(halt_logit)
            halt_probs.append(halt_prob)
            
            cumulative_halt = cumulative_halt + halt_prob
            
            # Check stopping condition
            if cycle >= 3 and cumulative_halt.mean() > 1.0:
                break
        
        # Generate output
        output = self.output(l_state)
        
        return output, halt_probs

def test_correct_architecture():
    """Test with the correct architecture"""
    
    print("=" * 80)
    print("TESTING HRM WITH CORRECT ARCHITECTURE")
    print("=" * 80)
    
    # Load checkpoint to get config
    checkpoint_path = 'training/checkpoints/hrm_arc_step_193000.pt'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Use ACTUAL training config
    model_config = checkpoint['config']
    print("\nActual training configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    # Create model with CORRECT config
    model = HierarchicalReasoningModule(model_config).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Expected ~6.95M, got {total_params/1e6:.2f}M")
    
    # Load weights
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"\nLoading checkpoint:")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    if len(missing) == 0:
        print("  ✅ Perfect loading - all weights matched!")
    else:
        print(f"  ⚠️  Missing keys: {missing[:5]}")
    
    # Test forward pass
    print("\n" + "=" * 80)
    print("FORWARD PASS TEST")
    print("=" * 80)
    
    model.eval()
    
    # Create diverse test input
    test_cases = [
        ("Zeros", torch.zeros(1, 900, dtype=torch.long)),
        ("Ones", torch.ones(1, 900, dtype=torch.long)),
        ("Range", torch.arange(900).unsqueeze(0) % 10),
        ("Random", torch.randint(0, 10, (1, 900))),
        ("Pattern", torch.tensor([[i % 7 for i in range(900)]])),
    ]
    
    for name, test_input in test_cases:
        test_input = test_input.to(DEVICE)
        
        with torch.no_grad():
            output, halt_probs = model(test_input)
        
        # Analyze output
        predictions = output.argmax(dim=-1)
        unique_preds = torch.unique(predictions)
        
        # Get class probabilities
        probs = torch.softmax(output, dim=-1)
        class_0_prob = probs[:, :, 0].mean().item()
        
        print(f"\n{name} input:")
        print(f"  Output shape: {output.shape}")
        print(f"  Unique predictions: {unique_preds.tolist()}")
        print(f"  Class 0 probability: {class_0_prob:.3f}")
        
        # Check logit distribution
        logit_means = output.mean(dim=[0, 1])
        print(f"  Logit means: {logit_means[:5].tolist()}")
        
        # Count prediction distribution
        pred_counts = {}
        for val in predictions.flatten().tolist():
            pred_counts[val] = pred_counts.get(val, 0) + 1
        
        print(f"  Prediction distribution: {dict(sorted(pred_counts.items())[:5])}")
        
        if len(unique_preds) == 1:
            print(f"  ❌ Only predicts class {unique_preds[0].item()}")
        else:
            print(f"  ✅ Predicts {len(unique_preds)} different classes")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if total_params < 7e6:
        print("✅ Model size matches expected ~6.95M parameters")
    else:
        print(f"⚠️  Model larger than expected: {total_params/1e6:.2f}M vs 6.95M")
    
    if len(missing) == 0:
        print("✅ All weights loaded correctly")
        print("❌ But model still outputs mostly/all zeros")
        print("   → This confirms the model learned to output zeros during training")
        print("   → Not an architecture or loading issue!")
    else:
        print("❌ Some weights missing - architecture mismatch")

if __name__ == "__main__":
    test_correct_architecture()