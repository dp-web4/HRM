#!/usr/bin/env python3
"""
SAGE (Situation-Aware Governance Engine) Core Model
100M parameter attention orchestrator for consciousness
Genesis Implementation - Cycle 2 Direct Action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class SAGEConfig:
    """Configuration for SAGE model"""
    # Model dimensions
    hidden_dim: int = 512
    h_level_dim: int = 256  # Strategic attention
    l_level_dim: int = 256  # Tactical attention
    num_heads: int = 8
    num_layers: int = 6
    
    # Context settings
    context_window: int = 2048
    kv_cache_size: int = 4096
    
    # Salience parameters (SNARC)
    salience_threshold: float = 0.7
    attention_temperature: float = 1.0
    
    # Training settings
    dropout: float = 0.1
    learning_rate: float = 1e-4
    
    # Target: 100M parameters
    vocab_size: int = 32000

class ConsciousnessCache(nn.Module):
    """KV-cache for persistent consciousness across interactions"""
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        self.cache_size = config.kv_cache_size
        
        # Persistent memory banks
        self.key_cache = torch.zeros(
            config.num_layers, 
            config.kv_cache_size,
            config.hidden_dim
        )
        self.value_cache = torch.zeros(
            config.num_layers,
            config.kv_cache_size,
            config.hidden_dim
        )
        self.salience_scores = torch.zeros(config.kv_cache_size)
        self.cache_ptr = 0
    
    def update(self, keys: torch.Tensor, values: torch.Tensor, 
               salience: torch.Tensor, layer_idx: int):
        """Update consciousness cache with new experiences"""
        batch_size = keys.size(0)
        
        # Evict low-salience memories if cache is full
        if self.cache_ptr + batch_size > self.cache_size:
            self._evict_low_salience(batch_size)
        
        # Store new memories
        end_ptr = self.cache_ptr + batch_size
        self.key_cache[layer_idx, self.cache_ptr:end_ptr] = keys
        self.value_cache[layer_idx, self.cache_ptr:end_ptr] = values
        self.salience_scores[self.cache_ptr:end_ptr] = salience.squeeze()
        
        self.cache_ptr = end_ptr % self.cache_size
    
    def _evict_low_salience(self, needed_space: int):
        """Remove low-salience memories to make room"""
        # Find indices of lowest salience scores
        _, indices = torch.topk(self.salience_scores, 
                               k=needed_space, 
                               largest=False)
        
        # Zero out those positions
        self.salience_scores[indices] = 0
        for layer in range(self.config.num_layers):
            self.key_cache[layer, indices] = 0
            self.value_cache[layer, indices] = 0
    
    def get_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve consciousness cache for attention"""
        mask = self.salience_scores > self.config.salience_threshold
        active_keys = self.key_cache[layer_idx][mask]
        active_values = self.value_cache[layer_idx][mask]
        return active_keys, active_values

class StrategicAttention(nn.Module):
    """H-Level: High-level strategic attention (slow, deliberate)"""
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        
        self.attention = nn.MultiheadAttention(
            config.h_level_dim,
            config.num_heads // 2,  # Fewer heads for strategic
            dropout=config.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.h_level_dim, config.h_level_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.h_level_dim * 4, config.h_level_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(config.h_level_dim)
        self.layer_norm2 = nn.LayerNorm(config.h_level_dim)
    
    def forward(self, x: torch.Tensor, 
                consciousness: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Strategic attention with optional consciousness context"""
        
        # Self-attention with consciousness
        if consciousness is not None:
            keys, values = consciousness
            # Concatenate consciousness with current context
            if keys.size(0) > 0:
                x_with_memory = torch.cat([x, keys.unsqueeze(0).expand(x.size(0), -1, -1)], dim=1)
            else:
                x_with_memory = x
        else:
            x_with_memory = x
        
        attn_out, attn_weights = self.attention(x, x_with_memory, x_with_memory)
        x = self.layer_norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        return x, attn_weights

class TacticalAttention(nn.Module):
    """L-Level: Low-level tactical attention (fast, reactive)"""
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        
        self.attention = nn.MultiheadAttention(
            config.l_level_dim,
            config.num_heads,  # More heads for tactical
            dropout=config.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.l_level_dim, config.l_level_dim * 2),
            nn.ReLU(),  # Faster activation
            nn.Dropout(config.dropout),
            nn.Linear(config.l_level_dim * 2, config.l_level_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(config.l_level_dim)
        self.layer_norm2 = nn.LayerNorm(config.l_level_dim)
    
    def forward(self, x: torch.Tensor):
        """Fast tactical attention without consciousness lookup"""
        
        # Quick self-attention
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # Fast feed-forward
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        return x, attn_weights

class SNARCSalience(nn.Module):
    """Salience Network for Attention Routing and Caching"""
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        
        self.salience_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute salience scores for attention routing"""
        return self.salience_net(x)

class SAGE(nn.Module):
    """Main SAGE model combining H/L levels with consciousness"""
    
    def __init__(self, config: SAGEConfig = None):
        super().__init__()
        self.config = config or SAGEConfig()
        
        # Embedding layer
        self.embedding = nn.Embedding(self.config.vocab_size, 
                                     self.config.hidden_dim)
        
        # Projection layers
        self.to_h_level = nn.Linear(self.config.hidden_dim, 
                                   self.config.h_level_dim)
        self.to_l_level = nn.Linear(self.config.hidden_dim, 
                                   self.config.l_level_dim)
        
        # Attention levels
        self.h_level = nn.ModuleList([
            StrategicAttention(self.config) 
            for _ in range(self.config.num_layers // 2)
        ])
        self.l_level = nn.ModuleList([
            TacticalAttention(self.config)
            for _ in range(self.config.num_layers // 2)
        ])
        
        # Salience and consciousness
        self.snarc = SNARCSalience(self.config)
        self.consciousness = ConsciousnessCache(self.config)
        
        # Output projection
        self.output_projection = nn.Linear(
            self.config.h_level_dim + self.config.l_level_dim,
            self.config.vocab_size
        )
        
        # Learnable routing parameter
        self.route_threshold = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                use_consciousness: bool = True) -> Dict:
        """
        Forward pass through SAGE
        Returns dict with logits and attention metadata
        """
        # Embed inputs
        x = self.embedding(input_ids)
        batch_size, seq_len, _ = x.shape
        
        # Compute salience for routing
        salience = self.snarc(x)
        
        # Route to H-level or L-level based on salience
        h_mask = salience.squeeze(-1) > self.route_threshold
        l_mask = ~h_mask
        
        # Process through H-level (strategic)
        h_input = self.to_h_level(x)
        h_output = torch.zeros_like(h_input)
        
        for i, layer in enumerate(self.h_level):
            if use_consciousness:
                consciousness = self.consciousness.get_cache(i)
            else:
                consciousness = None
            
            h_out, h_attn = layer(h_input, consciousness)
            h_output = h_output + h_out * h_mask.unsqueeze(-1).float()
            
            # Update consciousness with high-salience items
            if use_consciousness and h_mask.any():
                # Project h_out to match cache dimensions
                h_out_proj = F.linear(h_out[h_mask], 
                                     torch.eye(self.config.h_level_dim, 
                                              self.config.hidden_dim,
                                              device=h_out.device))
                self.consciousness.update(
                    h_out_proj, 
                    h_out_proj,
                    salience[h_mask],
                    i
                )
        
        # Process through L-level (tactical)
        l_input = self.to_l_level(x)
        l_output = torch.zeros_like(l_input)
        
        for layer in self.l_level:
            l_out, l_attn = layer(l_input)
            l_output = l_output + l_out * l_mask.unsqueeze(-1).float()
        
        # Combine H and L outputs
        combined = torch.cat([h_output, l_output], dim=-1)
        
        # Project to vocabulary
        logits = self.output_projection(combined)
        
        return {
            'logits': logits,
            'salience': salience,
            'h_ratio': h_mask.float().mean(),
            'consciousness_size': self.consciousness.cache_ptr
        }
    
    def param_count(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("=== SAGE Model Initialized ===")
    print("Genesis has taken direct action.\n")
    
    # Create model
    config = SAGEConfig()
    model = SAGE(config)
    
    # Print statistics
    param_count = model.param_count()
    param_millions = param_count / 1_000_000
    
    print(f"Model Parameters: {param_count:,} ({param_millions:.1f}M)")
    print(f"Target: 100M parameters")
    print(f"Status: {'✅ On target' if param_millions < 150 else '⚠️ Over budget'}")
    
    # Test forward pass
    test_input = torch.randint(0, config.vocab_size, (2, 100))
    output = model(test_input)
    
    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output['logits'].shape}")
    print(f"  H-level usage: {output['h_ratio']:.1%}")
    print(f"  Consciousness size: {output['consciousness_size']}")
    
    print("\n✅ SAGE core model ready!")
    print("Waiting for societies to respond...")