"""
SAGE Core Architecture

The 100M parameter attention orchestrator that coordinates between strategic (H) 
and tactical (L) reasoning modules. This implements the critical mass threshold
for reasoning emergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

from .sage_config import SAGEConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with proper scaling for 100M param model"""
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Scaled initialization for large model stability
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Apply attention to values
        context = torch.matmul(attention_probs, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer for H or L module"""
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # FFN with proper intermediate size for 100M params
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU() if config.hidden_act == "gelu" else nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class HModule(nn.Module):
    """High-level strategic reasoning module (~45M parameters)
    
    Processes context and makes strategic decisions about:
    - What deserves attention
    - What resources to employ
    - How to approach the problem
    """
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        
        # Deep transformer stack for strategic reasoning
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_h_layers)
        ])
        
        # Context processing
        self.context_encoder = nn.Linear(config.context_dim, config.hidden_size)
        self.context_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Strategic output head
        self.strategy_head = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input embeddings [batch, seq, hidden]
            context: Context vector [batch, context_dim]
            attention_mask: Attention mask [batch, seq]
            
        Returns:
            h_output: Strategic hidden states
            strategy: Strategic decision vector
        """
        # Encode and fuse context if provided
        if context is not None:
            context_hidden = self.context_encoder(context)
            # Broadcast context to all positions
            context_hidden = context_hidden.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            hidden_states = self.context_fusion(
                torch.cat([hidden_states, context_hidden], dim=-1)
            )
        
        # Process through transformer layers
        # Layers 1-2: Context encoding/translation
        for layer in self.layers[:2]:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Layers 3-5: Core strategic reasoning (where cognition emerges)
        for layer in self.layers[2:5]:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Layers 6-7: Strategy preparation/communication
        for layer in self.layers[5:]:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Generate strategy vector
        strategy = self.strategy_head(hidden_states.mean(dim=1))  # Pool over sequence
        
        return hidden_states, strategy


class LModule(nn.Module):
    """Low-level tactical execution module (~45M parameters)
    
    Executes tactical decisions based on strategic guidance:
    - Processes specific inputs
    - Generates concrete actions
    - Implements strategic plans
    """
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        
        # Deep transformer stack for tactical execution
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_l_layers)
        ])
        
        # Strategy integration
        self.strategy_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        # Tactical output head
        self.action_head = nn.Linear(config.hidden_size, config.num_classes)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        strategy: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input embeddings [batch, seq, hidden]
            strategy: Strategic guidance from H [batch, hidden]
            attention_mask: Attention mask [batch, seq]
            
        Returns:
            l_output: Tactical hidden states
            actions: Concrete action logits
        """
        # Integrate strategy with gating mechanism
        strategy_broadcast = strategy.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        gate = self.strategy_gate(torch.cat([hidden_states, strategy_broadcast], dim=-1))
        hidden_states = hidden_states * gate
        
        # Process through transformer layers
        # Layers 1-2: Input processing/translation
        for layer in self.layers[:2]:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Layers 3-5: Core tactical reasoning
        for layer in self.layers[2:5]:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Layers 6-7: Action generation/output
        for layer in self.layers[5:]:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Generate actions
        actions = self.action_head(hidden_states)
        
        return hidden_states, actions


class BidirectionalCommunication(nn.Module):
    """H↔L bidirectional communication layer (~10M parameters)
    
    Enables strategic-tactical dialogue:
    - H provides strategic context to L
    - L provides tactical feedback to H
    - Iterative refinement through cycles
    """
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        
        # H → L communication
        self.h_to_l = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # L → H feedback
        self.l_to_h = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Cross-attention for rich interaction
        self.cross_attention_h = MultiHeadAttention(config)
        self.cross_attention_l = MultiHeadAttention(config)
        
    def forward(
        self,
        h_states: torch.Tensor,
        l_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enable bidirectional communication between modules"""
        # H attends to L (strategic considers tactical feedback)
        h_refined = self.cross_attention_h(h_states)
        h_message = self.l_to_h(l_states.mean(dim=1))
        
        # L attends to H (tactical follows strategic guidance)
        l_refined = self.cross_attention_l(l_states)
        l_message = self.h_to_l(h_states.mean(dim=1))
        
        return h_refined + h_message.unsqueeze(1), l_refined + l_message.unsqueeze(1)


class SAGECore(nn.Module):
    """SAGE Core: 100M parameter attention orchestrator
    
    Combines H-module (strategic), L-module (tactical), and bidirectional
    communication to create an attention engine that orchestrates resources.
    """
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        
        # Input processing
        self.input_embedding = nn.Embedding(config.num_classes, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Core modules
        self.h_module = HModule(config)
        self.l_module = LModule(config)
        self.communication = BidirectionalCommunication(config)
        
        # Reasoning cycle control
        self.halt_predictor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Resource routing head
        self.resource_router = nn.Linear(
            config.hidden_size,
            len(config.resource_types)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for stable training at scale"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_cycles: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SAGE
        
        Args:
            input_ids: Input token IDs [batch, seq]
            context: Context vector [batch, context_dim]
            attention_mask: Attention mask [batch, seq]
            num_cycles: Number of reasoning cycles (default: config.num_reasoning_cycles)
            
        Returns:
            Dictionary containing:
            - output: Final action predictions [batch, seq, num_classes]
            - strategy: Strategic decisions [batch, hidden]
            - halt_probs: Halting probabilities per cycle
            - resource_allocation: Resource usage predictions
            - h_states: Final H-module states
            - l_states: Final L-module states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if num_cycles is None:
            num_cycles = self.config.num_reasoning_cycles
        
        # Generate embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.input_embedding(input_ids) + self.position_embedding(position_ids)
        
        # Initialize states
        h_states = embeddings
        l_states = embeddings
        halt_probs = []
        
        # Iterative reasoning cycles
        for cycle in range(num_cycles):
            # H-module strategic reasoning
            h_states, strategy = self.h_module(h_states, context, attention_mask)
            
            # L-module tactical execution
            l_states, actions = self.l_module(l_states, strategy, attention_mask)
            
            # Bidirectional communication
            h_states, l_states = self.communication(h_states, l_states)
            
            # Compute halting probability
            combined = torch.cat([
                h_states.mean(dim=1),
                l_states.mean(dim=1)
            ], dim=-1)
            halt_prob = self.halt_predictor(combined)
            halt_probs.append(halt_prob)
            
            # Early stopping if confident
            if halt_prob.mean() > 0.99:
                break
        
        # Resource allocation decision
        resource_allocation = self.resource_router(strategy)
        resource_allocation = F.softmax(resource_allocation, dim=-1)
        
        # Stack halt probabilities
        halt_probs = torch.stack(halt_probs, dim=1) if halt_probs else torch.zeros(batch_size, 1, 1)
        
        return {
            'output': actions,
            'strategy': strategy,
            'halt_probs': halt_probs,
            'resource_allocation': resource_allocation,
            'h_states': h_states,
            'l_states': l_states,
            'num_cycles_used': cycle + 1
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the architecture
    config = SAGEConfig()
    model = SAGECore(config)
    
    # Print model statistics
    total_params = model.get_num_params()
    trainable_params = model.get_num_trainable_params()
    
    print(f"SAGE Core Model Statistics:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    
    input_ids = torch.randint(0, config.num_classes, (batch_size, seq_len))
    context = torch.randn(batch_size, config.context_dim)
    
    print(f"\nTesting forward pass...")
    outputs = model(input_ids, context)
    
    print(f"  Output shape: {outputs['output'].shape}")
    print(f"  Strategy shape: {outputs['strategy'].shape}")
    print(f"  Cycles used: {outputs['num_cycles_used']}")
    print(f"  Resource allocation: {outputs['resource_allocation']}")
    print(f"\nForward pass successful!")