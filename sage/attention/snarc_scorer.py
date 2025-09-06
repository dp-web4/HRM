"""
SNARC Scorer Module

Implements the SNARC (Surprise, Novelty, Arousal, Reward, Conflict) scoring system
for intelligent attention allocation in SAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque


class SNARCScorer(nn.Module):
    """SNARC scoring system for attention prioritization
    
    Each component evaluates different aspects of salience:
    - Surprise: Deviation from expected patterns
    - Novelty: Presence of unseen patterns
    - Arousal: Complexity and information density
    - Reward: Task completion and success signals
    - Conflict: Ambiguity and uncertainty
    """
    
    def __init__(self, hidden_size: int = 768, memory_size: int = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Memory bank for novelty assessment
        self.memory_bank = deque(maxlen=memory_size)
        self.pattern_cache = {}
        
        # Learnable components for each SNARC dimension
        self.surprise_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.novelty_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.arousal_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.conflict_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Prediction network for surprise computation
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Attention weighting network
        self.attention_weight = nn.Sequential(
            nn.Linear(5, 16),  # 5 SNARC dimensions
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def compute_surprise(self, input_states: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute surprise as deviation from predicted patterns
        
        Args:
            input_states: Current input states [batch, seq, hidden]
            context: Optional context for prediction [batch, hidden]
            
        Returns:
            surprise_scores: Surprise scores [batch, seq, 1]
        """
        batch_size, seq_len, _ = input_states.shape
        
        # Generate predictions based on context or prior states
        if seq_len > 1:
            # Use previous states to predict current
            predictions = self.predictor(input_states[:, :-1])
            actuals = input_states[:, 1:]
            
            # Compute prediction error as surprise
            surprise = F.mse_loss(predictions, actuals, reduction='none').mean(dim=-1, keepdim=True)
            
            # Pad for first position (no prior)
            first_surprise = torch.ones(batch_size, 1, 1, device=input_states.device) * 0.5
            surprise = torch.cat([first_surprise, surprise], dim=1)
        else:
            # Single position - use learned surprise
            surprise = self.surprise_net(input_states)
        
        return surprise
    
    def compute_novelty(self, input_states: torch.Tensor) -> torch.Tensor:
        """Compute novelty by comparing to memory bank
        
        Args:
            input_states: Current input states [batch, seq, hidden]
            
        Returns:
            novelty_scores: Novelty scores [batch, seq, 1]
        """
        batch_size, seq_len, hidden_size = input_states.shape
        device = input_states.device
        
        if len(self.memory_bank) > 0:
            # Convert memory bank to tensor
            memory_tensor = torch.stack(list(self.memory_bank)).to(device)
            memory_tensor = memory_tensor.view(-1, hidden_size)
            
            # Compute similarity to memory
            input_flat = input_states.view(-1, hidden_size)
            similarities = F.cosine_similarity(
                input_flat.unsqueeze(1),
                memory_tensor.unsqueeze(0),
                dim=-1
            )
            
            # Novelty is inverse of maximum similarity
            max_similarity = similarities.max(dim=-1)[0]
            novelty = 1.0 - max_similarity
            novelty = novelty.view(batch_size, seq_len, 1)
        else:
            # Empty memory - everything is novel
            novelty = torch.ones(batch_size, seq_len, 1, device=device)
        
        # Refine with learned network
        memory_context = torch.zeros_like(input_states) if len(self.memory_bank) == 0 else input_states
        combined = torch.cat([input_states, memory_context], dim=-1)
        novelty_refined = self.novelty_net(combined)
        
        return novelty * novelty_refined
    
    def compute_arousal(self, input_states: torch.Tensor) -> torch.Tensor:
        """Compute arousal as complexity/information density
        
        Args:
            input_states: Current input states [batch, seq, hidden]
            
        Returns:
            arousal_scores: Arousal scores [batch, seq, 1]
        """
        # Compute entropy as proxy for complexity
        # Higher entropy = higher arousal
        probabilities = F.softmax(input_states, dim=-1)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1, keepdim=True)
        
        # Normalize entropy
        max_entropy = torch.log(torch.tensor(input_states.size(-1), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        # Refine with learned network
        arousal = self.arousal_net(input_states)
        
        return normalized_entropy * arousal
    
    def compute_reward(self, task_success: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute reward signal from task completion
        
        Args:
            task_success: Optional success signal [batch, 1]
            device: Device to create tensor on
            
        Returns:
            reward_scores: Reward scores [batch, 1]
        """
        if task_success is not None:
            return task_success
        else:
            # Default: no reward signal
            if device is None:
                device = torch.device('cpu')
            return torch.zeros(1, 1, device=device)
    
    def compute_conflict(self, input_states: torch.Tensor) -> torch.Tensor:
        """Compute conflict as ambiguity/uncertainty in patterns
        
        Args:
            input_states: Current input states [batch, seq, hidden]
            
        Returns:
            conflict_scores: Conflict scores [batch, seq, 1]
        """
        # Compute variance across hidden dimensions as uncertainty
        variance = input_states.var(dim=-1, keepdim=True)
        
        # High variance = high conflict
        normalized_variance = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
        
        # Refine with learned network
        conflict = self.conflict_net(input_states)
        
        return normalized_variance * conflict
    
    def forward(
        self,
        input_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        task_success: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute full SNARC scores
        
        Args:
            input_states: Input hidden states [batch, seq, hidden]
            context: Optional context vector [batch, hidden]
            task_success: Optional task success signal [batch, 1]
            return_components: Whether to return individual components
            
        Returns:
            Dictionary containing:
            - snarc_scores: Combined SNARC scores [batch, seq, 1]
            - attention_weights: Attention importance weights [batch, seq, 1]
            - (optional) individual component scores
        """
        # Compute individual SNARC components
        surprise = self.compute_surprise(input_states, context)
        novelty = self.compute_novelty(input_states)
        arousal = self.compute_arousal(input_states)
        conflict = self.compute_conflict(input_states)
        
        # Reward is sequence-independent
        batch_size = input_states.size(0)
        seq_len = input_states.size(1)
        reward = self.compute_reward(task_success, device=input_states.device)
        reward = reward.expand(batch_size, seq_len, 1)
        
        # Stack SNARC components
        snarc_stack = torch.stack([surprise, novelty, arousal, reward, conflict], dim=-1).squeeze(-2)
        
        # Compute attention weights based on SNARC
        attention_weights = self.attention_weight(snarc_stack)
        
        # Combined SNARC score (weighted average)
        weights = F.softmax(torch.tensor([1.0, 0.8, 0.6, 1.2, 0.7]), dim=0).to(input_states.device)
        snarc_scores = (snarc_stack * weights).mean(dim=-1, keepdim=True)
        
        # Update memory bank with current states
        self.update_memory(input_states)
        
        result = {
            'snarc_scores': snarc_scores,
            'attention_weights': attention_weights
        }
        
        if return_components:
            result.update({
                'surprise': surprise,
                'novelty': novelty,
                'arousal': arousal,
                'reward': reward,
                'conflict': conflict
            })
        
        return result
    
    def update_memory(self, states: torch.Tensor):
        """Update memory bank with new experiences
        
        Args:
            states: States to add to memory [batch, seq, hidden]
        """
        # Flatten and add to memory
        states_flat = states.detach().cpu().view(-1, self.hidden_size)
        
        # Sample to avoid memory explosion
        if states_flat.size(0) > 10:
            indices = torch.randperm(states_flat.size(0))[:10]
            states_flat = states_flat[indices]
        
        for state in states_flat:
            self.memory_bank.append(state)
    
    def bias_attention(
        self,
        attention_scores: torch.Tensor,
        snarc_weights: torch.Tensor,
        bias_strength: float = 0.5
    ) -> torch.Tensor:
        """Bias attention scores based on SNARC weights
        
        Args:
            attention_scores: Raw attention scores [batch, heads, seq, seq]
            snarc_weights: SNARC-based importance [batch, seq, 1]
            bias_strength: How strongly to bias (0=no bias, 1=full bias)
            
        Returns:
            biased_attention: Modified attention scores
        """
        # Expand SNARC weights to match attention dimensions
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        snarc_weights = snarc_weights.squeeze(-1).unsqueeze(1).unsqueeze(2)
        snarc_weights = snarc_weights.expand(batch_size, num_heads, 1, seq_len)
        
        # Apply bias
        bias = snarc_weights * bias_strength
        biased_attention = attention_scores + bias
        
        return biased_attention
    
    def get_top_k_salient(
        self,
        snarc_scores: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k most salient positions
        
        Args:
            snarc_scores: SNARC scores [batch, seq, 1]
            k: Number of top positions to return
            
        Returns:
            top_indices: Indices of top-k positions [batch, k]
            top_scores: Scores of top-k positions [batch, k]
        """
        snarc_flat = snarc_scores.squeeze(-1)
        top_scores, top_indices = torch.topk(snarc_flat, k=min(k, snarc_flat.size(1)), dim=1)
        
        return top_indices, top_scores


class AttentionRouter(nn.Module):
    """Routes attention based on SNARC scores and resource availability"""
    
    def __init__(self, hidden_size: int = 768, num_resources: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_resources = num_resources
        
        # Routing network
        self.router = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),  # Input + SNARC
            nn.ReLU(),
            nn.Linear(hidden_size, num_resources),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        input_states: torch.Tensor,
        snarc_scores: torch.Tensor
    ) -> torch.Tensor:
        """Route inputs to appropriate resources
        
        Args:
            input_states: Input states [batch, hidden]
            snarc_scores: SNARC scores [batch, 5]
            
        Returns:
            resource_allocation: Resource usage probabilities [batch, num_resources]
        """
        combined = torch.cat([input_states, snarc_scores], dim=-1)
        return self.router(combined)


if __name__ == "__main__":
    # Test SNARC scorer
    print("Testing SNARC Scorer...")
    
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    
    scorer = SNARCScorer(hidden_size=hidden_size)
    
    # Create test input
    input_states = torch.randn(batch_size, seq_len, hidden_size)
    context = torch.randn(batch_size, hidden_size)
    
    # Compute SNARC scores
    results = scorer(input_states, context, return_components=True)
    
    print(f"SNARC scores shape: {results['snarc_scores'].shape}")
    print(f"Attention weights shape: {results['attention_weights'].shape}")
    print(f"\nComponent scores:")
    print(f"  Surprise: {results['surprise'].mean():.3f}")
    print(f"  Novelty: {results['novelty'].mean():.3f}")
    print(f"  Arousal: {results['arousal'].mean():.3f}")
    print(f"  Reward: {results['reward'].mean():.3f}")
    print(f"  Conflict: {results['conflict'].mean():.3f}")
    
    # Test attention biasing
    attention_scores = torch.randn(batch_size, 12, seq_len, seq_len)
    biased_attention = scorer.bias_attention(attention_scores, results['attention_weights'])
    print(f"\nBiased attention shape: {biased_attention.shape}")
    
    # Test top-k selection
    top_indices, top_scores = scorer.get_top_k_salient(results['snarc_scores'], k=3)
    print(f"\nTop-3 salient positions: {top_indices}")
    print(f"Top-3 scores: {top_scores}")
    
    print("\nSNARC Scorer test successful!")