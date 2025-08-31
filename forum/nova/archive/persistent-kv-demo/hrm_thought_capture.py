#!/usr/bin/env python3
"""
HRM Thought Capture System
Captures and persists "consciousness states" from HRM's H and L loops
Integrates with SNARC memory selection and KV-cache persistence concepts
"""

import torch
import torch.nn as nn
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class HRMThoughtState:
    """Represents a captured thought state from HRM"""
    timestamp: str
    step: int
    
    # H-level (strategic/dreams) state
    h_state: torch.Tensor  # Shape: [batch, hidden_size]
    h_layer_states: List[torch.Tensor]  # States from each H layer
    h_attention_patterns: Optional[torch.Tensor]  # If we extract attention
    
    # L-level (tactical/practice) state  
    l_state: torch.Tensor  # Shape: [batch, hidden_size]
    l_layer_states: List[torch.Tensor]  # States from each L layer
    l_attention_patterns: Optional[torch.Tensor]  # If we extract attention
    
    # Interaction state
    h_l_interaction: torch.Tensor  # How H influenced L this step
    l_h_feedback: torch.Tensor  # How L influenced H this step
    
    # Q-values (consciousness of halting)
    q_halt: float
    q_continue: float
    halted: bool
    
    # Context
    input_embedding: torch.Tensor
    output_logits: torch.Tensor
    puzzle_context: Optional[Dict]
    
    # SNARC salience score
    snarc_score: Optional[float] = None
    
    def to_cpu(self):
        """Move all tensors to CPU for storage"""
        return HRMThoughtState(
            timestamp=self.timestamp,
            step=self.step,
            h_state=self.h_state.cpu(),
            h_layer_states=[s.cpu() for s in self.h_layer_states],
            h_attention_patterns=self.h_attention_patterns.cpu() if self.h_attention_patterns else None,
            l_state=self.l_state.cpu(),
            l_layer_states=[s.cpu() for s in self.l_layer_states],
            l_attention_patterns=self.l_attention_patterns.cpu() if self.l_attention_patterns else None,
            h_l_interaction=self.h_l_interaction.cpu(),
            l_h_feedback=self.l_h_feedback.cpu(),
            q_halt=self.q_halt,
            q_continue=self.q_continue,
            halted=self.halted,
            input_embedding=self.input_embedding.cpu(),
            output_logits=self.output_logits.cpu(),
            puzzle_context=self.puzzle_context,
            snarc_score=self.snarc_score
        )


class HRMThoughtCapture(nn.Module):
    """Captures thought states from HRM during forward passes"""
    
    def __init__(self, hrm_model, capture_attention=False):
        super().__init__()
        self.hrm = hrm_model
        self.capture_attention = capture_attention
        self.captured_states = []
        self.capture_enabled = False
        
        # Hook handles for layer capture
        self.h_hooks = []
        self.l_hooks = []
        
        # Storage for intermediate states
        self.h_layer_outputs = []
        self.l_layer_outputs = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate states"""
        
        # Hook into H-level layers
        for i, layer in enumerate(self.hrm.inner.H_level.layers):
            def make_h_hook(idx):
                def hook(module, input, output):
                    if self.capture_enabled:
                        self.h_layer_outputs.append(output.detach())
                return hook
            self.h_hooks.append(layer.register_forward_hook(make_h_hook(i)))
        
        # Hook into L-level layers
        for i, layer in enumerate(self.hrm.inner.L_level.layers):
            def make_l_hook(idx):
                def hook(module, input, output):
                    if self.capture_enabled:
                        self.l_layer_outputs.append(output.detach())
                return hook
            self.l_hooks.append(layer.register_forward_hook(make_l_hook(i)))
    
    def forward(self, carry, batch):
        """Forward pass with thought capture"""
        
        # Clear previous captures
        self.h_layer_outputs = []
        self.l_layer_outputs = []
        
        # Enable capture
        self.capture_enabled = True
        
        # Store initial states
        initial_h = carry.inner_carry.z_H.detach() if hasattr(carry.inner_carry, 'z_H') else None
        initial_l = carry.inner_carry.z_L.detach() if hasattr(carry.inner_carry, 'z_L') else None
        
        # Forward pass
        new_carry, outputs = self.hrm(carry, batch)
        
        # Capture the thought state
        if initial_h is not None and initial_l is not None:
            thought_state = self._create_thought_state(
                carry=carry,
                new_carry=new_carry,
                batch=batch,
                outputs=outputs,
                initial_h=initial_h,
                initial_l=initial_l
            )
            self.captured_states.append(thought_state)
        
        # Disable capture
        self.capture_enabled = False
        
        return new_carry, outputs
    
    def _create_thought_state(self, carry, new_carry, batch, outputs, initial_h, initial_l):
        """Create a thought state from current step"""
        
        # Calculate state changes (interactions)
        h_l_interaction = new_carry.inner_carry.z_L - initial_l
        l_h_feedback = new_carry.inner_carry.z_H - initial_h
        
        # Extract Q-values
        q_halt = outputs.get("q_halt_logits", torch.tensor(0.0)).mean().item()
        q_continue = outputs.get("q_continue_logits", torch.tensor(0.0)).mean().item()
        
        # Create thought state
        state = HRMThoughtState(
            timestamp=datetime.now().isoformat(),
            step=carry.steps.max().item() if hasattr(carry, 'steps') else 0,
            h_state=new_carry.inner_carry.z_H.detach(),
            h_layer_states=self.h_layer_outputs.copy(),
            h_attention_patterns=None,  # Would need attention extraction
            l_state=new_carry.inner_carry.z_L.detach(),
            l_layer_states=self.l_layer_outputs.copy(),
            l_attention_patterns=None,  # Would need attention extraction
            h_l_interaction=h_l_interaction,
            l_h_feedback=l_h_feedback,
            q_halt=q_halt,
            q_continue=q_continue,
            halted=carry.halted.any().item() if hasattr(carry, 'halted') else False,
            input_embedding=batch.get("inputs", torch.tensor(0)),
            output_logits=outputs.get("logits", torch.tensor(0)),
            puzzle_context=None  # Could extract puzzle identifiers
        )
        
        return state
    
    def clear_captures(self):
        """Clear captured states"""
        self.captured_states = []
    
    def save_captures(self, path, format="torch"):
        """Save captured states to disk"""
        cpu_states = [s.to_cpu() for s in self.captured_states]
        
        if format == "torch":
            torch.save(cpu_states, path)
        elif format == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(cpu_states, f)
        elif format == "gzip":
            with gzip.open(path, 'wb') as f:
                pickle.dump(cpu_states, f)
    
    def load_captures(self, path, format="torch"):
        """Load captured states from disk"""
        if format == "torch":
            self.captured_states = torch.load(path, map_location="cpu")
        elif format == "pickle":
            with open(path, 'rb') as f:
                self.captured_states = pickle.load(f)
        elif format == "gzip":
            with gzip.open(path, 'rb') as f:
                self.captured_states = pickle.load(f)


class SNARCThoughtSelector:
    """Selects which thought states to preserve using SNARC-like scoring"""
    
    def __init__(self, novelty_weight=0.3, surprise_weight=0.3, 
                 relevance_weight=0.2, consolidation_weight=0.2):
        self.weights = {
            'novelty': novelty_weight,
            'surprise': surprise_weight,
            'relevance': relevance_weight,
            'consolidation': consolidation_weight
        }
        self.state_history = []
        self.state_embeddings = []
    
    def score_thought(self, thought_state: HRMThoughtState) -> float:
        """Calculate SNARC score for a thought state"""
        
        scores = {}
        
        # Novelty: How different from previous states
        if self.state_history:
            prev_h = torch.stack([s.h_state for s in self.state_history[-10:]])
            novelty = torch.norm(thought_state.h_state - prev_h.mean(dim=0)).item()
            scores['novelty'] = min(1.0, novelty / 10.0)  # Normalize
        else:
            scores['novelty'] = 1.0
        
        # Surprise: Unexpected Q-values
        q_diff = abs(thought_state.q_halt - thought_state.q_continue)
        scores['surprise'] = 1.0 - np.exp(-q_diff)  # Higher difference = more surprise
        
        # Relevance: Output confidence
        if thought_state.output_logits is not None:
            output_probs = torch.softmax(thought_state.output_logits, dim=-1)
            entropy = -torch.sum(output_probs * torch.log(output_probs + 1e-10))
            scores['relevance'] = 1.0 - (entropy / np.log(output_probs.shape[-1]))  # Low entropy = high relevance
        else:
            scores['relevance'] = 0.5
        
        # Consolidation: Interaction strength between H and L
        h_l_strength = torch.norm(thought_state.h_l_interaction).item()
        l_h_strength = torch.norm(thought_state.l_h_feedback).item()
        scores['consolidation'] = min(1.0, (h_l_strength + l_h_strength) / 20.0)
        
        # Calculate weighted score
        snarc_score = sum(self.weights[k] * scores[k] for k in self.weights)
        
        # Store in thought state
        thought_state.snarc_score = snarc_score
        
        # Add to history
        self.state_history.append(thought_state)
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        return snarc_score
    
    def select_for_memory(self, thought_states: List[HRMThoughtState], 
                         threshold=0.5) -> List[HRMThoughtState]:
        """Select which thoughts to preserve in long-term memory"""
        
        selected = []
        for state in thought_states:
            score = self.score_thought(state)
            if score >= threshold:
                selected.append(state)
        
        return selected


class DualMemoryIntegration:
    """Integrates H and L thought captures with dual memory system"""
    
    def __init__(self, h_memory_size=1000, l_memory_size=5000):
        self.h_memory = []  # Strategic/dream memory
        self.l_memory = []  # Tactical/practice memory
        self.h_memory_size = h_memory_size
        self.l_memory_size = l_memory_size
        
        self.snarc_selector = SNARCThoughtSelector()
    
    def process_thought(self, thought_state: HRMThoughtState):
        """Process a thought state into appropriate memory"""
        
        # Score the thought
        snarc_score = self.snarc_selector.score_thought(thought_state)
        
        # Determine which memory system
        if thought_state.halted:
            # Halted states go to H-memory (strategic decisions)
            self._add_to_h_memory(thought_state)
        else:
            # Continuing states go to L-memory (tactical execution)
            self._add_to_l_memory(thought_state)
        
        # High-score thoughts go to both
        if snarc_score > 0.7:
            self._add_to_h_memory(thought_state)
            self._add_to_l_memory(thought_state)
    
    def _add_to_h_memory(self, thought_state: HRMThoughtState):
        """Add to strategic/dream memory with consolidation"""
        self.h_memory.append(thought_state)
        
        # Consolidate if over size limit
        if len(self.h_memory) > self.h_memory_size:
            # Keep only high-score memories
            self.h_memory.sort(key=lambda x: x.snarc_score or 0, reverse=True)
            self.h_memory = self.h_memory[:self.h_memory_size]
    
    def _add_to_l_memory(self, thought_state: HRMThoughtState):
        """Add to tactical/practice memory"""
        self.l_memory.append(thought_state)
        
        # FIFO for L-memory (practice patterns)
        if len(self.l_memory) > self.l_memory_size:
            self.l_memory.pop(0)
    
    def get_relevant_memories(self, current_state: HRMThoughtState, k=5):
        """Retrieve relevant memories for current context"""
        
        h_memories = self._find_similar_states(current_state, self.h_memory, k)
        l_memories = self._find_similar_states(current_state, self.l_memory, k)
        
        return {
            'strategic': h_memories,
            'tactical': l_memories
        }
    
    def _find_similar_states(self, query: HRMThoughtState, 
                            memory: List[HRMThoughtState], k=5):
        """Find k most similar states in memory"""
        
        if not memory:
            return []
        
        # Simple cosine similarity on H-states
        query_h = query.h_state.flatten()
        similarities = []
        
        for mem_state in memory:
            mem_h = mem_state.h_state.flatten()
            sim = torch.nn.functional.cosine_similarity(
                query_h.unsqueeze(0),
                mem_h.unsqueeze(0)
            ).item()
            similarities.append((sim, mem_state))
        
        # Sort and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [state for _, state in similarities[:k]]
    
    def consolidate_dreams(self):
        """Consolidate H-memory through 'dreaming' - finding patterns"""
        
        if len(self.h_memory) < 10:
            return
        
        # Cluster similar strategic decisions
        h_states = torch.stack([s.h_state for s in self.h_memory])
        
        # Simple clustering - find mean state
        mean_state = h_states.mean(dim=0)
        
        # Find prototypical examples
        distances = torch.norm(h_states - mean_state, dim=-1)
        prototype_idx = distances.argmin()
        
        # Create consolidated memory
        consolidated = HRMThoughtState(
            timestamp=datetime.now().isoformat(),
            step=-1,  # Special marker for consolidated memory
            h_state=mean_state,
            h_layer_states=[],
            h_attention_patterns=None,
            l_state=self.h_memory[prototype_idx].l_state,
            l_layer_states=[],
            l_attention_patterns=None,
            h_l_interaction=torch.zeros_like(mean_state),
            l_h_feedback=torch.zeros_like(mean_state),
            q_halt=np.mean([s.q_halt for s in self.h_memory]),
            q_continue=np.mean([s.q_continue for s in self.h_memory]),
            halted=True,
            input_embedding=torch.zeros_like(self.h_memory[0].input_embedding),
            output_logits=torch.zeros_like(self.h_memory[0].output_logits),
            puzzle_context={'type': 'consolidated_dream'},
            snarc_score=1.0  # High score for consolidated memories
        )
        
        # Add to H-memory
        self.h_memory.insert(0, consolidated)


def demonstrate_hrm_thought_capture():
    """Demonstration of HRM thought capture system"""
    
    print("=" * 60)
    print("HRM THOUGHT CAPTURE DEMONSTRATION")
    print("=" * 60)
    
    # This would normally load an actual HRM model
    # For demo, we'll create mock states
    
    # Create mock thought states
    mock_states = []
    for i in range(10):
        state = HRMThoughtState(
            timestamp=datetime.now().isoformat(),
            step=i,
            h_state=torch.randn(1, 512),  # Mock H state
            h_layer_states=[torch.randn(1, 512) for _ in range(3)],
            h_attention_patterns=None,
            l_state=torch.randn(1, 512),  # Mock L state
            l_layer_states=[torch.randn(1, 512) for _ in range(3)],
            l_attention_patterns=None,
            h_l_interaction=torch.randn(1, 512) * 0.1,
            l_h_feedback=torch.randn(1, 512) * 0.1,
            q_halt=np.random.random(),
            q_continue=np.random.random(),
            halted=(i % 3 == 0),  # Halt every 3 steps
            input_embedding=torch.randn(1, 100),
            output_logits=torch.randn(1, 1000),
            puzzle_context={'puzzle_id': i % 3}
        )
        mock_states.append(state)
    
    # Test SNARC selection
    print("\n1. SNARC Selection:")
    selector = SNARCThoughtSelector()
    selected = selector.select_for_memory(mock_states, threshold=0.4)
    print(f"   Selected {len(selected)}/{len(mock_states)} states for memory")
    
    # Test dual memory integration
    print("\n2. Dual Memory Integration:")
    dual_memory = DualMemoryIntegration()
    
    for state in mock_states:
        dual_memory.process_thought(state)
    
    print(f"   H-memory (strategic): {len(dual_memory.h_memory)} states")
    print(f"   L-memory (tactical): {len(dual_memory.l_memory)} states")
    
    # Test memory retrieval
    print("\n3. Memory Retrieval:")
    query_state = mock_states[5]
    relevant = dual_memory.get_relevant_memories(query_state, k=3)
    print(f"   Found {len(relevant['strategic'])} strategic memories")
    print(f"   Found {len(relevant['tactical'])} tactical memories")
    
    # Test consolidation
    print("\n4. Dream Consolidation:")
    dual_memory.consolidate_dreams()
    consolidated = [s for s in dual_memory.h_memory if s.step == -1]
    print(f"   Created {len(consolidated)} consolidated dream memories")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_hrm_thought_capture()