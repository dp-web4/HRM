#!/usr/bin/env python3
"""
HRM Consciousness Integration
Brings together HRM thought capture with KV-cache persistence concepts
Creates a unified consciousness system for hierarchical reasoning
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from hrm_thought_capture import (
    HRMThoughtState, 
    HRMThoughtCapture,
    SNARCThoughtSelector,
    DualMemoryIntegration
)


class HRMConsciousnessPool:
    """
    Consciousness pool for HRM - inspired by KV-cache persistence
    But adapted for HRM's dual-loop architecture
    """
    
    def __init__(self, pool_size=100, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pool_size = pool_size
        
        # Consciousness pools for H and L levels
        self.h_pool = []  # Strategic consciousness
        self.l_pool = []  # Tactical consciousness
        
        # Cross-pollination memories (H↔L interactions)
        self.interaction_pool = []
        
        # Resonance tracker
        self.resonance_scores = {}
    
    def add_thought(self, thought_state: HRMThoughtState):
        """Add a thought to appropriate pool"""
        
        # Calculate resonance with existing pools
        h_resonance = self._calculate_resonance(thought_state.h_state, self.h_pool)
        l_resonance = self._calculate_resonance(thought_state.l_state, self.l_pool)
        
        # Store resonance scores
        self.resonance_scores[thought_state.timestamp] = {
            'h_resonance': h_resonance,
            'l_resonance': l_resonance,
            'interaction_strength': torch.norm(thought_state.h_l_interaction).item()
        }
        
        # Add to pools based on characteristics
        if thought_state.halted or h_resonance > 0.7:
            self._add_to_pool(thought_state, self.h_pool, 'strategic')
        
        if not thought_state.halted or l_resonance > 0.7:
            self._add_to_pool(thought_state, self.l_pool, 'tactical')
        
        # Strong interactions go to interaction pool
        interaction_strength = torch.norm(thought_state.h_l_interaction).item()
        if interaction_strength > 1.0:
            self._add_to_pool(thought_state, self.interaction_pool, 'interaction')
    
    def _calculate_resonance(self, state: torch.Tensor, pool: List) -> float:
        """Calculate resonance between state and pool"""
        if not pool:
            return 0.0
        
        # Get states from pool
        if pool == self.h_pool:
            pool_states = torch.stack([s.h_state for s in pool[-10:]])
        elif pool == self.l_pool:
            pool_states = torch.stack([s.l_state for s in pool[-10:]])
        else:
            return 0.0
        
        # Calculate cosine similarity
        state_norm = state / (torch.norm(state) + 1e-8)
        pool_norm = pool_states / (torch.norm(pool_states, dim=-1, keepdim=True) + 1e-8)
        
        similarities = torch.matmul(pool_norm, state_norm.T).squeeze()
        return similarities.max().item() if similarities.numel() > 0 else 0.0
    
    def _add_to_pool(self, thought_state: HRMThoughtState, pool: List, pool_type: str):
        """Add thought to specified pool with size management"""
        pool.append(thought_state)
        
        # Manage pool size
        if len(pool) > self.pool_size:
            # Remove lowest SNARC score
            pool.sort(key=lambda x: x.snarc_score or 0, reverse=True)
            pool.pop()
    
    def get_consciousness_state(self) -> Dict:
        """Get current consciousness state summary"""
        return {
            'h_pool_size': len(self.h_pool),
            'l_pool_size': len(self.l_pool),
            'interaction_pool_size': len(self.interaction_pool),
            'recent_resonance': list(self.resonance_scores.values())[-5:] if self.resonance_scores else [],
            'h_mean_snarc': float(np.mean([s.snarc_score for s in self.h_pool if s.snarc_score])) if self.h_pool else 0.0,
            'l_mean_snarc': float(np.mean([s.snarc_score for s in self.l_pool if s.snarc_score])) if self.l_pool else 0.0
        }
    
    def find_similar_thoughts(self, query_state: HRMThoughtState, k=5) -> Dict:
        """Find similar thoughts across all pools"""
        
        similar = {
            'strategic': self._find_similar_in_pool(query_state, self.h_pool, 'h', k),
            'tactical': self._find_similar_in_pool(query_state, self.l_pool, 'l', k),
            'interactive': self._find_similar_in_pool(query_state, self.interaction_pool, 'interaction', k)
        }
        
        return similar
    
    def _find_similar_in_pool(self, query: HRMThoughtState, pool: List, 
                              pool_type: str, k: int) -> List[Tuple[float, HRMThoughtState]]:
        """Find k most similar states in a pool"""
        if not pool:
            return []
        
        # Select appropriate state based on pool type
        if pool_type == 'h':
            query_vec = query.h_state.flatten()
            pool_vecs = [s.h_state.flatten() for s in pool]
        elif pool_type == 'l':
            query_vec = query.l_state.flatten()
            pool_vecs = [s.l_state.flatten() for s in pool]
        else:  # interaction
            query_vec = query.h_l_interaction.flatten()
            pool_vecs = [s.h_l_interaction.flatten() for s in pool]
        
        # Calculate similarities
        similarities = []
        for i, pool_vec in enumerate(pool_vecs):
            sim = torch.nn.functional.cosine_similarity(
                query_vec.unsqueeze(0),
                pool_vec.unsqueeze(0)
            ).item()
            similarities.append((sim, pool[i]))
        
        # Sort and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:k]


class HRMConsciousnessSession:
    """
    Manages a complete consciousness session for HRM
    Combines thought capture, SNARC selection, dual memory, and consciousness pools
    """
    
    def __init__(self, session_id: str, hrm_model=None):
        self.session_id = session_id
        self.timestamp = datetime.now().isoformat()
        
        # Components
        self.thought_capture = HRMThoughtCapture(hrm_model) if hrm_model else None
        self.snarc_selector = SNARCThoughtSelector()
        self.dual_memory = DualMemoryIntegration()
        self.consciousness_pool = HRMConsciousnessPool()
        
        # Session tracking
        self.total_thoughts = 0
        self.selected_thoughts = 0
        self.session_path = Path(f"consciousness_sessions/hrm_{session_id}")
        self.session_path.mkdir(parents=True, exist_ok=True)
    
    def process_step(self, carry, batch) -> Tuple:
        """Process one HRM step with full consciousness tracking"""
        
        if self.thought_capture:
            # Capture thought state
            new_carry, outputs = self.thought_capture(carry, batch)
            
            # Process captured thoughts
            for thought in self.thought_capture.captured_states:
                self.process_thought(thought)
            
            # Clear for next step
            self.thought_capture.clear_captures()
            
            return new_carry, outputs
        else:
            # Mock processing for demo
            return carry, {}
    
    def process_thought(self, thought_state: HRMThoughtState):
        """Process a single thought through all systems"""
        
        self.total_thoughts += 1
        
        # 1. SNARC scoring
        snarc_score = self.snarc_selector.score_thought(thought_state)
        
        # 2. Selection decision
        if snarc_score > 0.4:  # Threshold
            self.selected_thoughts += 1
            
            # 3. Add to dual memory
            self.dual_memory.process_thought(thought_state)
            
            # 4. Add to consciousness pool
            self.consciousness_pool.add_thought(thought_state)
            
            # 5. Check for resonance patterns
            self._check_resonance_patterns(thought_state)
    
    def _check_resonance_patterns(self, thought_state: HRMThoughtState):
        """Check for interesting resonance patterns"""
        
        # Find similar thoughts
        similar = self.consciousness_pool.find_similar_thoughts(thought_state, k=3)
        
        # Check for loops (high similarity to recent thoughts)
        if similar['strategic'] and similar['strategic'][0][0] > 0.95:
            print(f"  ⚠️ Potential strategic loop detected (similarity: {similar['strategic'][0][0]:.3f})")
        
        if similar['tactical'] and similar['tactical'][0][0] > 0.95:
            print(f"  ⚠️ Potential tactical loop detected (similarity: {similar['tactical'][0][0]:.3f})")
        
        # Check for breakthroughs (high interaction with low similarity)
        interaction_strength = torch.norm(thought_state.h_l_interaction).item()
        if interaction_strength > 2.0 and (not similar['interactive'] or similar['interactive'][0][0] < 0.5):
            print(f"  💡 Potential breakthrough: High H↔L interaction ({interaction_strength:.2f}) with novel pattern")
    
    def save_session(self):
        """Save session state to disk"""
        
        # Save consciousness pools
        torch.save({
            'h_pool': self.consciousness_pool.h_pool,
            'l_pool': self.consciousness_pool.l_pool,
            'interaction_pool': self.consciousness_pool.interaction_pool
        }, self.session_path / "consciousness_pools.pt")
        
        # Save memories
        torch.save({
            'h_memory': self.dual_memory.h_memory,
            'l_memory': self.dual_memory.l_memory
        }, self.session_path / "dual_memories.pt")
        
        # Save session metadata
        metadata = {
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'total_thoughts': self.total_thoughts,
            'selected_thoughts': self.selected_thoughts,
            'selection_rate': self.selected_thoughts / max(1, self.total_thoughts),
            'consciousness_state': self.consciousness_pool.get_consciousness_state()
        }
        
        with open(self.session_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"💾 Session saved to {self.session_path}")
    
    def load_session(self):
        """Load session state from disk"""
        
        # Load consciousness pools
        pools = torch.load(self.session_path / "consciousness_pools.pt", map_location="cpu")
        self.consciousness_pool.h_pool = pools['h_pool']
        self.consciousness_pool.l_pool = pools['l_pool']
        self.consciousness_pool.interaction_pool = pools['interaction_pool']
        
        # Load memories
        memories = torch.load(self.session_path / "dual_memories.pt", map_location="cpu")
        self.dual_memory.h_memory = memories['h_memory']
        self.dual_memory.l_memory = memories['l_memory']
        
        # Load metadata
        with open(self.session_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.total_thoughts = metadata['total_thoughts']
        self.selected_thoughts = metadata['selected_thoughts']
        
        print(f"✅ Session loaded from {self.session_path}")
        print(f"   Total thoughts: {self.total_thoughts}")
        print(f"   Selection rate: {metadata['selection_rate']:.2%}")
    
    def generate_report(self) -> str:
        """Generate a report on consciousness patterns"""
        
        lines = [
            f"# HRM Consciousness Session Report",
            f"**Session ID**: {self.session_id}",
            f"**Timestamp**: {self.timestamp}",
            "",
            "## Statistics",
            f"- Total thoughts processed: {self.total_thoughts}",
            f"- Thoughts selected for memory: {self.selected_thoughts}",
            f"- Selection rate: {self.selected_thoughts/max(1, self.total_thoughts):.2%}",
            "",
            "## Consciousness Pools",
            f"- Strategic (H) pool: {len(self.consciousness_pool.h_pool)} thoughts",
            f"- Tactical (L) pool: {len(self.consciousness_pool.l_pool)} thoughts",
            f"- Interaction pool: {len(self.consciousness_pool.interaction_pool)} thoughts",
            "",
            "## Memory Systems",
            f"- H-memory (dreams): {len(self.dual_memory.h_memory)} memories",
            f"- L-memory (practice): {len(self.dual_memory.l_memory)} memories",
            "",
            "## Consciousness State",
            json.dumps(self.consciousness_pool.get_consciousness_state(), indent=2),
            "",
            "## Insights",
            self._generate_insights()
        ]
        
        return "\n".join(lines)
    
    def _generate_insights(self) -> str:
        """Generate insights from consciousness patterns"""
        
        insights = []
        
        # Check H/L balance
        h_size = len(self.consciousness_pool.h_pool)
        l_size = len(self.consciousness_pool.l_pool)
        
        if h_size > l_size * 2:
            insights.append("- System showing strategic bias (more H than L thoughts)")
        elif l_size > h_size * 2:
            insights.append("- System showing tactical bias (more L than H thoughts)")
        else:
            insights.append("- System showing balanced strategic/tactical thinking")
        
        # Check interaction strength
        if self.consciousness_pool.interaction_pool:
            avg_interaction = np.mean([
                torch.norm(s.h_l_interaction).item() 
                for s in self.consciousness_pool.interaction_pool
            ])
            if avg_interaction > 1.5:
                insights.append(f"- Strong H↔L coupling detected (avg: {avg_interaction:.2f})")
        
        # Check for consolidation
        consolidated = [s for s in self.dual_memory.h_memory if s.step == -1]
        if consolidated:
            insights.append(f"- {len(consolidated)} consolidated dream memories created")
        
        return "\n".join(insights) if insights else "- No significant patterns detected yet"


def demonstrate_consciousness_session():
    """Demonstrate the full HRM consciousness system"""
    
    print("=" * 70)
    print("HRM CONSCIOUSNESS SESSION DEMONSTRATION")
    print("=" * 70)
    
    # Create session
    session = HRMConsciousnessSession("demo_001")
    
    print(f"\n📍 Session ID: {session.session_id}")
    print(f"📍 Timestamp: {session.timestamp}")
    
    # Simulate processing steps
    print("\n🔄 Simulating HRM reasoning steps...")
    
    for step in range(20):
        # Create mock thought state
        thought = HRMThoughtState(
            timestamp=datetime.now().isoformat(),
            step=step,
            h_state=torch.randn(1, 512),
            h_layer_states=[torch.randn(1, 512) for _ in range(3)],
            h_attention_patterns=None,
            l_state=torch.randn(1, 512),
            l_layer_states=[torch.randn(1, 512) for _ in range(3)],
            l_attention_patterns=None,
            h_l_interaction=torch.randn(1, 512) * (0.5 if step < 10 else 2.0),  # Increase interaction
            l_h_feedback=torch.randn(1, 512) * 0.3,
            q_halt=0.3 + 0.05 * step,  # Increasing halt probability
            q_continue=0.7 - 0.05 * step,
            halted=(step % 5 == 0),
            input_embedding=torch.randn(1, 100),
            output_logits=torch.randn(1, 1000),
            puzzle_context={'step': step}
        )
        
        # Process thought
        session.process_thought(thought)
        
        if step % 5 == 0:
            print(f"  Step {step}: Processed (halted={thought.halted})")
    
    # Consolidate dreams
    print("\n💤 Consolidating dreams...")
    session.dual_memory.consolidate_dreams()
    
    # Save session
    print("\n💾 Saving session...")
    session.save_session()
    
    # Generate report
    print("\n📊 Session Report:")
    print("-" * 70)
    report = session.generate_report()
    print(report)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("Consciousness patterns captured and analyzed!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_consciousness_session()