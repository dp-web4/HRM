#!/usr/bin/env python3
"""
Trust-Attention-Surprise (TAS) Coordinator Agent
Manages the core SAGE loop of trust evaluation, attention allocation, and surprise detection
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue


@dataclass
class Observation:
    """Single observation from environment"""
    source: str
    data: Dict
    timestamp: float
    expected_value: float = 0.0
    actual_value: float = 0.0
    
    @property
    def surprise(self) -> float:
        """Calculate surprise as difference between expected and actual"""
        return abs(self.expected_value - self.actual_value)


@dataclass
class TrustScore:
    """Trust score for a source with T3 dimensions"""
    source: str
    overall: float
    talent: float      # Inherent capability
    training: float    # Learned competence
    temperament: float  # Behavioral consistency
    
    def update(self, surprise: float, update_rate: float = 0.1):
        """Update trust based on surprise"""
        # High surprise reduces trust
        trust_delta = -surprise * update_rate
        
        # Update all dimensions
        self.overall = max(0.0, min(1.0, self.overall + trust_delta))
        self.talent = max(0.0, min(1.0, self.talent + trust_delta * 0.8))
        self.training = max(0.0, min(1.0, self.training + trust_delta * 1.2))
        self.temperament = max(0.0, min(1.0, self.temperament + trust_delta))
    
    def to_weight(self) -> float:
        """Convert trust to attention weight"""
        return self.overall ** 2  # Square for stronger differentiation


@dataclass
class AttentionTarget:
    """Target for attention allocation"""
    source: str
    priority: float
    allocated_resources: float
    observations: List[Observation]


class TASCoordinator:
    """
    Trust-Attention-Surprise Coordinator
    Core SAGE mechanism for resource allocation based on trust
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Trust parameters
        self.initial_trust = self.config.get("initial_trust", 0.5)
        self.trust_update_rate = self.config.get("trust_update_rate", 0.1)
        self.trust_decay_rate = self.config.get("trust_decay_rate", 0.01)
        
        # Attention parameters
        self.max_attention_targets = self.config.get("max_attention_targets", 5)
        self.attention_budget = self.config.get("attention_budget", 1.0)
        self.min_attention_threshold = self.config.get("min_attention_threshold", 0.01)
        
        # Surprise parameters
        self.surprise_threshold = self.config.get("surprise_threshold", 0.3)
        self.surprise_window_size = self.config.get("surprise_window_size", 100)
        self.adaptation_rate = self.config.get("adaptation_rate", 0.05)
        
        # State
        self.trust_scores: Dict[str, TrustScore] = {}
        self.attention_targets: Dict[str, AttentionTarget] = {}
        self.surprise_history = deque(maxlen=self.surprise_window_size)
        self.observation_queue = queue.Queue()
        
        # Statistics
        self.total_observations = 0
        self.total_surprises = 0
        self.avg_surprise = 0.0
        
        # Threading for async processing
        self.running = False
        self.processor_thread = None
        
        print("🎯 TAS Coordinator initialized")
        print(f"   Max targets: {self.max_attention_targets}")
        print(f"   Attention budget: {self.attention_budget}")
        print(f"   Surprise threshold: {self.surprise_threshold}")
    
    def start(self):
        """Start the TAS coordinator"""
        if not self.running:
            self.running = True
            self.processor_thread = threading.Thread(target=self._process_loop)
            self.processor_thread.start()
            print("🚀 TAS Coordinator started")
    
    def stop(self):
        """Stop the TAS coordinator"""
        if self.running:
            self.running = False
            if self.processor_thread:
                self.processor_thread.join()
            print("🛑 TAS Coordinator stopped")
    
    def submit_observation(self, observation: Observation):
        """Submit an observation for processing"""
        self.observation_queue.put(observation)
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Process observations with timeout
                observation = self.observation_queue.get(timeout=0.1)
                self._process_observation(observation)
            except queue.Empty:
                # No observations, apply decay
                self._apply_trust_decay()
            except Exception as e:
                print(f"Error in TAS loop: {e}")
    
    def _process_observation(self, observation: Observation):
        """Process a single observation through TAS loop"""
        
        # 1. Evaluate surprise
        surprise = observation.surprise
        self.surprise_history.append(surprise)
        self.total_observations += 1
        
        if surprise > self.surprise_threshold:
            self.total_surprises += 1
            print(f"⚠️ High surprise from {observation.source}: {surprise:.3f}")
        
        # 2. Update trust
        if observation.source not in self.trust_scores:
            self.trust_scores[observation.source] = TrustScore(
                source=observation.source,
                overall=self.initial_trust,
                talent=self.initial_trust,
                training=self.initial_trust,
                temperament=self.initial_trust
            )
        
        trust = self.trust_scores[observation.source]
        trust.update(surprise, self.trust_update_rate)
        
        # 3. Update attention allocation
        self._update_attention_allocation()
        
        # 4. Store observation in attention target
        if observation.source in self.attention_targets:
            self.attention_targets[observation.source].observations.append(observation)
        
        # 5. Update statistics
        self.avg_surprise = np.mean(self.surprise_history) if self.surprise_history else 0.0
    
    def _update_attention_allocation(self):
        """Reallocate attention based on current trust scores"""
        
        # Sort sources by trust-weighted priority
        sources_by_priority = sorted(
            self.trust_scores.items(),
            key=lambda x: x[1].to_weight(),
            reverse=True
        )
        
        # Select top targets up to max
        selected_sources = sources_by_priority[:self.max_attention_targets]
        
        # Calculate total weight
        total_weight = sum(trust.to_weight() for _, trust in selected_sources)
        
        if total_weight > 0:
            # Clear old targets
            self.attention_targets.clear()
            
            # Allocate attention proportionally
            for source, trust in selected_sources:
                weight = trust.to_weight()
                allocation = (weight / total_weight) * self.attention_budget
                
                if allocation >= self.min_attention_threshold:
                    self.attention_targets[source] = AttentionTarget(
                        source=source,
                        priority=weight,
                        allocated_resources=allocation,
                        observations=[]
                    )
    
    def _apply_trust_decay(self):
        """Apply gradual trust decay to all sources"""
        for trust in self.trust_scores.values():
            # Decay towards initial trust
            diff = trust.overall - self.initial_trust
            decay = diff * self.trust_decay_rate
            trust.overall -= decay
            trust.talent -= decay * 0.8
            trust.training -= decay * 1.2
            trust.temperament -= decay
    
    def get_attention_allocation(self) -> Dict[str, float]:
        """Get current attention allocation"""
        return {
            target.source: target.allocated_resources
            for target in self.attention_targets.values()
        }
    
    def get_trust_scores(self) -> Dict[str, Dict]:
        """Get all trust scores"""
        return {
            source: asdict(trust)
            for source, trust in self.trust_scores.items()
        }
    
    def get_statistics(self) -> Dict:
        """Get coordinator statistics"""
        return {
            "total_observations": self.total_observations,
            "total_surprises": self.total_surprises,
            "surprise_rate": self.total_surprises / max(1, self.total_observations),
            "avg_surprise": self.avg_surprise,
            "num_sources": len(self.trust_scores),
            "num_attention_targets": len(self.attention_targets),
            "attention_utilization": sum(
                t.allocated_resources for t in self.attention_targets.values()
            )
        }


def main():
    """Test the TAS Coordinator"""
    print("🧪 Testing Trust-Attention-Surprise Coordinator")
    print("=" * 50)
    
    # Create coordinator
    config = {
        "max_attention_targets": 3,
        "attention_budget": 1.0,
        "surprise_threshold": 0.3,
        "trust_update_rate": 0.1
    }
    
    coordinator = TASCoordinator(config)
    coordinator.start()
    
    # Simulate observations from different sources
    sources = ["vision", "audio", "lidar", "gps", "imu"]
    
    print("\n📊 Simulating observations...")
    for i in range(20):
        # Pick random source
        source = sources[i % len(sources)]
        
        # Generate observation with varying surprise
        expected = np.random.random()
        
        if np.random.random() < 0.2:  # 20% high surprise
            actual = expected + np.random.random() * 0.5
        else:  # 80% low surprise
            actual = expected + np.random.normal(0, 0.05)
        
        obs = Observation(
            source=source,
            data={"value": actual},
            timestamp=time.time(),
            expected_value=expected,
            actual_value=actual
        )
        
        coordinator.submit_observation(obs)
        time.sleep(0.1)
    
    # Wait for processing
    time.sleep(1)
    
    # Display results
    print("\n🎯 Attention Allocation:")
    allocation = coordinator.get_attention_allocation()
    for source, resources in allocation.items():
        print(f"  {source}: {resources:.2%} of budget")
    
    print("\n🤝 Trust Scores:")
    trust_scores = coordinator.get_trust_scores()
    for source, scores in trust_scores.items():
        print(f"  {source}:")
        print(f"    Overall: {scores['overall']:.3f}")
        print(f"    T3: talent={scores['talent']:.3f}, "
              f"training={scores['training']:.3f}, "
              f"temperament={scores['temperament']:.3f}")
    
    print("\n📈 Statistics:")
    stats = coordinator.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Stop coordinator
    coordinator.stop()
    print("\n✅ TAS Coordinator test complete!")


if __name__ == "__main__":
    main()