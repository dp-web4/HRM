"""
SAGE with SNARC Integration
Complete example of how SNARC memory enhances SAGE's hierarchical reasoning
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

# Import the bridge
from snarc_bridge import SNARCSAGEBridge, HRMMemoryIntegration, MemoryState


@dataclass
class SAGECarryState:
    """Extended carry state including memory"""
    z_H: torch.Tensor  # H-module state
    z_L: torch.Tensor  # L-module state
    memory_h: torch.Tensor  # H-module memory
    memory_l: torch.Tensor  # L-module memory
    iteration: int  # Current iteration
    coherence_history: List[float]  # Coherence over time


class SAGEWithSNARC(nn.Module):
    """
    SAGE (Sentient Agentic Generative Engine) with SNARC memory integration.
    
    This demonstrates how SNARC's selective attention mechanism enhances
    SAGE's hierarchical reasoning and dual memory architecture.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        h_layers: int = 4,
        l_layers: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 2,
        enable_snarc: bool = True,
        buffer_capacity: int = 10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        
        # Initialize SNARC bridge
        self.enable_snarc = enable_snarc
        if enable_snarc:
            self.snarc_bridge = SNARCSAGEBridge(
                hidden_size=hidden_dim,
                buffer_capacity=buffer_capacity,
                enable_verbatim=True
            )
            self.hrm_memory = HRMMemoryIntegration(
                self.snarc_bridge,
                hidden_size=hidden_dim
            )
        else:
            self.snarc_bridge = None
            self.hrm_memory = None
        
        # Simplified H and L modules (placeholders for actual HRM implementation)
        self.h_module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.l_module = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coherence computation
        self.coherence_head = nn.Linear(hidden_dim, 1)
        
        # Trust weighting for different sensors
        self.sensor_trust = nn.Parameter(torch.ones(3))  # camera, audio, memory
        
        # Statistics
        self.stats = {
            'total_iterations': 0,
            'memory_writes': 0,
            'memory_reads': 0,
            'coherence_peaks': 0
        }
    
    def initial_carry(self, batch_size: int = 1) -> SAGECarryState:
        """Initialize carry state for new sequence"""
        return SAGECarryState(
            z_H=torch.zeros(batch_size, self.hidden_dim),
            z_L=torch.zeros(batch_size, self.hidden_dim),
            memory_h=torch.zeros(batch_size, self.hidden_dim),
            memory_l=torch.zeros(batch_size, self.hidden_dim),
            iteration=0,
            coherence_history=[]
        )
    
    def forward(
        self,
        carry: SAGECarryState,
        sensors: Dict[str, torch.Tensor],
        affect_signals: Optional[Dict[str, float]] = None
    ) -> tuple[SAGECarryState, Dict[str, Any]]:
        """
        Forward pass through SAGE with SNARC memory.
        
        Args:
            carry: Current carry state
            sensors: Dict with 'camera', 'audio', etc. tensors
            affect_signals: Optional affect signals for memory gating
            
        Returns:
            Updated carry state and outputs dict
        """
        outputs = {
            'coherence_scores': [],
            'memory_states': [],
            'trust_weights': []
        }
        
        # Default affect signals
        if affect_signals is None:
            affect_signals = self._compute_affect_from_sensors(sensors)
        
        # Hierarchical reasoning cycles
        for h_cycle in range(self.h_cycles):
            # L-module cycles (tactical processing)
            for l_cycle in range(self.l_cycles):
                # Process sensors with memory
                if self.enable_snarc and self.hrm_memory:
                    # Get memory-augmented input
                    sensor_input = self._fuse_sensors(sensors)
                    l_input = self.hrm_memory.prepare_l_module_input(
                        sensor_input,
                        memory_query=carry.z_L
                    )
                    self.stats['memory_reads'] += 1
                else:
                    # Just fuse sensors without memory
                    l_input = torch.cat([
                        self._fuse_sensors(sensors),
                        carry.z_H,
                        carry.memory_l
                    ], dim=-1)
                
                # L-module update
                carry.z_L = self.l_module(l_input)
                
                # Store L-module experience in SNARC
                if self.enable_snarc and self.snarc_bridge:
                    memory_state = self.snarc_bridge.process_for_sage(
                        carry.z_L,
                        entity_id=f"l_module_cycle_{l_cycle}",
                        context_key=f"h_cycle_{h_cycle}",
                        affect_signals=affect_signals
                    )
                    carry.memory_l = memory_state.content
                    outputs['memory_states'].append(memory_state)
                    self.stats['memory_writes'] += 1
            
            # H-module update (strategic processing)
            h_input = torch.cat([carry.z_L, carry.memory_h], dim=-1)
            carry.z_H = self.h_module(h_input)
            
            # Compute coherence
            coherence = torch.sigmoid(self.coherence_head(carry.z_H)).item()
            carry.coherence_history.append(coherence)
            outputs['coherence_scores'].append(coherence)
            
            # Update H-module memory based on coherence
            if self.enable_snarc and self.hrm_memory:
                self.hrm_memory.update_h_module_memory(
                    carry.z_H,
                    coherence
                )
                carry.memory_h = self.hrm_memory.h_memory_state
            
            # Track coherence peaks
            if coherence > 0.8:
                self.stats['coherence_peaks'] += 1
            
            # Update trust weights based on coherence
            trust_update = self._update_trust(coherence, affect_signals)
            outputs['trust_weights'].append(trust_update)
        
        # Update iteration count
        carry.iteration += 1
        self.stats['total_iterations'] += 1
        
        # Final outputs
        outputs['final_coherence'] = coherence
        outputs['final_state'] = carry.z_H
        
        # Get memory statistics if SNARC enabled
        if self.enable_snarc and self.snarc_bridge:
            outputs['memory_stats'] = self.snarc_bridge.get_stats()
        
        return carry, outputs
    
    def sleep_consolidation(
        self,
        num_dreams: int = 10
    ) -> Dict[str, Any]:
        """
        Run sleep consolidation cycle.
        
        Args:
            num_dreams: Number of dream sequences to generate
            
        Returns:
            Consolidation results
        """
        if not self.enable_snarc or not self.hrm_memory:
            return {'status': 'SNARC not enabled'}
        
        # Get memories for consolidation
        sleep_data = self.hrm_memory.get_sleep_data()
        
        results = {
            'memories_processed': len(sleep_data),
            'dreams_generated': 0,
            'patterns_extracted': []
        }
        
        # Generate dreams (synthetic experiences)
        for i in range(min(num_dreams, len(sleep_data))):
            if i < len(sleep_data):
                memory = sleep_data[i]
                
                # Create dream by interpolating memories
                if i > 0:
                    prev_memory = sleep_data[i-1]
                    # Interpolate between consecutive memories
                    alpha = np.random.random()
                    dream_state = (
                        alpha * memory['h_state'] +
                        (1 - alpha) * prev_memory['h_state']
                    )
                else:
                    # Amplify single memory
                    dream_state = memory['h_state'] * (1 + np.random.random())
                
                # Process dream through system
                dream_carry = self.initial_carry()
                dream_carry.z_H = dream_state
                
                # Run abbreviated cycles
                dream_sensors = {
                    'camera': torch.randn(1, self.input_dim // 3),
                    'audio': torch.randn(1, self.input_dim // 3),
                    'imu': torch.randn(1, self.input_dim // 3)
                }
                
                _, dream_output = self.forward(
                    dream_carry,
                    dream_sensors,
                    affect_signals={'arousal': 0.3, 'reward': 0.5}
                )
                
                results['dreams_generated'] += 1
                
                # Extract patterns from dreams
                if dream_output['final_coherence'] > 0.7:
                    results['patterns_extracted'].append({
                        'coherence': dream_output['final_coherence'],
                        'source_memories': [i, i-1] if i > 0 else [i]
                    })
        
        return results
    
    def _fuse_sensors(self, sensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple sensors with trust weighting"""
        # Simple concatenation for this example
        # In practice, would use trust-weighted attention
        sensor_list = []
        weights = []
        
        if 'camera' in sensors:
            sensor_list.append(sensors['camera'])
            weights.append(self.sensor_trust[0])
        if 'audio' in sensors:
            sensor_list.append(sensors['audio'])
            weights.append(self.sensor_trust[1])
        if 'imu' in sensors:
            sensor_list.append(sensors['imu'])
            weights.append(self.sensor_trust[2])
        
        if sensor_list:
            # Weighted average
            weights = torch.softmax(torch.stack(weights), dim=0)
            fused = sum(w * s for w, s in zip(weights, sensor_list))
            return fused
        else:
            return torch.zeros(1, self.input_dim // 3)
    
    def _compute_affect_from_sensors(
        self,
        sensors: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute affect signals from sensor data"""
        # Simplified affect computation
        # In practice, would use learned affect detection
        
        affect = {
            'surprise': 0.0,
            'novelty': 0.0,
            'arousal': 0.0,
            'conflict': 0.0,
            'reward': 0.0
        }
        
        # Check for sensor conflicts
        if len(sensors) > 1:
            # Compute variance across sensors as conflict signal
            sensor_tensors = list(sensors.values())
            if len(sensor_tensors) > 1:
                variance = torch.var(torch.stack([s.mean() for s in sensor_tensors]))
                affect['conflict'] = min(variance.item(), 1.0)
        
        # Compute novelty from sensor magnitudes
        for sensor in sensors.values():
            magnitude = torch.norm(sensor).item()
            if magnitude > 2.0:  # Threshold for novelty
                affect['novelty'] = min(magnitude / 5.0, 1.0)
                affect['arousal'] = min(magnitude / 10.0, 1.0)
        
        return affect
    
    def _update_trust(
        self,
        coherence: float,
        affect_signals: Dict[str, float]
    ) -> torch.Tensor:
        """Update sensor trust weights based on coherence"""
        # High coherence + low conflict = increase trust
        # Low coherence + high conflict = decrease trust
        
        conflict = affect_signals.get('conflict', 0.0)
        trust_delta = (coherence - 0.5) * (1.0 - conflict) * 0.1
        
        # Update trust weights
        with torch.no_grad():
            self.sensor_trust += trust_delta
            # Keep trust weights positive and normalized
            self.sensor_trust.clamp_(min=0.1, max=2.0)
        
        return self.sensor_trust.clone()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()
        if self.enable_snarc and self.snarc_bridge:
            stats['snarc_stats'] = self.snarc_bridge.get_stats()
        stats['sensor_trust'] = self.sensor_trust.detach().numpy().tolist()
        return stats


def demonstrate_sage_snarc_integration():
    """Demonstrate SAGE with SNARC in action"""
    print("=" * 60)
    print("SAGE with SNARC Integration Demo")
    print("=" * 60)
    
    # Initialize SAGE with SNARC
    sage = SAGEWithSNARC(
        input_dim=96,  # 32 per sensor
        hidden_dim=128,
        h_cycles=2,
        l_cycles=2,
        enable_snarc=True,
        buffer_capacity=5
    )
    
    # Initialize carry state
    carry = sage.initial_carry()
    
    # Simulate 5 time steps
    print("\n📊 Processing sensor streams...")
    for t in range(5):
        # Generate sensor data
        sensors = {
            'camera': torch.randn(1, 32) * (1 + t * 0.2),  # Increasing intensity
            'audio': torch.randn(1, 32) * 0.5,
            'imu': torch.randn(1, 32) * 0.3
        }
        
        # Add some structure to create patterns
        if t % 2 == 0:
            sensors['camera'] += torch.ones(1, 32) * 0.5  # Pattern in camera
        
        # Process through SAGE
        carry, outputs = sage(carry, sensors)
        
        print(f"\nTime {t}:")
        print(f"  Coherence: {outputs['final_coherence']:.3f}")
        if outputs['memory_states']:
            print(f"  SNARC Score: {outputs['memory_states'][0].snarc_score:.3f}")
        print(f"  Trust weights: {outputs['trust_weights'][0].numpy()}")
    
    # Run sleep consolidation
    print("\n😴 Running sleep consolidation...")
    sleep_results = sage.sleep_consolidation(num_dreams=3)
    print(f"  Memories processed: {sleep_results['memories_processed']}")
    print(f"  Dreams generated: {sleep_results['dreams_generated']}")
    print(f"  Patterns extracted: {len(sleep_results['patterns_extracted'])}")
    
    # Final statistics
    print("\n📈 Final Statistics:")
    stats = sage.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ SAGE-SNARC integration demonstration complete!")
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("- SNARC provides selective attention for memory formation")
    print("- Circular buffer enables context binding at x-from-last")
    print("- Dual memory (verbatim + conceptual) preserves fidelity")
    print("- Sleep consolidation extracts patterns from experience")
    print("- Trust weights evolve based on coherence and conflict")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_sage_snarc_integration()