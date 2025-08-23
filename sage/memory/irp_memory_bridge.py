"""
IRP-SNARC Memory Bridge
Connects SNARC selective memory to IRP refinement process
Enables memory-guided refinement and experience consolidation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Try to import SNARC from Memory project if available
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', 'Memory'))
    from SNARC.circular_buffer import CircularBuffer
    from SNARC.full_implementation.snarc_core import SNARCCore
    SNARC_AVAILABLE = True
except ImportError:
    SNARC_AVAILABLE = False
    print("Warning: SNARC not available. Using mock implementation.")


@dataclass
class RefinementMemory:
    """Memory of a refinement trajectory"""
    plugin_id: str
    initial_state: Any
    final_state: Any
    energy_trajectory: List[float]
    iterations: int
    compute_saved: float
    trust_score: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def convergence_rate(self) -> float:
        """How quickly the refinement converged"""
        if len(self.energy_trajectory) < 2:
            return 0.0
        initial_delta = abs(self.energy_trajectory[1] - self.energy_trajectory[0])
        final_delta = abs(self.energy_trajectory[-1] - self.energy_trajectory[-2])
        return 1.0 - (final_delta / (initial_delta + 1e-6))
    
    @property
    def efficiency(self) -> float:
        """Overall efficiency of this refinement"""
        return self.trust_score * self.compute_saved


class MockSNARC:
    """Mock SNARC implementation for when the real one isn't available"""
    
    def __init__(self, capacity: int = 1000):
        self.memories = []
        self.capacity = capacity
    
    def evaluate(self, memory: RefinementMemory) -> float:
        """Simple salience scoring"""
        # High salience for efficient, fast-converging refinements
        salience = memory.efficiency * memory.convergence_rate
        return min(salience, 1.0)
    
    def store(self, memory: RefinementMemory, salience: float):
        """Store if salient enough"""
        if salience > 0.3:  # Threshold
            self.memories.append(memory)
            if len(self.memories) > self.capacity:
                self.memories.pop(0)
    
    def retrieve_similar(self, query: Dict[str, Any], k: int = 5) -> List[RefinementMemory]:
        """Retrieve similar memories"""
        # Simple retrieval based on plugin_id
        plugin_id = query.get('plugin_id', '')
        similar = [m for m in self.memories if m.plugin_id == plugin_id]
        return similar[-k:] if similar else []


class IRPMemoryBridge:
    """
    Bridges IRP refinement with SNARC selective memory
    Stores successful refinement patterns for reuse
    """
    
    def __init__(
        self,
        buffer_size: int = 100,
        snarc_capacity: int = 1000,
        consolidation_threshold: int = 50,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SNARC or mock
        if SNARC_AVAILABLE:
            self.buffer = CircularBuffer(buffer_size)
            self.snarc = SNARCCore(
                input_dim=256,  # Placeholder
                hidden_dim=128,
                output_dim=64,
                capacity=snarc_capacity
            )
            print("Using real SNARC implementation")
        else:
            self.buffer = []
            self.snarc = MockSNARC(snarc_capacity)
            print("Using mock SNARC implementation")
        
        self.consolidation_threshold = consolidation_threshold
        self.pending_memories = []
        self.pattern_library = {}  # Extracted patterns for reuse
        
    def record_refinement(
        self,
        plugin_id: str,
        initial_state: Any,
        final_state: Any,
        energy_trajectory: List[float],
        telemetry: Dict[str, Any]
    ) -> RefinementMemory:
        """Record a refinement trajectory"""
        
        memory = RefinementMemory(
            plugin_id=plugin_id,
            initial_state=initial_state,
            final_state=final_state,
            energy_trajectory=energy_trajectory,
            iterations=telemetry.get('iterations', 0),
            compute_saved=telemetry.get('compute_saved', 0),
            trust_score=telemetry.get('trust', 0.5),
            context=telemetry
        )
        
        # Add to pending for consolidation
        self.pending_memories.append(memory)
        
        # Evaluate salience
        salience = self._evaluate_salience(memory)
        
        # Store if salient
        if isinstance(self.snarc, MockSNARC):
            self.snarc.store(memory, salience)
        else:
            # Real SNARC evaluation
            if salience > 0.3:
                # Convert to SNARC format and store
                pass  # TODO: Implement real SNARC storage
        
        # Check if consolidation needed
        if len(self.pending_memories) >= self.consolidation_threshold:
            self.consolidate()
        
        return memory
    
    def _evaluate_salience(self, memory: RefinementMemory) -> float:
        """Evaluate how salient/important this memory is"""
        if isinstance(self.snarc, MockSNARC):
            return self.snarc.evaluate(memory)
        
        # Real SNARC salience computation
        # Based on surprise, novelty, affect
        salience = memory.efficiency * memory.convergence_rate
        
        # Boost for very efficient refinements
        if memory.compute_saved > 0.9:
            salience *= 1.5
        
        # Boost for high trust
        if memory.trust_score > 0.8:
            salience *= 1.2
        
        return min(salience, 1.0)
    
    def retrieve_guidance(
        self,
        plugin_id: str,
        current_state: Any,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve guidance from memory for current refinement
        Returns suggested parameters based on past success
        """
        
        query = {
            'plugin_id': plugin_id,
            'state': current_state
        }
        
        # Retrieve similar successful refinements
        if isinstance(self.snarc, MockSNARC):
            similar_memories = self.snarc.retrieve_similar(query, k)
        else:
            similar_memories = []  # TODO: Implement real SNARC retrieval
        
        if not similar_memories:
            return {
                'max_iterations': 30,
                'early_stop_threshold': 0.01,
                'trust_weight': 1.0
            }
        
        # Aggregate guidance from similar memories
        avg_iterations = sum(m.iterations for m in similar_memories) / len(similar_memories)
        avg_convergence = sum(m.convergence_rate for m in similar_memories) / len(similar_memories)
        best_efficiency = max(m.efficiency for m in similar_memories)
        
        # Check if we have a pattern for this
        pattern_key = f"{plugin_id}_pattern"
        if pattern_key in self.pattern_library:
            pattern = self.pattern_library[pattern_key]
        else:
            pattern = None
        
        return {
            'max_iterations': int(avg_iterations * 1.2),  # Allow some headroom
            'early_stop_threshold': 0.01 * (2 - avg_convergence),  # Tighter for fast convergers
            'trust_weight': min(best_efficiency * 1.1, 1.0),
            'pattern': pattern,
            'similar_memories': len(similar_memories)
        }
    
    def consolidate(self):
        """
        Consolidate pending memories into patterns
        This is the 'sleep' phase where we extract reusable knowledge
        """
        
        if not self.pending_memories:
            return
        
        print(f"Consolidating {len(self.pending_memories)} memories...")
        
        # Group by plugin
        by_plugin = {}
        for memory in self.pending_memories:
            if memory.plugin_id not in by_plugin:
                by_plugin[memory.plugin_id] = []
            by_plugin[memory.plugin_id].append(memory)
        
        # Extract patterns per plugin
        for plugin_id, memories in by_plugin.items():
            pattern = self._extract_pattern(memories)
            if pattern:
                self.pattern_library[f"{plugin_id}_pattern"] = pattern
        
        # Clear pending
        self.pending_memories = []
        
        print(f"Extracted {len(self.pattern_library)} patterns")
    
    def _extract_pattern(self, memories: List[RefinementMemory]) -> Optional[Dict[str, Any]]:
        """Extract reusable pattern from a set of memories"""
        
        if len(memories) < 3:
            return None
        
        # Find common successful strategies
        efficient_memories = [m for m in memories if m.efficiency > 0.5]
        
        if not efficient_memories:
            return None
        
        # Extract pattern
        pattern = {
            'avg_iterations': sum(m.iterations for m in efficient_memories) / len(efficient_memories),
            'avg_convergence_rate': sum(m.convergence_rate for m in efficient_memories) / len(efficient_memories),
            'best_efficiency': max(m.efficiency for m in efficient_memories),
            'energy_profile': self._compute_energy_profile(efficient_memories),
            'confidence': len(efficient_memories) / len(memories)
        }
        
        return pattern
    
    def _compute_energy_profile(self, memories: List[RefinementMemory]) -> List[float]:
        """Compute average energy descent profile"""
        
        max_len = max(len(m.energy_trajectory) for m in memories)
        profile = []
        
        for i in range(max_len):
            values = []
            for m in memories:
                if i < len(m.energy_trajectory):
                    values.append(m.energy_trajectory[i])
            
            if values:
                profile.append(sum(values) / len(values))
        
        return profile
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        
        total_memories = len(self.snarc.memories) if isinstance(self.snarc, MockSNARC) else 0
        
        stats = {
            'total_memories': total_memories,
            'pending_consolidation': len(self.pending_memories),
            'patterns_extracted': len(self.pattern_library),
            'plugins_with_patterns': list(set(k.replace('_pattern', '') for k in self.pattern_library.keys()))
        }
        
        if total_memories > 0 and isinstance(self.snarc, MockSNARC):
            memories = self.snarc.memories
            stats['avg_efficiency'] = sum(m.efficiency for m in memories) / len(memories)
            stats['avg_convergence'] = sum(m.convergence_rate for m in memories) / len(memories)
            stats['avg_iterations'] = sum(m.iterations for m in memories) / len(memories)
        
        return stats


class MemoryGuidedIRP:
    """
    Wrapper that adds memory guidance to any IRP plugin
    """
    
    def __init__(
        self,
        irp_plugin: Any,
        memory_bridge: IRPMemoryBridge
    ):
        self.plugin = irp_plugin
        self.memory = memory_bridge
        self.plugin_id = getattr(irp_plugin, 'entity_id', 'unknown')
        
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Refine with memory guidance
        """
        
        # Get guidance from memory
        guidance = self.memory.retrieve_guidance(self.plugin_id, x)
        
        # Update plugin parameters based on guidance
        if hasattr(self.plugin, 'max_iterations'):
            self.plugin.max_iterations = guidance['max_iterations']
        if hasattr(self.plugin, 'eps'):
            self.plugin.eps = guidance['early_stop_threshold']
        
        # Track energy for memory
        energy_trajectory = []
        
        # Store original compute_energy if exists
        original_compute_energy = None
        if hasattr(self.plugin, 'compute_energy'):
            original_compute_energy = self.plugin.compute_energy
            
            # Wrap to track energy
            def tracked_compute_energy(state):
                energy = original_compute_energy(state)
                energy_trajectory.append(energy)
                return energy
            
            self.plugin.compute_energy = tracked_compute_energy
        
        # Run refinement
        initial_state = x
        refined, telemetry = self.plugin.refine(x, early_stop)
        
        # Restore original compute_energy
        if original_compute_energy:
            self.plugin.compute_energy = original_compute_energy
        
        # Record to memory
        memory = self.memory.record_refinement(
            plugin_id=self.plugin_id,
            initial_state=initial_state,
            final_state=refined,
            energy_trajectory=energy_trajectory,
            telemetry=telemetry
        )
        
        # Add memory info to telemetry
        telemetry['memory_guidance'] = guidance
        telemetry['memory_efficiency'] = memory.efficiency
        telemetry['convergence_rate'] = memory.convergence_rate
        
        return refined, telemetry


def test_memory_bridge():
    """Test the memory bridge with IRP plugins"""
    
    print("=" * 60)
    print("Testing IRP-SNARC Memory Bridge")
    print("=" * 60)
    
    # Create memory bridge
    memory_bridge = IRPMemoryBridge(
        buffer_size=50,
        consolidation_threshold=10
    )
    
    # Create vision IRP with memory
    from sage.irp.plugins.vision_impl import create_vision_irp
    
    vision_irp = create_vision_irp()
    memory_guided_vision = MemoryGuidedIRP(vision_irp, memory_bridge)
    
    # Run multiple refinements to build memory
    print("\n1. Building memory with multiple refinements...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(15):
        # Random test image
        test_image = torch.randn(1, 3, 224, 224).to(device)
        
        # Refine with memory guidance
        refined, telemetry = memory_guided_vision.refine(test_image, early_stop=True)
        
        if i % 5 == 0:
            print(f"   Refinement {i+1}: {telemetry['iterations']} iterations, "
                  f"efficiency: {telemetry.get('memory_efficiency', 0):.3f}")
    
    # Check memory stats
    print("\n2. Memory Statistics:")
    stats = memory_bridge.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test pattern extraction
    print("\n3. Testing pattern extraction...")
    
    # Retrieve guidance
    guidance = memory_bridge.retrieve_guidance(
        'vision_irp',
        torch.randn(1, 3, 224, 224).to(device)
    )
    
    print(f"   Guidance from memory:")
    print(f"     Max iterations: {guidance['max_iterations']}")
    print(f"     Early stop threshold: {guidance['early_stop_threshold']:.4f}")
    print(f"     Trust weight: {guidance['trust_weight']:.3f}")
    if 'similar_memories' in guidance:
        print(f"     Based on {guidance['similar_memories']} similar memories")
    
    print("\n" + "=" * 60)
    print("Memory bridge test complete!")
    
    return memory_bridge


if __name__ == "__main__":
    test_memory_bridge()