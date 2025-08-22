"""
SNARC-SAGE Memory Bridge
Integrates SNARC selective memory with SAGE's hierarchical reasoning architecture
"""

import sys
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
import numpy as np

# Add memory project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../memory')))

try:
    from SNARC.snarc_core import SNARCMemory
    from SNARC.circular_buffer import CircularScratchpad, VerbatimStorage
    from SNARC.consolidation import ConsolidationStrategy, SimpleConceptExtractor
    SNARC_AVAILABLE = True
except ImportError:
    print("Warning: SNARC not available, using stub implementation")
    SNARC_AVAILABLE = False
    
    class SNARCMemory:
        """Stub SNARC for when module not available"""
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, *args, **kwargs):
            return {'total_score': 0.5}
        def write(self, *args, **kwargs):
            pass
        def read(self, *args, **kwargs):
            return torch.zeros(512)


@dataclass
class MemoryState:
    """Combined memory state for SAGE integration"""
    entity_trust: float  # Trust score for entity memory
    sidecar_affect: float  # Affect gating for sidecar
    snarc_score: float  # SNARC attention score
    content: torch.Tensor  # Memory content tensor
    metadata: Dict[str, Any]  # Additional metadata


class SNARCSAGEBridge:
    """
    Bridge between SNARC selective memory and SAGE's dual memory architecture.
    
    This integrates with:
    - Entity Memory: Registry + reputation (who to trust)
    - Sidecar Memory: Episodic/experiential traces (what to recall)
    - SNARC: Selective attention mechanism (what matters)
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        buffer_capacity: int = 10,
        snarc_position: int = 3,
        enable_verbatim: bool = True,
        verbatim_db_path: str = "sage_verbatim.db",
        consolidation_strategy: Optional[ConsolidationStrategy] = None
    ):
        """
        Initialize SNARC-SAGE bridge.
        
        Args:
            hidden_size: Size of memory vectors
            buffer_capacity: Circular buffer capacity
            snarc_position: Position in buffer to trigger SNARC
            enable_verbatim: Whether to store verbatim copies
            verbatim_db_path: Path for verbatim SQLite storage
            consolidation_strategy: Strategy for memory consolidation
        """
        self.hidden_size = hidden_size
        
        # Initialize SNARC components
        if SNARC_AVAILABLE:
            self.snarc = SNARCMemory(
                input_dim=hidden_size,
                memory_size=1000,
                compression_ratio=0.1
            )
            
            # Verbatim storage for full fidelity
            self.verbatim_storage = VerbatimStorage(verbatim_db_path) if enable_verbatim else None
            
            # Circular buffer for short-term memory
            self.buffer = CircularScratchpad(
                capacity=buffer_capacity,
                snarc_position=snarc_position,
                snarc_callback=self._snarc_callback,
                consolidation_callback=self._consolidation_callback,
                enable_consolidation=consolidation_strategy is not None,
                verbatim_storage=self.verbatim_storage,
                enable_verbatim=enable_verbatim
            )
            
            # Consolidation strategy
            self.consolidation_strategy = consolidation_strategy or SimpleConceptExtractor()
        else:
            self.snarc = SNARCMemory()
            self.buffer = None
            self.verbatim_storage = None
            self.consolidation_strategy = None
        
        # Memory statistics
        self.stats = {
            'total_writes': 0,
            'snarc_triggers': 0,
            'consolidations': 0,
            'entity_updates': 0,
            'sidecar_updates': 0
        }
    
    def process_for_sage(
        self,
        input_data: torch.Tensor,
        entity_id: Optional[str] = None,
        context_key: Optional[str] = None,
        affect_signals: Optional[Dict[str, float]] = None
    ) -> MemoryState:
        """
        Process input through SNARC for SAGE integration.
        
        Args:
            input_data: Input tensor to process
            entity_id: Entity identifier for Entity Memory
            context_key: Context for Entity Memory
            affect_signals: Affect signals for Sidecar gating
            
        Returns:
            MemoryState with integrated memory signals
        """
        # Default affect signals
        if affect_signals is None:
            affect_signals = {
                'surprise': 0.0,
                'novelty': 0.0,
                'arousal': 0.0,
                'conflict': 0.0,
                'reward': 0.0
            }
        
        # Process through SNARC
        snarc_result = self.snarc.evaluate(
            input_data,
            surprise=affect_signals.get('surprise', 0.0),
            novelty=affect_signals.get('novelty', 0.0),
            arousal=affect_signals.get('arousal', 0.0),
            conflict=affect_signals.get('conflict', 0.0),
            reward=affect_signals.get('reward', 0.0)
        )
        
        # Add to circular buffer if available
        if self.buffer:
            self.buffer.add({
                'data': input_data,
                'entity_id': entity_id,
                'context_key': context_key,
                'snarc_score': snarc_result['total_score'],
                'timestamp': self._get_timestamp()
            })
        
        # Calculate trust score for Entity Memory
        entity_trust = self._calculate_entity_trust(
            entity_id,
            snarc_result['total_score'],
            affect_signals.get('conflict', 0.0)
        )
        
        # Calculate affect gating for Sidecar
        sidecar_affect = self._calculate_sidecar_affect(
            snarc_result['total_score'],
            affect_signals
        )
        
        # Update statistics
        self.stats['total_writes'] += 1
        
        return MemoryState(
            entity_trust=entity_trust,
            sidecar_affect=sidecar_affect,
            snarc_score=snarc_result['total_score'],
            content=input_data,
            metadata={
                'entity_id': entity_id,
                'context_key': context_key,
                'affect_signals': affect_signals,
                'snarc_components': snarc_result
            }
        )
    
    def bridge_to_entity_memory(
        self,
        memory_state: MemoryState
    ) -> Dict[str, Any]:
        """
        Create Entity Memory update from SNARC state.
        
        Returns dict formatted for Entity Memory system.
        """
        self.stats['entity_updates'] += 1
        
        return {
            'kind': 'snarc_summary',
            'episode_id': f"ep:{self._get_timestamp()}",
            'entity_id': memory_state.metadata.get('entity_id', 'unknown'),
            'ctx_key': memory_state.metadata.get('context_key', 'default'),
            'score_components': {
                'reward': memory_state.metadata['affect_signals'].get('reward', 0.0),
                'coherence': 1.0 - memory_state.metadata['affect_signals'].get('conflict', 0.0),
                'conflict': memory_state.metadata['affect_signals'].get('conflict', 0.0)
            },
            'trust_adjustment': memory_state.entity_trust,
            'ts': self._get_timestamp()
        }
    
    def bridge_to_sidecar(
        self,
        memory_state: MemoryState
    ) -> Tuple[torch.Tensor, float]:
        """
        Create Sidecar Memory update from SNARC state.
        
        Returns:
            Tuple of (memory_vector, write_threshold)
        """
        self.stats['sidecar_updates'] += 1
        
        # Memory vector is the content with affect weighting
        memory_vector = memory_state.content * memory_state.sidecar_affect
        
        # Write threshold based on SNARC score
        # Higher SNARC score = more selective (higher threshold)
        write_threshold = 0.3 + (memory_state.snarc_score * 0.5)
        
        return memory_vector, write_threshold
    
    def get_memory_for_hrm(
        self,
        query: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get memory readout for HRM L-module input.
        
        Args:
            query: Optional query vector for memory retrieval
            
        Returns:
            Memory readout vector for HRM processing
        """
        if query is not None and SNARC_AVAILABLE:
            # Use SNARC's read mechanism
            memory_readout = self.snarc.read(query)
        else:
            # Return zero vector if no query or SNARC unavailable
            memory_readout = torch.zeros(self.hidden_size)
        
        return memory_readout
    
    def consolidate_for_sleep(self) -> List[Dict[str, Any]]:
        """
        Prepare memories for SAGE sleep consolidation.
        
        Returns:
            List of significant memories for sleep processing
        """
        if not self.buffer:
            return []
        
        consolidated = []
        buffer_contents = self.buffer.get_all()
        
        for item in buffer_contents:
            if item.get('snarc_score', 0) > 0.5:  # Only high-attention memories
                consolidated.append({
                    'state': item.get('data'),
                    'entity_id': item.get('entity_id'),
                    'context': item.get('context_key'),
                    'score': item.get('snarc_score'),
                    'timestamp': item.get('timestamp')
                })
        
        self.stats['consolidations'] += 1
        return consolidated
    
    def _snarc_callback(self, item: Dict[str, Any]) -> None:
        """Callback when SNARC processes item from buffer"""
        self.stats['snarc_triggers'] += 1
        
        # Write to SNARC memory if score is high enough
        if item.get('snarc_score', 0) > 0.5:
            self.snarc.write(
                item['data'],
                importance=item['snarc_score']
            )
    
    def _consolidation_callback(self, items: List[Dict[str, Any]]) -> Any:
        """Callback for consolidation strategy"""
        if self.consolidation_strategy:
            return self.consolidation_strategy.consolidate(
                [item['data'] for item in items]
            )
        return None
    
    def _calculate_entity_trust(
        self,
        entity_id: Optional[str],
        snarc_score: float,
        conflict: float
    ) -> float:
        """
        Calculate trust adjustment for Entity Memory.
        
        High SNARC score + low conflict = increase trust
        Low SNARC score + high conflict = decrease trust
        """
        if entity_id is None:
            return 0.0
        
        # Trust increases with attention, decreases with conflict
        trust_delta = (snarc_score * 0.7) - (conflict * 0.3)
        
        # Clamp to reasonable range
        return np.clip(trust_delta, -0.2, 0.2)
    
    def _calculate_sidecar_affect(
        self,
        snarc_score: float,
        affect_signals: Dict[str, float]
    ) -> float:
        """
        Calculate affect gating strength for Sidecar.
        
        Combines SNARC score with affect signals.
        """
        # Weight different signals
        affect_sum = (
            affect_signals.get('surprise', 0.0) * 0.2 +
            affect_signals.get('novelty', 0.0) * 0.2 +
            affect_signals.get('arousal', 0.0) * 0.3 +
            affect_signals.get('reward', 0.0) * 0.2 +
            (1.0 - affect_signals.get('conflict', 0.0)) * 0.1
        )
        
        # Combine with SNARC score
        return (snarc_score * 0.6) + (affect_sum * 0.4)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return self.stats.copy()


class HRMMemoryIntegration:
    """
    Integration layer for HRM (Hierarchical Reasoning Model) with SNARC memory.
    
    Maps SNARC to HRM's dual-module architecture:
    - L-module: Processes immediate sensor/memory data
    - H-module: Maintains strategic memory state
    """
    
    def __init__(
        self,
        bridge: SNARCSAGEBridge,
        hidden_size: int = 512
    ):
        """
        Initialize HRM memory integration.
        
        Args:
            bridge: SNARC-SAGE bridge instance
            hidden_size: Size of HRM hidden states
        """
        self.bridge = bridge
        self.hidden_size = hidden_size
        
        # Track H and L module states
        self.h_memory_state = torch.zeros(hidden_size)
        self.l_memory_state = torch.zeros(hidden_size)
    
    def prepare_l_module_input(
        self,
        sensor_data: torch.Tensor,
        memory_query: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prepare input for HRM L-module including memory.
        
        Args:
            sensor_data: Raw sensor input
            memory_query: Optional query for memory retrieval
            
        Returns:
            Combined sensor + memory input for L-module
        """
        # Get memory readout from SNARC
        memory_readout = self.bridge.get_memory_for_hrm(memory_query)
        
        # Concatenate sensor and memory data
        combined_input = torch.cat([sensor_data, memory_readout], dim=-1)
        
        # Update L-module memory state
        self.l_memory_state = memory_readout
        
        return combined_input
    
    def update_h_module_memory(
        self,
        h_state: torch.Tensor,
        coherence_score: float
    ) -> None:
        """
        Update H-module memory state based on coherence.
        
        Args:
            h_state: Current H-module state
            coherence_score: Coherence score from H-module
        """
        # Process through SNARC with coherence as reward signal
        memory_state = self.bridge.process_for_sage(
            h_state,
            affect_signals={'reward': coherence_score}
        )
        
        # Update H-module memory
        self.h_memory_state = memory_state.content
    
    def get_sleep_data(self) -> List[Dict[str, Any]]:
        """
        Get consolidated memories for HRM sleep cycle.
        
        Returns:
            List of memories for sleep consolidation
        """
        sleep_data = self.bridge.consolidate_for_sleep()
        
        # Add HRM-specific metadata
        for item in sleep_data:
            item['h_state'] = self.h_memory_state.clone()
            item['l_state'] = self.l_memory_state.clone()
        
        return sleep_data


# Example usage and testing
if __name__ == "__main__":
    print("Testing SNARC-SAGE Bridge...")
    
    # Initialize bridge
    bridge = SNARCSAGEBridge(
        hidden_size=512,
        buffer_capacity=10,
        enable_verbatim=True
    )
    
    # Test data
    test_input = torch.randn(512)
    
    # Process through bridge
    memory_state = bridge.process_for_sage(
        test_input,
        entity_id="entity:sensor/camera@1.0",
        context_key="visual@1080p",
        affect_signals={
            'surprise': 0.7,
            'novelty': 0.5,
            'arousal': 0.3,
            'conflict': 0.1,
            'reward': 0.8
        }
    )
    
    print(f"SNARC Score: {memory_state.snarc_score:.3f}")
    print(f"Entity Trust: {memory_state.entity_trust:.3f}")
    print(f"Sidecar Affect: {memory_state.sidecar_affect:.3f}")
    
    # Create updates for SAGE subsystems
    entity_update = bridge.bridge_to_entity_memory(memory_state)
    print(f"\nEntity Memory Update: {entity_update}")
    
    sidecar_vector, threshold = bridge.bridge_to_sidecar(memory_state)
    print(f"\nSidecar Write Threshold: {threshold:.3f}")
    
    # Test HRM integration
    hrm_integration = HRMMemoryIntegration(bridge)
    
    sensor_data = torch.randn(256)
    l_input = hrm_integration.prepare_l_module_input(sensor_data)
    print(f"\nL-module input shape: {l_input.shape}")
    
    # Get statistics
    stats = bridge.get_stats()
    print(f"\nBridge Statistics: {stats}")
    
    print("\nSNARC-SAGE Bridge test complete!")