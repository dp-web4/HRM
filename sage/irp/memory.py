"""
Memory IRP Plugin Implementation  
Version: 1.0 (2025-08-23)

Sleep consolidation through progressive abstraction layers.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import sqlite3
import json
from datetime import datetime

from .base import IRPPlugin, IRPState


class MemoryIRP(IRPPlugin):
    """
    Memory plugin for sleep consolidation and abstraction.
    
    Key features:
    - Progressive abstraction from episodic to strategic
    - Offline refinement during "sleep" cycles
    - Pattern extraction through augmentation
    - Value attribution for consolidated knowledge
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Memory IRP with abstraction networks.
        
        Config parameters:
            - memory_dim: Dimension of memory representations (default 256)
            - abstraction_levels: List of abstraction levels
            - consolidation_rate: Rate of memory consolidation
            - augmentation_types: Types of augmentation for pattern extraction
            - db_path: Path to verbatim memory database
            - device: Compute device
        """
        super().__init__(config)
        
        self.memory_dim = config.get('memory_dim', 256)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Abstraction levels (from concrete to abstract)
        self.abstraction_levels = config.get('abstraction_levels', [
            'episodic',     # Specific events
            'semantic',     # General knowledge
            'procedural',   # How-to knowledge
            'conceptual',   # Abstract principles
            'strategic'     # Meta-level patterns
        ])
        
        # Build abstraction networks for each level
        self.abstractors = nn.ModuleDict({
            level: self._build_abstractor(level)
            for level in self.abstraction_levels
        })
        
        # Memory consolidation parameters
        self.consolidation_rate = config.get('consolidation_rate', 0.1)
        self.augmentation_types = config.get('augmentation_types', [
            'temporal_shift',
            'feature_dropout',
            'noise_injection',
            'permutation'
        ])
        
        # Verbatim storage
        self.db_path = config.get('db_path', 'memory.db')
        self._init_database()
        
        # Memory layers for hierarchical storage
        self.memory_layers = {}
        
    def _build_abstractor(self, level: str) -> nn.Module:
        """
        Build abstraction network for specific level.
        
        Different architectures for different abstraction levels.
        """
        if level == 'episodic':
            # Minimal processing for episodic memories
            return nn.Sequential(
                nn.Linear(self.memory_dim, self.memory_dim),
                nn.LayerNorm(self.memory_dim),
                nn.ReLU(),
                nn.Linear(self.memory_dim, self.memory_dim)
            )
        
        elif level == 'semantic':
            # Extract semantic content
            return nn.Sequential(
                nn.Linear(self.memory_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, self.memory_dim),
                nn.Tanh()
            )
        
        elif level == 'procedural':
            # Extract action sequences
            return nn.Sequential(
                nn.Linear(self.memory_dim, 384),
                nn.LayerNorm(384),
                nn.ReLU(),
                nn.Linear(384, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, self.memory_dim)
            )
        
        elif level == 'conceptual':
            # Extract abstract concepts
            return nn.Sequential(
                nn.Linear(self.memory_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, self.memory_dim // 2),  # Compress
                nn.Tanh()
            )
        
        else:  # strategic
            # Extract meta-patterns
            return nn.Sequential(
                nn.Linear(self.memory_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, self.memory_dim // 4),  # High compression
                nn.Tanh()
            )
    
    def _init_database(self):
        """Initialize SQLite database for verbatim storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                level TEXT,
                content BLOB,
                metadata TEXT,
                consolidation_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def augment_memory(self, memory: torch.Tensor, aug_type: str) -> torch.Tensor:
        """
        Apply augmentation to create training variations.
        
        Args:
            memory: Memory representation [B, D]
            aug_type: Type of augmentation
            
        Returns:
            Augmented memory
        """
        if aug_type == 'temporal_shift':
            # Shift temporal aspects
            shift = torch.randn_like(memory) * 0.1
            return memory + shift
        
        elif aug_type == 'feature_dropout':
            # Randomly drop features
            mask = torch.bernoulli(torch.ones_like(memory) * 0.8)
            return memory * mask
        
        elif aug_type == 'noise_injection':
            # Add Gaussian noise
            noise = torch.randn_like(memory) * 0.2
            return memory + noise
        
        elif aug_type == 'permutation':
            # Permute dimensions
            perm = torch.randperm(memory.shape[-1])
            return memory[..., perm]
        
        else:
            return memory
    
    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize consolidation state from raw experiences.
        
        Args:
            x0: Raw experience data (list of experience dicts)
            task_ctx: Context including consolidation goals
            
        Returns:
            Initial IRPState for consolidation
        """
        # Convert experiences to tensor representation
        if isinstance(x0, list):
            # Assume each experience has 'embedding' field
            embeddings = []
            for exp in x0:
                if isinstance(exp, dict) and 'embedding' in exp:
                    embeddings.append(exp['embedding'])
                else:
                    # Create random embedding for demo
                    embeddings.append(np.random.randn(self.memory_dim))
            
            memory_batch = torch.tensor(np.array(embeddings), dtype=torch.float32)
        else:
            memory_batch = x0
        
        memory_batch = memory_batch.to(self.device)
        
        # Add batch dimension if needed
        if memory_batch.dim() == 1:
            memory_batch = memory_batch.unsqueeze(0)
        
        # Initialize at episodic level
        meta = {
            'raw_experiences': x0,
            'current_level': 'episodic',
            'level_idx': 0,
            'compression_history': [],
            'retrieval_accuracy_history': [],
            'task_ctx': task_ctx
        }
        
        return IRPState(
            x=memory_batch,
            step_idx=0,
            energy_val=None,
            meta=meta
        )
    
    def energy(self, state: IRPState) -> float:
        """
        Compute energy as negative of (compression * retrieval_accuracy).
        
        Good consolidation achieves high compression with high retrieval.
        
        Args:
            state: Current consolidation state
            
        Returns:
            Scalar energy (negative reward)
        """
        memory = state.x
        level = state.meta['current_level']
        
        # Measure compression
        original_size = self.memory_dim
        if level == 'conceptual':
            compressed_size = self.memory_dim // 2
        elif level == 'strategic':
            compressed_size = self.memory_dim // 4
        else:
            compressed_size = self.memory_dim
        
        compression_ratio = original_size / compressed_size
        
        # Measure retrieval accuracy (simplified)
        # In practice, would test on retrieval tasks
        with torch.no_grad():
            # Reconstruct from compressed representation
            if level in ['conceptual', 'strategic']:
                # Pad back to original size for comparison
                padded = torch.nn.functional.pad(
                    memory,
                    (0, original_size - memory.shape[-1])
                )
                reconstruction_error = 0.1  # Simplified
            else:
                reconstruction_error = 0.05
        
        retrieval_accuracy = 1.0 - reconstruction_error
        
        # Track metrics
        state.meta['compression_history'].append(compression_ratio)
        state.meta['retrieval_accuracy_history'].append(retrieval_accuracy)
        
        # Energy is negative reward
        energy = -(compression_ratio * retrieval_accuracy)
        
        return float(energy)
    
    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Execute one consolidation step.
        
        Progressive abstraction through levels with augmentation.
        
        Args:
            state: Current memory state
            noise_schedule: Optional schedule
            
        Returns:
            Refined memory state
        """
        memory = state.x
        step_idx = state.step_idx
        current_level = state.meta['current_level']
        
        # Determine if we should progress to next level
        max_steps = self.config.get('max_iterations', 100)
        steps_per_level = max_steps // len(self.abstraction_levels)
        
        level_idx = min(
            step_idx // steps_per_level,
            len(self.abstraction_levels) - 1
        )
        new_level = self.abstraction_levels[level_idx]
        
        # Update level if changed
        if new_level != current_level:
            state.meta['current_level'] = new_level
            state.meta['level_idx'] = level_idx
            
            # Store current level's consolidation
            self.memory_layers[current_level] = memory.detach().clone()
        
        # Apply augmentation for pattern extraction
        aug_type = self.augmentation_types[step_idx % len(self.augmentation_types)]
        augmented = self.augment_memory(memory, aug_type)
        
        # Pass through appropriate abstractor
        abstractor = self.abstractors[new_level]
        with torch.no_grad():
            refined = abstractor(augmented)
            
            # Blend with original (momentum-based update)
            alpha = self.consolidation_rate
            if new_level in ['conceptual', 'strategic']:
                # These levels output smaller dimensions
                # Take only the refined output
                new_memory = refined
            else:
                new_memory = (1 - alpha) * memory + alpha * refined
        
        # Store in database periodically
        if step_idx % 10 == 0:
            self._store_to_database(new_memory, new_level, state.meta)
        
        # Create new state
        new_state = IRPState(
            x=new_memory,
            step_idx=step_idx + 1,
            energy_val=None,
            meta=state.meta
        )
        
        return new_state
    
    def _store_to_database(self, memory: torch.Tensor, level: str, meta: Dict):
        """Store consolidated memory to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert tensor to bytes
        memory_bytes = memory.cpu().numpy().tobytes()
        
        # Prepare metadata
        metadata = {
            'level': level,
            'shape': list(memory.shape),
            'compression_ratio': meta['compression_history'][-1] if meta['compression_history'] else 1.0,
            'retrieval_accuracy': meta['retrieval_accuracy_history'][-1] if meta['retrieval_accuracy_history'] else 0.0
        }
        
        cursor.execute('''
            INSERT INTO memories (timestamp, level, content, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().timestamp(),
            level,
            memory_bytes,
            json.dumps(metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def halt(self, history: List[IRPState]) -> bool:
        """
        Halt when all abstraction levels processed or convergence achieved.
        
        Args:
            history: Consolidation history
            
        Returns:
            True if should halt
        """
        if not history:
            return False
        
        current_state = history[-1]
        
        # Check if reached highest abstraction level
        if current_state.meta['level_idx'] >= len(self.abstraction_levels) - 1:
            # At strategic level, check convergence
            if len(history) >= 5:
                # Check if consolidation has stabilized
                recent_accuracies = current_state.meta['retrieval_accuracy_history'][-5:]
                if len(recent_accuracies) >= 5:
                    accuracy_variance = np.var(recent_accuracies)
                    if accuracy_variance < 0.001:
                        return True
        
        return super().halt(history)
    
    def get_consolidated_memory(self, state: IRPState) -> Dict[str, Any]:
        """
        Extract consolidated memory hierarchy.
        
        Args:
            state: Final consolidation state
            
        Returns:
            Dictionary with memory layers and metrics
        """
        # Add final level to layers
        self.memory_layers[state.meta['current_level']] = state.x
        
        # Compute value attribution
        value_created = 0.0
        if state.meta['compression_history'] and state.meta['retrieval_accuracy_history']:
            initial_value = state.meta['compression_history'][0] * state.meta['retrieval_accuracy_history'][0]
            final_value = state.meta['compression_history'][-1] * state.meta['retrieval_accuracy_history'][-1]
            value_created = final_value - initial_value
        
        return {
            'memory_layers': {
                level: layer.cpu().numpy() if level in self.memory_layers else None
                for level in self.abstraction_levels
            },
            'final_level': state.meta['current_level'],
            'compression_achieved': state.meta['compression_history'][-1] if state.meta['compression_history'] else 1.0,
            'retrieval_accuracy': state.meta['retrieval_accuracy_history'][-1] if state.meta['retrieval_accuracy_history'] else 0.0,
            'consolidation_steps': state.step_idx,
            'value_created': value_created,
            'augmentations_applied': len(self.augmentation_types) * (state.step_idx // len(self.augmentation_types))
        }
    
    def retrieve(self, query: torch.Tensor, level: str = 'semantic') -> torch.Tensor:
        """
        Retrieve memories at specified abstraction level.
        
        Args:
            query: Query vector
            level: Abstraction level to retrieve from
            
        Returns:
            Retrieved memory
        """
        if level in self.memory_layers:
            memory = self.memory_layers[level]
            
            # Simple similarity-based retrieval
            similarities = torch.cosine_similarity(query.unsqueeze(1), memory, dim=-1)
            best_idx = torch.argmax(similarities)
            
            return memory[best_idx]
        else:
            # Return zero memory if level not available
            return torch.zeros(self.memory_dim).to(self.device)