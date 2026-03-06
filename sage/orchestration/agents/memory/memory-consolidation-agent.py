#!/usr/bin/env python3
"""
Memory Consolidation Agent
Implements sleep-cycle pattern extraction and memory consolidation for SAGE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
import time
import json
import sqlite3
from pathlib import Path
import threading
import queue


@dataclass
class Experience:
    """Single experience to be consolidated"""
    timestamp: float
    source: str
    features: np.ndarray
    action: Optional[np.ndarray]
    reward: float
    salience: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['features'] = data['features'].tolist()
        if data['action'] is not None:
            data['action'] = data['action'].tolist()
        return data


class DreamAugmentor:
    """
    Augments experiences during sleep/dream cycles
    Implements the key insight: augmentation = wisdom extraction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.augmentation_factor = self.config.get("augmentation_factor", 4)
        self.noise_scale = self.config.get("noise_scale", 0.1)
        self.permutation_prob = self.config.get("permutation_prob", 0.3)
        
    def augment_experience(self, exp: Experience) -> List[Experience]:
        """
        Generate augmented versions of an experience
        This is the dream process - creating variations to extract patterns
        """
        augmented = [exp]  # Original
        
        for i in range(self.augmentation_factor - 1):
            aug_exp = self._create_variation(exp, i)
            augmented.append(aug_exp)
        
        return augmented
    
    def _create_variation(self, exp: Experience, variation_id: int) -> Experience:
        """Create a single variation of an experience"""
        
        # Clone the experience
        features = exp.features.copy()
        action = exp.action.copy() if exp.action is not None else None
        
        # Apply transformations based on variation type
        if variation_id % 4 == 0:
            # Add noise (small perturbations)
            features += np.random.randn(*features.shape) * self.noise_scale
            
        elif variation_id % 4 == 1:
            # Permute dimensions (reorder features)
            if np.random.random() < self.permutation_prob:
                perm = np.random.permutation(len(features))
                features = features[perm]
                
        elif variation_id % 4 == 2:
            # Scale variation (amplitude changes)
            scale = np.random.uniform(0.8, 1.2)
            features *= scale
            
        else:
            # Temporal shift (simulate different timing)
            shift = np.random.randint(-5, 6)
            features = np.roll(features, shift)
        
        # Create augmented experience
        return Experience(
            timestamp=exp.timestamp,
            source=exp.source,
            features=features,
            action=action,
            reward=exp.reward * 0.9,  # Slightly discount augmented rewards
            salience=exp.salience * 0.95,  # Slightly reduce salience
            metadata={**exp.metadata, "augmented": True, "variation": variation_id}
        )


class PatternExtractor(nn.Module):
    """
    Neural network for extracting persistent patterns from experiences
    This is the wisdom distillation component
    """
    
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, 
                 pattern_dim: int = 128):
        super().__init__()
        
        # Encoder: compress experiences to patterns
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pattern_dim)
        )
        
        # Decoder: reconstruct from patterns
        self.decoder = nn.Sequential(
            nn.Linear(pattern_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.pattern_dim = pattern_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: (patterns, reconstructed)
        """
        patterns = self.encoder(x)
        reconstructed = self.decoder(patterns)
        return patterns, reconstructed
    
    def extract_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pattern representation only"""
        with torch.no_grad():
            return self.encoder(x)


class MemoryConsolidationAgent:
    """
    Manages memory consolidation through sleep cycles
    Implements dual storage: fast (buffer) and slow (consolidated)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Consolidation parameters
        self.buffer_size = self.config.get("buffer_size", 1000)
        self.consolidation_interval = self.config.get("consolidation_interval", 100)
        self.min_salience = self.config.get("min_salience", 0.3)
        self.compression_ratio = self.config.get("compression_ratio", 0.1)
        
        # Components
        self.augmentor = DreamAugmentor(self.config)
        self.pattern_extractor = PatternExtractor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pattern_extractor.to(self.device)
        
        # Storage
        self.fast_buffer = deque(maxlen=self.buffer_size)
        self.consolidated_patterns = []
        self.db_path = Path(self.config.get("db_path", 
                                           "/home/dp/ai-workspace/HRM/sage/orchestration/memory/consolidated.db"))
        self._init_database()
        
        # State
        self.total_experiences = 0
        self.total_consolidated = 0
        self.is_consolidating = False
        self.consolidation_thread = None
        
        # Training
        self.optimizer = torch.optim.Adam(
            self.pattern_extractor.parameters(),
            lr=1e-3
        )
        
        print("💤 Memory Consolidation Agent initialized")
        print(f"   Buffer size: {self.buffer_size}")
        print(f"   Consolidation interval: {self.consolidation_interval}")
        print(f"   Compression ratio: {self.compression_ratio}")
    
    def _init_database(self):
        """Initialize SQLite database for consolidated memories"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                pattern BLOB,
                source TEXT,
                salience REAL,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_experience(self, experience: Experience):
        """Add experience to fast buffer"""
        self.fast_buffer.append(experience)
        self.total_experiences += 1
        
        # Trigger consolidation if needed
        if (self.total_experiences % self.consolidation_interval == 0 and 
            not self.is_consolidating):
            self.consolidate_async()
    
    def consolidate_async(self):
        """Start asynchronous consolidation (sleep cycle)"""
        if not self.is_consolidating:
            self.is_consolidating = True
            self.consolidation_thread = threading.Thread(
                target=self._consolidation_cycle
            )
            self.consolidation_thread.start()
            print("😴 Starting consolidation cycle...")
    
    def _consolidation_cycle(self):
        """
        Main consolidation cycle (sleep/dream process)
        1. Select salient experiences
        2. Augment them (dream variations)
        3. Extract patterns (wisdom)
        4. Store compressed representations
        """
        try:
            # Select experiences to consolidate
            experiences = list(self.fast_buffer)
            salient_experiences = [
                exp for exp in experiences 
                if exp.salience >= self.min_salience
            ]
            
            if not salient_experiences:
                print("  No salient experiences to consolidate")
                return
            
            print(f"  Consolidating {len(salient_experiences)} experiences...")
            
            # Augment experiences (dreaming)
            all_augmented = []
            for exp in salient_experiences:
                augmented = self.augmentor.augment_experience(exp)
                all_augmented.extend(augmented)
            
            print(f"  Generated {len(all_augmented)} augmented experiences")
            
            # Extract patterns (wisdom extraction)
            patterns = self._extract_and_train_patterns(all_augmented)
            
            # Store consolidated patterns
            self._store_patterns(patterns, salient_experiences)
            
            # Update statistics
            self.total_consolidated += len(patterns)
            
            print(f"  ✅ Consolidated {len(patterns)} patterns")
            print(f"  Compression ratio: {len(patterns)/len(all_augmented):.2%}")
            
        except Exception as e:
            print(f"  ❌ Consolidation error: {e}")
        finally:
            self.is_consolidating = False
    
    def _extract_and_train_patterns(self, experiences: List[Experience]) -> List[np.ndarray]:
        """
        Extract patterns from experiences and train the pattern extractor
        """
        # Convert to tensors
        features = torch.stack([
            torch.from_numpy(exp.features).float()
            for exp in experiences
        ]).to(self.device)
        
        # Train pattern extractor (autoencoder)
        self.pattern_extractor.train()
        
        num_epochs = 10
        batch_size = 32
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                
                # Forward pass
                patterns, reconstructed = self.pattern_extractor(batch)
                
                # Reconstruction loss
                loss = F.mse_loss(reconstructed, batch)
                
                # Add sparsity constraint on patterns
                sparsity_loss = 0.01 * torch.mean(torch.abs(patterns))
                loss += sparsity_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 5 == 0:
                avg_loss = total_loss / num_batches
                print(f"    Epoch {epoch}: loss={avg_loss:.4f}")
        
        # Extract final patterns
        self.pattern_extractor.eval()
        with torch.no_grad():
            all_patterns = self.pattern_extractor.extract_pattern(features)
        
        # Select most representative patterns (compression)
        num_patterns = int(len(experiences) * self.compression_ratio)
        
        # Use k-means style selection for diversity
        selected_indices = self._select_diverse_patterns(all_patterns, num_patterns)
        selected_patterns = all_patterns[selected_indices].cpu().numpy()
        
        return [selected_patterns[i] for i in range(len(selected_patterns))]
    
    def _select_diverse_patterns(self, patterns: torch.Tensor, k: int) -> List[int]:
        """Select k most diverse patterns"""
        if k >= len(patterns):
            return list(range(len(patterns)))
        
        # Simple furthest point sampling
        selected = [0]  # Start with first pattern
        
        while len(selected) < k:
            # Compute distances to all selected patterns
            min_distances = torch.full((len(patterns),), float('inf')).to(self.device)
            
            for idx in selected:
                distances = torch.norm(patterns - patterns[idx:idx+1], dim=1)
                min_distances = torch.minimum(min_distances, distances)
            
            # Select pattern with maximum minimum distance
            next_idx = torch.argmax(min_distances).item()
            selected.append(next_idx)
        
        return selected
    
    def _store_patterns(self, patterns: List[np.ndarray], source_experiences: List[Experience]):
        """Store consolidated patterns in database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for i, pattern in enumerate(patterns):
            # Get corresponding source experience
            source_exp = source_experiences[min(i, len(source_experiences)-1)]
            
            cursor.execute("""
                INSERT INTO consolidated_patterns 
                (timestamp, pattern, source, salience, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                time.time(),
                pattern.tobytes(),
                source_exp.source,
                source_exp.salience,
                json.dumps(source_exp.metadata)
            ))
        
        conn.commit()
        conn.close()
    
    def retrieve_patterns(self, k: int = 10) -> List[Dict]:
        """Retrieve most recent consolidated patterns"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, pattern, source, salience, metadata
            FROM consolidated_patterns
            ORDER BY timestamp DESC
            LIMIT ?
        """, (k,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "timestamp": row[0],
                "pattern": np.frombuffer(row[1], dtype=np.float32),
                "source": row[2],
                "salience": row[3],
                "metadata": json.loads(row[4])
            })
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """Get consolidation statistics"""
        return {
            "total_experiences": self.total_experiences,
            "buffer_size": len(self.fast_buffer),
            "total_consolidated": self.total_consolidated,
            "is_consolidating": self.is_consolidating,
            "compression_achieved": self.total_consolidated / max(1, self.total_experiences)
        }


def main():
    """Test the Memory Consolidation Agent"""
    print("🧪 Testing Memory Consolidation Agent")
    print("=" * 50)
    
    # Create agent
    config = {
        "buffer_size": 100,
        "consolidation_interval": 20,
        "min_salience": 0.3,
        "compression_ratio": 0.2,
        "augmentation_factor": 4
    }
    
    agent = MemoryConsolidationAgent(config)
    
    # Simulate experiences
    print("\n📊 Simulating experiences...")
    for i in range(30):
        # Create mock experience
        exp = Experience(
            timestamp=time.time(),
            source=f"sensor_{i % 3}",
            features=np.random.randn(1536),
            action=np.random.randn(7) if i % 2 == 0 else None,
            reward=np.random.random(),
            salience=np.random.random(),
            metadata={"episode": i // 10, "step": i % 10}
        )
        
        agent.add_experience(exp)
        
        if i % 10 == 0:
            print(f"  Added {i+1} experiences...")
    
    # Wait for consolidation
    time.sleep(2)
    
    # Check statistics
    print("\n📈 Statistics:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Retrieve patterns
    print("\n🔍 Retrieved patterns:")
    patterns = agent.retrieve_patterns(k=5)
    for i, pattern_data in enumerate(patterns):
        print(f"  Pattern {i}: source={pattern_data['source']}, "
              f"salience={pattern_data['salience']:.3f}, "
              f"shape={pattern_data['pattern'].shape}")
    
    print("\n✅ Memory Consolidation Agent test complete!")


if __name__ == "__main__":
    main()