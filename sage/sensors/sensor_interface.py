"""
Sensor Interface Module

Defines the base interface and implementations for various sensors that
gather "attention puzzles" for SAGE to process.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class AttentionPuzzle:
    """Represents a puzzle that requires attention
    
    Attributes:
        sensor_type: Type of sensor that generated this puzzle
        data: Raw sensor data
        metadata: Additional information about the puzzle
        timestamp: When the puzzle was created
        priority: Initial priority score
        snarc_scores: Optional pre-computed SNARC scores
    """
    sensor_type: str
    data: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    priority: float = 0.5
    snarc_scores: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'sensor_type': self.sensor_type,
            'data': self.data.cpu().numpy().tolist() if isinstance(self.data, torch.Tensor) else self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'priority': self.priority
        }


class BaseSensor(ABC):
    """Abstract base class for all sensors"""
    
    def __init__(self, name: str, hidden_size: int = 768):
        self.name = name
        self.hidden_size = hidden_size
        self.active = True
        self.puzzle_count = 0
        
    @abstractmethod
    def sense(self) -> Optional[AttentionPuzzle]:
        """Gather sensor data and create attention puzzle"""
        pass
    
    @abstractmethod
    def encode(self, data: Any) -> torch.Tensor:
        """Encode sensor data into hidden representation"""
        pass
    
    def activate(self):
        """Activate the sensor"""
        self.active = True
    
    def deactivate(self):
        """Deactivate the sensor"""
        self.active = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sensor statistics"""
        return {
            'name': self.name,
            'active': self.active,
            'puzzle_count': self.puzzle_count
        }


class VisionSensor(BaseSensor):
    """Visual input sensor for processing images/grids"""
    
    def __init__(self, hidden_size: int = 768, grid_size: int = 30):
        super().__init__("vision", hidden_size)
        self.grid_size = grid_size
        
        # Vision encoder (can be replaced with VAE or other)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size)
        )
        
    def sense(self, grid: Optional[torch.Tensor] = None) -> Optional[AttentionPuzzle]:
        """Capture visual input and create puzzle
        
        Args:
            grid: Optional input grid [batch, height, width] or [height, width]
            
        Returns:
            AttentionPuzzle or None if no input
        """
        if not self.active:
            return None
        
        if grid is None:
            # Generate random grid for testing
            grid = torch.randint(0, 10, (self.grid_size, self.grid_size))
        
        # Ensure proper dimensions
        if grid.dim() == 2:
            grid = grid.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif grid.dim() == 3:
            grid = grid.unsqueeze(1)  # Add channel dim
        
        puzzle = AttentionPuzzle(
            sensor_type="vision",
            data=grid,
            metadata={
                'shape': list(grid.shape),
                'unique_values': len(torch.unique(grid)),
                'sparsity': (grid == 0).float().mean().item()
            },
            timestamp=time.time(),
            priority=0.7  # Vision typically high priority
        )
        
        self.puzzle_count += 1
        return puzzle
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode visual data into hidden representation
        
        Args:
            data: Visual input [batch, channel, height, width]
            
        Returns:
            Encoded representation [batch, hidden_size]
        """
        # Normalize to [0, 1] if needed
        if data.max() > 1:
            data = data.float() / 10.0
        else:
            data = data.float()
        
        return self.encoder(data)


class LanguageSensor(BaseSensor):
    """Language/text input sensor"""
    
    def __init__(self, hidden_size: int = 768, vocab_size: int = 50000):
        super().__init__("language", hidden_size)
        self.vocab_size = vocab_size
        
        # Simple embedding (can be replaced with LLM embeddings)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=2
        )
        
    def sense(self, text: Optional[str] = None) -> Optional[AttentionPuzzle]:
        """Capture language input and create puzzle
        
        Args:
            text: Optional text input
            
        Returns:
            AttentionPuzzle or None if no input
        """
        if not self.active:
            return None
        
        if text is None:
            return None
        
        # Tokenize (simplified - use real tokenizer in production)
        tokens = self._simple_tokenize(text)
        
        puzzle = AttentionPuzzle(
            sensor_type="language",
            data=tokens,
            metadata={
                'text': text,
                'length': len(tokens),
                'complexity': self._estimate_complexity(text)
            },
            timestamp=time.time(),
            priority=0.6
        )
        
        self.puzzle_count += 1
        return puzzle
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode text tokens into hidden representation
        
        Args:
            data: Token IDs [batch, seq_len] or [seq_len]
            
        Returns:
            Encoded representation [batch, hidden_size] or [hidden_size]
        """
        # Ensure batch dimension
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        embeddings = self.embedding(data)
        encoded = self.encoder(embeddings)
        # Pool over sequence
        result = encoded.mean(dim=1)
        
        # Remove batch dim if input was 1D
        if result.size(0) == 1:
            result = result.squeeze(0)
        
        return result
    
    def _simple_tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization (replace with real tokenizer)"""
        # Hash words to token IDs
        words = text.lower().split()
        tokens = [hash(word) % self.vocab_size for word in words]
        return torch.tensor(tokens, dtype=torch.long)
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity"""
        words = text.split()
        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / (total_words + 1)


class MemorySensor(BaseSensor):
    """Memory retrieval sensor for accessing past experiences"""
    
    def __init__(self, hidden_size: int = 768, memory_capacity: int = 10000):
        super().__init__("memory", hidden_size)
        self.memory_capacity = memory_capacity
        self.memory_store = []
        
        # Memory encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def sense(self, query: Optional[torch.Tensor] = None) -> Optional[AttentionPuzzle]:
        """Retrieve relevant memories based on query
        
        Args:
            query: Optional query vector [hidden_size]
            
        Returns:
            AttentionPuzzle containing retrieved memories
        """
        if not self.active or len(self.memory_store) == 0:
            return None
        
        if query is None:
            # Random memory retrieval
            idx = np.random.randint(0, len(self.memory_store))
            memory = self.memory_store[idx]
        else:
            # Similarity-based retrieval
            memory = self._retrieve_similar(query)
        
        puzzle = AttentionPuzzle(
            sensor_type="memory",
            data=memory['data'],
            metadata={
                'age': time.time() - memory['timestamp'],
                'access_count': memory.get('access_count', 0)
            },
            timestamp=time.time(),
            priority=0.4  # Memories typically lower priority
        )
        
        self.puzzle_count += 1
        return puzzle
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode memory data"""
        return self.encoder(data)
    
    def store_memory(self, data: torch.Tensor, metadata: Dict = None):
        """Store a new memory
        
        Args:
            data: Memory data [hidden_size]
            metadata: Optional metadata
        """
        memory = {
            'data': data.detach().cpu(),
            'timestamp': time.time(),
            'metadata': metadata or {},
            'access_count': 0
        }
        
        self.memory_store.append(memory)
        
        # Maintain capacity
        if len(self.memory_store) > self.memory_capacity:
            self.memory_store.pop(0)
    
    def _retrieve_similar(self, query: torch.Tensor) -> Dict:
        """Retrieve most similar memory to query"""
        similarities = []
        for memory in self.memory_store:
            sim = F.cosine_similarity(query, memory['data'], dim=0)
            similarities.append(sim.item())
        
        best_idx = np.argmax(similarities)
        self.memory_store[best_idx]['access_count'] += 1
        return self.memory_store[best_idx]


class TimeSensor(BaseSensor):
    """Temporal awareness sensor"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__("time", hidden_size)
        self.start_time = time.time()
        
        # Time encoder
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
    def sense(self) -> Optional[AttentionPuzzle]:
        """Capture temporal state
        
        Returns:
            AttentionPuzzle containing temporal information
        """
        if not self.active:
            return None
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Create temporal features
        time_features = torch.tensor([
            np.sin(elapsed / 10),  # Short-term cycle
            np.cos(elapsed / 10),
            np.sin(elapsed / 100),  # Long-term cycle
            np.cos(elapsed / 100)
        ], dtype=torch.float32)
        
        puzzle = AttentionPuzzle(
            sensor_type="time",
            data=time_features,
            metadata={
                'elapsed': elapsed,
                'timestamp': current_time
            },
            timestamp=current_time,
            priority=0.3  # Time usually background priority
        )
        
        self.puzzle_count += 1
        return puzzle
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode temporal features"""
        # Ensure batch dimension
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        result = self.encoder(data)
        
        # Remove batch dim if needed
        if result.size(0) == 1:
            result = result.squeeze(0)
        
        return result


class SensorHub:
    """Central hub for managing all sensors"""
    
    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size
        self.sensors = {}
        self.puzzle_queue = []
        
        # Initialize default sensors
        self.add_sensor(VisionSensor(hidden_size))
        self.add_sensor(LanguageSensor(hidden_size))
        self.add_sensor(MemorySensor(hidden_size))
        self.add_sensor(TimeSensor(hidden_size))
        
    def add_sensor(self, sensor: BaseSensor):
        """Add a sensor to the hub"""
        self.sensors[sensor.name] = sensor
    
    def gather_puzzles(self, inputs: Dict[str, Any] = None) -> List[AttentionPuzzle]:
        """Gather puzzles from all active sensors
        
        Args:
            inputs: Optional sensor-specific inputs
            
        Returns:
            List of attention puzzles
        """
        puzzles = []
        inputs = inputs or {}
        
        for name, sensor in self.sensors.items():
            if not sensor.active:
                continue
            
            # Get sensor-specific input if available
            sensor_input = inputs.get(name)
            
            # Gather puzzle based on sensor type
            if name == "vision" and sensor_input is not None:
                puzzle = sensor.sense(sensor_input)
            elif name == "language" and sensor_input is not None:
                puzzle = sensor.sense(sensor_input)
            elif name == "memory":
                puzzle = sensor.sense(sensor_input)
            elif name == "time":
                puzzle = sensor.sense()
            else:
                puzzle = None
            
            if puzzle is not None:
                puzzles.append(puzzle)
        
        return puzzles
    
    def encode_puzzle(self, puzzle: AttentionPuzzle) -> torch.Tensor:
        """Encode a puzzle using appropriate sensor encoder
        
        Args:
            puzzle: AttentionPuzzle to encode
            
        Returns:
            Encoded representation [hidden_size]
        """
        sensor = self.sensors.get(puzzle.sensor_type)
        if sensor is None:
            raise ValueError(f"Unknown sensor type: {puzzle.sensor_type}")
        
        return sensor.encode(puzzle.data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all sensors"""
        stats = {}
        for name, sensor in self.sensors.items():
            stats[name] = sensor.get_stats()
        return stats


if __name__ == "__main__":
    # Test sensor interface
    print("Testing Sensor Interface...")
    
    # Create sensor hub
    hub = SensorHub()
    
    # Test gathering puzzles
    inputs = {
        'vision': torch.randint(0, 10, (30, 30)),
        'language': "This is a test of the attention system",
        'memory': torch.randn(768)
    }
    
    puzzles = hub.gather_puzzles(inputs)
    print(f"Gathered {len(puzzles)} puzzles")
    
    for puzzle in puzzles:
        print(f"  - {puzzle.sensor_type}: priority={puzzle.priority:.2f}")
        encoded = hub.encode_puzzle(puzzle)
        print(f"    Encoded shape: {encoded.shape}")
    
    # Get sensor statistics
    stats = hub.get_stats()
    print(f"\nSensor statistics:")
    for name, stat in stats.items():
        print(f"  {name}: {stat}")
    
    print("\nSensor Interface test successful!")