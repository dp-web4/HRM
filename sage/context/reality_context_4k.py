"""
Reality Context Encoder - 4K Dimensional Implementation
First concrete step towards reality-scale context understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RealityContext4K:
    """
    4096-dimensional context representation for reality.
    Structured into logical groups for interpretability.
    """
    # Sensory dimensions (1536 total)
    visual: torch.Tensor        # 512 dims
    depth: torch.Tensor         # 256 dims  
    audio: torch.Tensor         # 256 dims
    tactile: torch.Tensor       # 256 dims
    proprioceptive: torch.Tensor # 256 dims
    
    # Semantic dimensions (1024 total)
    objects: torch.Tensor       # 256 dims - what things are
    affordances: torch.Tensor   # 256 dims - what can be done
    relationships: torch.Tensor # 256 dims - how things relate
    intentions: torch.Tensor    # 256 dims - goals and purposes
    
    # Physical dimensions (768 total)
    dynamics: torch.Tensor      # 256 dims - motion, forces
    materials: torch.Tensor     # 256 dims - properties
    constraints: torch.Tensor   # 256 dims - possibilities
    
    # Temporal dimensions (768 total)
    immediate: torch.Tensor     # 256 dims - current state
    historical: torch.Tensor    # 256 dims - recent past
    predictive: torch.Tensor    # 256 dims - near future
    
    def to_tensor(self) -> torch.Tensor:
        """Concatenate all dimensions into single 4096D tensor."""
        return torch.cat([
            self.visual, self.depth, self.audio, self.tactile, self.proprioceptive,
            self.objects, self.affordances, self.relationships, self.intentions,
            self.dynamics, self.materials, self.constraints,
            self.immediate, self.historical, self.predictive
        ], dim=-1)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'RealityContext4K':
        """Reconstruct structured context from 4096D tensor."""
        assert tensor.shape[-1] == 4096, f"Expected 4096 dims, got {tensor.shape[-1]}"
        
        # Split tensor back into components
        splits = [512, 256, 256, 256, 256,  # Sensory (1536)
                  256, 256, 256, 256,        # Semantic (1024)
                  256, 256, 256,              # Physical (768)
                  256, 256, 256]              # Temporal (768)
        
        components = torch.split(tensor, splits, dim=-1)
        
        return cls(
            visual=components[0], depth=components[1], audio=components[2],
            tactile=components[3], proprioceptive=components[4],
            objects=components[5], affordances=components[6],
            relationships=components[7], intentions=components[8],
            dynamics=components[9], materials=components[10],
            constraints=components[11], immediate=components[12],
            historical=components[13], predictive=components[14]
        )


class SensoryEncoder(nn.Module):
    """Encode raw sensory inputs into structured representations."""
    
    def __init__(self):
        super().__init__()
        
        # Visual encoder (expecting image input)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512)
        )
        
        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256)
        )
        
        # Audio encoder (expecting spectrogram)
        self.audio_encoder = nn.Sequential(
            nn.Linear(1024, 512),  # Simplified for now
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Tactile encoder (force/pressure readings)
        self.tactile_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Proprioceptive encoder (joint states)
        self.proprioceptive_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode raw sensory inputs."""
        encoded = {}
        
        if 'visual' in observations:
            encoded['visual'] = self.visual_encoder(observations['visual'])
        else:
            encoded['visual'] = torch.zeros(observations.get('batch_size', 1), 512)
            
        if 'depth' in observations:
            encoded['depth'] = self.depth_encoder(observations['depth'])
        else:
            encoded['depth'] = torch.zeros(observations.get('batch_size', 1), 256)
            
        if 'audio' in observations:
            encoded['audio'] = self.audio_encoder(observations['audio'])
        else:
            encoded['audio'] = torch.zeros(observations.get('batch_size', 1), 256)
            
        if 'tactile' in observations:
            encoded['tactile'] = self.tactile_encoder(observations['tactile'])
        else:
            encoded['tactile'] = torch.zeros(observations.get('batch_size', 1), 256)
            
        if 'proprioceptive' in observations:
            encoded['proprioceptive'] = self.proprioceptive_encoder(observations['proprioceptive'])
        else:
            encoded['proprioceptive'] = torch.zeros(observations.get('batch_size', 1), 256)
        
        return encoded


class SemanticEncoder(nn.Module):
    """Extract semantic understanding from sensory features."""
    
    def __init__(self):
        super().__init__()
        
        # Takes concatenated sensory features as input
        input_dim = 1536  # All sensory dimensions
        
        # Object recognition
        self.object_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Affordance detection
        self.affordance_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Relationship understanding
        self.relationship_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Intention inference
        self.intention_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, sensory_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract semantic features from sensory input."""
        return {
            'objects': self.object_encoder(sensory_features),
            'affordances': self.affordance_encoder(sensory_features),
            'relationships': self.relationship_encoder(sensory_features),
            'intentions': self.intention_encoder(sensory_features)
        }


class PhysicalEncoder(nn.Module):
    """Understand physical properties and constraints."""
    
    def __init__(self):
        super().__init__()
        
        # Combines sensory and semantic features
        input_dim = 1536 + 1024  # Sensory + Semantic
        
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.materials_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.constraints_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, combined_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract physical understanding."""
        return {
            'dynamics': self.dynamics_encoder(combined_features),
            'materials': self.materials_encoder(combined_features),
            'constraints': self.constraints_encoder(combined_features)
        }


class TemporalEncoder(nn.Module):
    """Encode temporal context across multiple timescales."""
    
    def __init__(self):
        super().__init__()
        
        # Full context dimension
        input_dim = 1536 + 1024 + 768  # All non-temporal features
        
        # Immediate state
        self.immediate_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Historical context (LSTM for sequence)
        self.historical_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Predictive model
        self.predictive_encoder = nn.Sequential(
            nn.Linear(input_dim + 256, 512),  # Current + historical
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(
        self, 
        current_features: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode temporal context."""
        
        # Immediate is just current state
        immediate = self.immediate_encoder(current_features)
        
        # Historical processing
        if history and len(history) > 0:
            # Stack history into sequence
            hist_sequence = torch.stack(history, dim=1)
            _, (h_n, _) = self.historical_lstm(hist_sequence)
            # Combine forward and backward final hidden states
            historical = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            # No history available
            historical = torch.zeros_like(immediate)
        
        # Predictive based on current + historical
        pred_input = torch.cat([current_features, historical], dim=-1)
        predictive = self.predictive_encoder(pred_input)
        
        return {
            'immediate': immediate,
            'historical': historical,
            'predictive': predictive
        }


class RealityContextEncoder4K(nn.Module):
    """
    Main 4K dimensional reality context encoder.
    Combines all sub-encoders into unified context representation.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sub-encoders
        self.sensory_encoder = SensoryEncoder()
        self.semantic_encoder = SemanticEncoder()
        self.physical_encoder = PhysicalEncoder()
        self.temporal_encoder = TemporalEncoder()
        
        # Optional: learnable blending weights
        self.context_norm = nn.LayerNorm(4096)
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        history: Optional[List[torch.Tensor]] = None
    ) -> RealityContext4K:
        """
        Encode observations into 4K dimensional context.
        
        Args:
            observations: Dict with keys like 'visual', 'depth', 'audio', etc.
            history: Optional list of previous context vectors
            
        Returns:
            RealityContext4K object with all dimensions filled
        """
        # 1. Encode sensory inputs
        sensory = self.sensory_encoder(observations)
        sensory_concat = torch.cat(list(sensory.values()), dim=-1)
        
        # 2. Extract semantic understanding
        semantic = self.semantic_encoder(sensory_concat)
        semantic_concat = torch.cat(list(semantic.values()), dim=-1)
        
        # 3. Understand physical properties
        sensory_semantic = torch.cat([sensory_concat, semantic_concat], dim=-1)
        physical = self.physical_encoder(sensory_semantic)
        physical_concat = torch.cat(list(physical.values()), dim=-1)
        
        # 4. Encode temporal context
        non_temporal = torch.cat([sensory_concat, semantic_concat, physical_concat], dim=-1)
        temporal = self.temporal_encoder(non_temporal, history)
        
        # 5. Create structured context
        context = RealityContext4K(
            # Sensory
            visual=sensory['visual'],
            depth=sensory['depth'],
            audio=sensory['audio'],
            tactile=sensory['tactile'],
            proprioceptive=sensory['proprioceptive'],
            # Semantic
            objects=semantic['objects'],
            affordances=semantic['affordances'],
            relationships=semantic['relationships'],
            intentions=semantic['intentions'],
            # Physical
            dynamics=physical['dynamics'],
            materials=physical['materials'],
            constraints=physical['constraints'],
            # Temporal
            immediate=temporal['immediate'],
            historical=temporal['historical'],
            predictive=temporal['predictive']
        )
        
        # Optional: normalize the full context
        full_tensor = context.to_tensor()
        normalized = self.context_norm(full_tensor)
        
        return RealityContext4K.from_tensor(normalized)


if __name__ == "__main__":
    print("Testing Reality Context 4K Encoder...")
    
    # Create encoder
    encoder = RealityContextEncoder4K()
    
    # Create dummy observations
    batch_size = 2
    observations = {
        'visual': torch.randn(batch_size, 3, 224, 224),
        'depth': torch.randn(batch_size, 1, 224, 224),
        'audio': torch.randn(batch_size, 1024),
        'tactile': torch.randn(batch_size, 128),
        'proprioceptive': torch.randn(batch_size, 64),
        'batch_size': batch_size
    }
    
    # Encode to 4K context
    context = encoder(observations)
    
    # Check dimensions
    full_tensor = context.to_tensor()
    print(f"✅ Context tensor shape: {full_tensor.shape}")
    assert full_tensor.shape == (batch_size, 4096), "Wrong context dimensions!"
    
    # Check structure
    print(f"✅ Visual dims: {context.visual.shape}")
    print(f"✅ Semantic object dims: {context.objects.shape}")
    print(f"✅ Physical dynamics dims: {context.dynamics.shape}")
    print(f"✅ Temporal immediate dims: {context.immediate.shape}")
    
    # Test with history
    history = [full_tensor.clone() for _ in range(3)]
    context_with_history = encoder(observations, history)
    print(f"✅ Context with history shape: {context_with_history.to_tensor().shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n📊 Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    print("\n✅ Reality Context 4K Encoder ready!")