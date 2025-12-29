"""
SAGE Configuration Module

Defines the configuration for the 100M parameter SAGE attention orchestrator.
This configuration ensures we hit the critical mass threshold for reasoning emergence.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class SAGEConfig:
    """Configuration for SAGE (Situation-Aware Governance Engine)
    
    Target: ~100M parameters distributed as:
    - H-module (strategic): ~45M params
    - L-module (tactical): ~45M params  
    - Interaction/context: ~10M params
    """
    
    # Core architecture dimensions
    hidden_size: int = 768  # Rich representations for complex reasoning
    intermediate_size: int = 3072  # 4x hidden size (standard transformer ratio)
    num_attention_heads: int = 12
    attention_head_dim: int = 64  # hidden_size // num_attention_heads
    
    # Layer configuration
    num_h_layers: int = 7  # Deep strategic reasoning
    num_l_layers: int = 7  # Deep tactical execution
    
    # Context and interaction
    context_dim: int = 256  # Rich context encoding
    snarc_dim: int = 5  # Surprise, Novelty, Arousal, Reward, Conflict
    num_reasoning_cycles: int = 8  # Maximum iterative refinement cycles
    
    # Sequence and position
    max_seq_length: int = 512
    max_grid_size: int = 30  # For ARC tasks
    num_classes: int = 10  # ARC color palette
    
    # Resource management
    resource_types: List[str] = None  # Set in __post_init__
    max_resource_calls: int = 10  # Per task budget
    
    # Model behavior
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Activation functions
    hidden_act: str = "gelu"  # or "relu", "silu"
    
    # Initialization
    initializer_range: float = 0.02
    
    # Training configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Batch and memory
    batch_size: int = 16  # Smaller due to 100M params
    gradient_accumulation_steps: int = 4  # Effective batch = 64
    gradient_checkpointing: bool = True  # Essential for memory efficiency
    mixed_precision: bool = True  # FP16/BF16 training
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/sage"
    checkpoint_interval: int = 10000  # Steps between checkpoints
    keep_checkpoints: int = 5  # Number of checkpoints to keep
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    
    # Input/Output
    use_patch_embedding: bool = True  # For vision inputs
    patch_size: int = 2  # Divide 30x30 into 15x15 patches
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 3
    
    def __post_init__(self):
        """Initialize computed fields"""
        if self.resource_types is None:
            self.resource_types = ['llm', 'vision', 'memory', 'time', 'effector']
        
        # Validate configuration achieves ~100M parameters
        self.validate_parameter_count()
    
    def validate_parameter_count(self):
        """Estimate and validate parameter count is ~100M"""
        # Rough parameter estimation
        params = 0
        
        # H-module transformer layers
        h_layer_params = (
            4 * self.hidden_size * self.hidden_size +  # Self-attention
            2 * self.hidden_size * self.intermediate_size  # FFN
        )
        params += self.num_h_layers * h_layer_params
        
        # L-module transformer layers  
        l_layer_params = h_layer_params  # Same architecture
        params += self.num_l_layers * l_layer_params
        
        # Context and interaction layers
        context_params = (
            self.context_dim * self.hidden_size * 2 +  # Context encoders
            self.hidden_size * self.hidden_size * 4  # H↔L interaction
        )
        params += context_params
        
        # Embeddings and heads
        embedding_params = (
            self.num_classes * self.hidden_size +  # Token embeddings
            self.max_seq_length * self.hidden_size  # Position embeddings
        )
        params += embedding_params
        
        # Resource routing and SNARC
        routing_params = (
            len(self.resource_types) * self.hidden_size +
            self.snarc_dim * self.hidden_size * 2
        )
        params += routing_params
        
        params_in_millions = params / 1_000_000
        
        if params_in_millions < 80 or params_in_millions > 120:
            print(f"Warning: Estimated {params_in_millions:.1f}M parameters. "
                  f"Target is ~100M for reasoning emergence.")
        else:
            print(f"✓ Model configuration: ~{params_in_millions:.1f}M parameters")
    
    @property
    def total_layers(self):
        """Total number of transformer layers"""
        return self.num_h_layers + self.num_l_layers
    
    def to_dict(self):
        """Convert config to dictionary"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


# Preset configurations for different scenarios
class SAGEPresets:
    """Preset configurations for common use cases"""
    
    @staticmethod
    def development():
        """Smaller config for development/debugging (25M params)"""
        return SAGEConfig(
            hidden_size=384,
            intermediate_size=1536,
            num_attention_heads=6,
            num_h_layers=4,
            num_l_layers=4,
            batch_size=32,
            gradient_checkpointing=False
        )
    
    @staticmethod
    def standard():
        """Standard 100M parameter configuration"""
        return SAGEConfig()
    
    @staticmethod
    def large():
        """Larger 200M config for enhanced capability"""
        return SAGEConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_h_layers=10,
            num_l_layers=10,
            batch_size=8,
            gradient_accumulation_steps=8
        )
    
    @staticmethod
    def jetson():
        """Optimized for Jetson Orin Nano deployment"""
        return SAGEConfig(
            hidden_size=768,
            num_h_layers=7,
            num_l_layers=7,
            batch_size=8,
            gradient_checkpointing=True,
            mixed_precision=True,
            num_workers=2  # Limited CPU cores
        )


if __name__ == "__main__":
    # Test configuration
    config = SAGEConfig()
    print("SAGE Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  H-layers: {config.num_h_layers}")
    print(f"  L-layers: {config.num_l_layers}")
    print(f"  Total layers: {config.total_layers}")
    print(f"  Context dim: {config.context_dim}")
    print(f"  Device: {config.device}")
    
    # Test presets
    print("\nTesting presets:")
    dev_config = SAGEPresets.development()
    print("Development config created")
    
    jetson_config = SAGEPresets.jetson()
    print("Jetson config created")