"""
SAGE V2: A Working Baseline that Actually Learns
Integrates LLM guidance, proper context encoding, and improved training objectives.

This version addresses all the issues that caused Agent Zero:
1. External LLM for conceptual understanding
2. Meaningful context encoding (not random vectors)
3. Training objectives that reward pattern solving
4. Sufficient parameters for emergence (100M+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import logging
from dataclasses import dataclass

# Import our new components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.external_llm import ExternalLLMInterface
from context.context_encoder import MultiModalContextEncoder
from training.improved_objectives import CombinedSAGELoss

logger = logging.getLogger(__name__)


@dataclass
class SAGEV2Config:
    """Configuration for SAGE V2."""
    # Model dimensions
    hidden_size: int = 768
    num_h_layers: int = 8
    num_l_layers: int = 8
    num_heads: int = 12
    intermediate_size: int = 3072
    
    # LLM integration
    use_external_llm: bool = True
    llm_model: str = "microsoft/phi-2"
    llm_threshold: float = 0.7
    
    # Context encoding
    use_meaningful_context: bool = True
    use_temporal_context: bool = True
    use_memory_context: bool = True
    
    # Training
    dropout: float = 0.1
    max_sequence_length: int = 1024
    vocab_size: int = 10  # For ARC tasks (10 colors)
    
    # H↔L communication
    bidirectional_layers: int = 2
    communication_hidden: int = 512
    
    # Halt mechanism
    use_adaptive_halt: bool = True
    halt_threshold: float = 0.01
    max_halt_steps: int = 10


class ImprovedHModule(nn.Module):
    """
    Strategic reasoning module with LLM-guided understanding.
    Understands context and guides tactical execution.
    """
    
    def __init__(self, config: SAGEV2Config):
        super().__init__()
        self.config = config
        
        # Context encoder for meaningful representations
        self.context_encoder = MultiModalContextEncoder(
            hidden_dim=config.hidden_size,
            use_llm_context=config.use_external_llm,
            use_temporal=config.use_temporal_context,
            use_memory=config.use_memory_context
        )
        
        # Transformer layers for strategic reasoning
        # Ensure num_heads divides hidden_size
        actual_num_heads = config.num_heads
        while config.hidden_size % actual_num_heads != 0 and actual_num_heads > 1:
            actual_num_heads -= 1
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=actual_num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(config.num_h_layers)
        ])
        
        # Layer normalization
        self.ln = nn.LayerNorm(config.hidden_size)
        
        # Halt mechanism
        if config.use_adaptive_halt:
            self.halt_predictor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        input_grid: torch.Tensor,
        llm_context: Optional[torch.Tensor] = None,
        temporal_context: Optional[List[torch.Tensor]] = None,
        memory_context: Optional[torch.Tensor] = None,
        l_feedback: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with context understanding.
        
        Args:
            input_grid: Visual input
            llm_context: Language understanding from LLM
            temporal_context: Previous states
            memory_context: Retrieved memories
            l_feedback: Feedback from L-module
            
        Returns:
            Dictionary with hidden states and halt probabilities
        """
        # Encode meaningful context
        context = self.context_encoder(
            input_grid,
            llm_context=llm_context,
            temporal_context=temporal_context,
            memory_context=memory_context
        )
        
        # Add L-module feedback if available
        if l_feedback is not None:
            context = context + 0.3 * l_feedback
        
        # Prepare for transformer (add sequence dimension)
        x = context.unsqueeze(1)  # [B, 1, D]
        
        # Apply transformer layers with optional halting
        hidden_states = []
        halt_probs = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            hidden_states.append(x)
            
            if self.config.use_adaptive_halt:
                halt_prob = self.halt_predictor(x.squeeze(1))
                halt_probs.append(halt_prob)
                
                # Stop if halt probability is high enough
                if halt_prob.mean() > self.config.halt_threshold:
                    break
        
        # Final normalization
        output = self.ln(x)
        
        return {
            'hidden': output.squeeze(1),
            'hidden_states': hidden_states,
            'halt_probs': halt_probs if halt_probs else None,
            'context': context
        }


class ImprovedLModule(nn.Module):
    """
    Tactical execution module that implements strategies from H-module.
    Generates actual solutions based on strategic guidance.
    """
    
    def __init__(self, config: SAGEV2Config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(
            config.vocab_size,
            config.hidden_size
        )
        
        # Transformer layers for tactical execution
        # Ensure num_heads divides hidden_size
        actual_num_heads = config.num_heads
        while config.hidden_size % actual_num_heads != 0 and actual_num_heads > 1:
            actual_num_heads -= 1
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=actual_num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(config.num_l_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.vocab_size)
        )
        
        # Layer norm
        self.ln = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self,
        input_features: torch.Tensor,
        h_guidance: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Execute tactical solution based on strategic guidance.
        
        Args:
            input_features: Encoded input features
            h_guidance: Strategic guidance from H-module
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with predictions and hidden states
        """
        # Project input
        if input_features.dim() == 3:
            # Already has sequence dimension
            x = input_features
        else:
            # Add sequence dimension
            x = input_features.unsqueeze(1)
        
        if x.shape[-1] != self.config.hidden_size:
            x = self.input_projection(x)
        
        # Add strategic guidance
        x = x + h_guidance.unsqueeze(1) * 0.5
        
        # Apply transformer layers
        hidden_states = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
            hidden_states.append(x)
        
        # Final normalization
        x = self.ln(x)
        
        # Generate output predictions
        logits = self.output_head(x)
        
        return {
            'logits': logits,
            'hidden': x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1),
            'hidden_states': hidden_states
        }


class BidirectionalCommunicationV2(nn.Module):
    """
    Enhanced bidirectional communication between H and L modules.
    Enables iterative refinement through multiple rounds.
    """
    
    def __init__(self, config: SAGEV2Config):
        super().__init__()
        
        # H → L communication
        self.h_to_l = nn.Sequential(
            nn.Linear(config.hidden_size, config.communication_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.communication_hidden, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # L → H feedback
        self.l_to_h = nn.Sequential(
            nn.Linear(config.hidden_size, config.communication_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.communication_hidden, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Gating mechanisms
        self.h_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        self.l_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        h_state: torch.Tensor,
        l_state: torch.Tensor,
        round_num: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exchange information between modules.
        
        Args:
            h_state: H-module state
            l_state: L-module state
            round_num: Current communication round
            
        Returns:
            Updated (h_message, l_message)
        """
        # Generate messages
        h_message = self.h_to_l(h_state)
        l_feedback = self.l_to_h(l_state)
        
        # Apply gating based on round number
        if round_num > 0:
            # Gate messages based on previous states
            h_gate = self.h_gate(torch.cat([h_state, l_feedback], dim=-1))
            l_gate = self.l_gate(torch.cat([l_state, h_message], dim=-1))
            
            h_message = h_message * h_gate
            l_feedback = l_feedback * l_gate
        
        return h_message, l_feedback


class SAGEV2Core(nn.Module):
    """
    SAGE V2: Complete architecture with all improvements.
    This version should actually learn patterns instead of outputting zeros.
    """
    
    def __init__(self, config: SAGEV2Config):
        super().__init__()
        self.config = config
        
        # Core modules
        self.h_module = ImprovedHModule(config)
        self.l_module = ImprovedLModule(config)
        self.communication = BidirectionalCommunicationV2(config)
        
        # External LLM for conceptual understanding
        if config.use_external_llm:
            self.llm = ExternalLLMInterface(
                model_name=config.llm_model,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.llm.set_sage_dimensions(config.hidden_size)
        else:
            self.llm = None
        
        # Memory bank for storing experiences
        self.memory_bank = []
        self.max_memory_size = 100
        
        # Loss function
        self.loss_fn = CombinedSAGELoss()
        
        logger.info(f"Initialized SAGE V2 with {self._count_parameters()}M parameters")
    
    def _count_parameters(self) -> float:
        """Count total parameters in millions."""
        total = sum(p.numel() for p in self.parameters())
        return total / 1e6
    
    def forward(
        self,
        input_grid: torch.Tensor,
        target_grid: Optional[torch.Tensor] = None,
        num_rounds: int = 3,
        return_all_rounds: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass with iterative refinement.
        
        Args:
            input_grid: Input pattern [B, H, W]
            target_grid: Target pattern (for training)
            num_rounds: Number of H↔L communication rounds
            return_all_rounds: Whether to return results from all rounds
            
        Returns:
            Dictionary with predictions and auxiliary outputs
        """
        batch_size = input_grid.shape[0]
        device = input_grid.device
        
        # Get LLM understanding if available
        llm_context = None
        llm_reasoning = None
        
        if self.llm is not None and self.config.use_external_llm:
            try:
                llm_context, llm_reasoning = self.llm.understand_pattern(
                    input_grid,
                    return_reasoning=True
                )
                llm_context = llm_context.to(device)
            except Exception as e:
                logger.warning(f"LLM failed: {e}. Proceeding without LLM context.")
                llm_context = None
        
        # Get temporal context from memory
        temporal_context = None
        if len(self.memory_bank) > 0:
            # Use last 3 memories as temporal context
            recent_memories = self.memory_bank[-3:]
            temporal_context = [m['h_state'] for m in recent_memories]
        
        # Initial H-module processing
        h_output = self.h_module(
            input_grid,
            llm_context=llm_context,
            temporal_context=temporal_context
        )
        
        # Iterative refinement through H↔L communication
        all_predictions = []
        h_state = h_output['hidden']
        l_state = torch.zeros_like(h_state)
        
        for round_num in range(num_rounds):
            # Communication
            h_guidance, l_feedback = self.communication(
                h_state, l_state, round_num
            )
            
            # L-module execution with H guidance
            # Convert input grid to features
            if input_grid.dim() == 3:
                input_features = F.one_hot(
                    input_grid.long(),
                    num_classes=self.config.vocab_size
                ).float()
            else:
                input_features = input_grid
            
            l_output = self.l_module(
                input_features,
                h_guidance
            )
            
            # Update states
            l_state = l_output['hidden']
            
            # H-module refinement with L feedback
            if round_num < num_rounds - 1:
                h_output = self.h_module(
                    input_grid,
                    llm_context=llm_context,
                    l_feedback=l_feedback
                )
                h_state = h_output['hidden']
            
            all_predictions.append(l_output['logits'])
        
        # Store in memory for future temporal context
        if len(self.memory_bank) >= self.max_memory_size:
            self.memory_bank.pop(0)
        
        self.memory_bank.append({
            'input': input_grid,
            'h_state': h_state.detach(),
            'l_state': l_state.detach()
        })
        
        # Prepare output
        final_prediction = all_predictions[-1]
        
        output = {
            'logits': final_prediction,
            'h_hidden': h_state,
            'l_hidden': l_state,
            'llm_reasoning': llm_reasoning,
            'halt_probs': h_output.get('halt_probs'),
        }
        
        if return_all_rounds:
            output['all_predictions'] = all_predictions
        
        # Compute loss if target provided
        if target_grid is not None:
            losses = self.loss_fn(
                final_prediction,
                target_grid,
                inputs=input_grid,
                features={'anchor': h_state, 'positive': l_state}
            )
            output['loss'] = losses['total']
            output['loss_components'] = losses
        
        return output
    
    def predict(
        self,
        input_grid: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate prediction for input grid.
        
        Args:
            input_grid: Input pattern
            temperature: Sampling temperature
            
        Returns:
            Predicted grid
        """
        with torch.no_grad():
            output = self.forward(input_grid)
            logits = output['logits']
            
            if temperature > 0:
                # Sample from distribution
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Handle different shapes
                if probs.dim() == 4:  # [B, H, W, C]
                    B, H, W, C = probs.shape
                    probs_flat = probs.reshape(B * H * W, C)
                    samples = torch.multinomial(probs_flat, 1)
                    prediction = samples.reshape(B, H, W)
                else:  # [B, 1, C] or similar
                    prediction = torch.multinomial(probs.squeeze(), 1)
            else:
                # Greedy decoding
                prediction = logits.argmax(dim=-1)
        
        return prediction


def create_sage_v2(
    config: Optional[SAGEV2Config] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    load_checkpoint: Optional[str] = None
) -> SAGEV2Core:
    """
    Create SAGE V2 model.
    
    Args:
        config: Model configuration
        device: Device to use
        load_checkpoint: Path to checkpoint to load
        
    Returns:
        SAGE V2 model
    """
    if config is None:
        config = SAGEV2Config()
    
    model = SAGEV2Core(config).to(device)
    
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {load_checkpoint}")
    
    return model


if __name__ == "__main__":
    print("Testing SAGE V2...")
    
    # Create model with small config for testing
    test_config = SAGEV2Config(
        hidden_size=256,
        num_h_layers=2,
        num_l_layers=2,
        num_heads=4,
        intermediate_size=512,
        use_external_llm=False  # Disable for quick test
    )
    
    model = create_sage_v2(test_config, device='cpu')
    
    # Test input
    batch_size = 2
    input_grid = torch.randint(0, 10, (batch_size, 15, 15))
    target_grid = torch.randint(0, 10, (batch_size, 15, 15))
    
    # Forward pass
    output = model(input_grid, target_grid)
    
    print(f"✅ Model output shape: {output['logits'].shape}")
    print(f"✅ Loss computed: {output['loss'].item():.4f}")
    print(f"✅ Parameters: {model._count_parameters():.2f}M")
    
    # Test prediction
    prediction = model.predict(input_grid)
    print(f"✅ Prediction shape: {prediction.shape}")
    
    print("\nSAGE V2 ready for training! This version should actually learn patterns.")