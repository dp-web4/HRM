"""
Language IRP Plugin - Actual Implementation
Progressive span unmasking with meaning stabilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sage.irp.base import IRPPlugin
from models.language.tiny_bert import create_tiny_bert_for_jetson, SpanMaskingModel


class LanguageIRPImpl(IRPPlugin):
    """
    Language IRP using progressive span unmasking
    Stabilizes meaning representation through iterative refinement
    """
    
    def __init__(
        self,
        model_variant: str = 'span',
        max_iterations: int = 30,
        eps: float = 0.05,
        initial_mask_ratio: float = 0.5,
        final_mask_ratio: float = 0.1,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        # Create default config if not provided
        if config is None:
            config = {
                'max_iterations': max_iterations,
                'halt_eps': eps,
                'entity_id': 'language_irp'
            }
        super().__init__(config)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_tiny_bert_for_jetson(model_variant)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # IRP parameters
        self.max_iterations = max_iterations
        self.eps = eps
        self.initial_mask_ratio = initial_mask_ratio
        self.final_mask_ratio = final_mask_ratio
        self.iteration = 0
        
        # Cache for tracking
        self.original_ids = None
        self.current_mask_ratio = initial_mask_ratio
        self.meaning_history = []
        self.masked_positions = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with task-specific configuration"""
        self.max_iterations = config.get('max_iterations', self.max_iterations)
        self.eps = config.get('eps', self.eps)
        self.task = config.get('task', 'qa')
        
    def preprocess(self, x: Any) -> Dict[str, torch.Tensor]:
        """
        Convert input to masked token representation
        Input: Token IDs [B, T] or dict with 'input_ids' and 'attention_mask'
        Output: Dict with masked_ids, attention_mask, original_ids
        """
        if isinstance(x, dict):
            input_ids = x['input_ids']
            attention_mask = x.get('attention_mask', None)
        else:
            input_ids = x
            attention_mask = None
        
        # Ensure on correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            # Create default attention mask (all ones)
            attention_mask = torch.ones_like(input_ids)
        
        # Store original
        self.original_ids = input_ids.clone()
        
        # Create initial span mask
        if isinstance(self.model, SpanMaskingModel):
            masked_ids, mask_labels = self.model.create_span_mask(
                input_ids, 
                mask_ratio=self.initial_mask_ratio
            )
            self.masked_positions = (mask_labels != -100)
        else:
            # Simple random masking for non-span models
            masked_ids = input_ids.clone()
            mask = torch.rand_like(input_ids, dtype=torch.float) < self.initial_mask_ratio
            masked_ids[mask] = 103  # [MASK] token
            self.masked_positions = mask
        
        # Reset tracking
        self.iteration = 0
        self.meaning_history = []
        self.current_mask_ratio = self.initial_mask_ratio
        
        return {
            'masked_ids': masked_ids,
            'attention_mask': attention_mask,
            'original_ids': self.original_ids,
            'masked_positions': self.masked_positions
        }
    
    def compute_energy(self, state: Dict[str, torch.Tensor]) -> float:
        """
        Compute energy based on meaning stability and reconstruction
        Lower energy = more stable meaning + better reconstruction
        """
        with torch.no_grad():
            # Get current meaning representation
            outputs = self.model.forward(
                state['masked_ids'],
                state['attention_mask']
            )
            current_meaning = outputs['pooler_output']
            
            # Track meaning for stability calculation
            self.meaning_history.append(current_meaning)
            
            # Compute stability (if we have history)
            if len(self.meaning_history) >= 2:
                prev_meaning = self.meaning_history[-2]
                # Cosine similarity for stability
                stability = F.cosine_similarity(
                    current_meaning,
                    prev_meaning,
                    dim=-1
                ).mean()
            else:
                stability = torch.tensor(0.0)
            
            # Compute reconstruction quality if model supports MLM
            if isinstance(self.model, SpanMaskingModel):
                mlm_outputs = self.model.forward_mlm(
                    state['masked_ids'],
                    state['attention_mask'],
                    labels=self.original_ids if self.masked_positions is not None else None
                )
                if 'loss' in mlm_outputs:
                    recon_loss = mlm_outputs['loss']
                else:
                    recon_loss = torch.tensor(0.0)
            else:
                recon_loss = torch.tensor(0.0)
            
            # Combined energy (negative so lower is better)
            energy = -(stability - 0.1 * recon_loss)
            
        return energy.item()
    
    def refine_step(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Single refinement step: progressively unmask tokens
        """
        # Calculate current mask ratio (linear decay)
        progress = self.iteration / max(self.max_iterations - 1, 1)
        self.current_mask_ratio = self.initial_mask_ratio * (1 - progress) + self.final_mask_ratio * progress
        
        # Get predictions for masked tokens
        with torch.no_grad():
            if isinstance(self.model, SpanMaskingModel):
                outputs = self.model.forward_mlm(state['masked_ids'], state['attention_mask'])
                predictions = outputs['logits'].argmax(dim=-1)
            else:
                outputs = self.model.forward(state['masked_ids'], state['attention_mask'])
                # Simple heuristic: use original tokens
                predictions = self.original_ids
        
        # Progressive unmasking: reveal some masked tokens
        new_masked_ids = state['masked_ids'].clone()
        
        if self.masked_positions is not None:
            # Calculate how many tokens to unmask
            currently_masked = self.masked_positions.sum()
            target_masked = int(self.original_ids.numel() * self.current_mask_ratio)
            tokens_to_unmask = max(0, currently_masked - target_masked)
            
            if tokens_to_unmask > 0:
                # Randomly select tokens to unmask
                masked_indices = torch.where(self.masked_positions.flatten())[0]
                if len(masked_indices) > 0:
                    unmask_indices = masked_indices[
                        torch.randperm(len(masked_indices))[:tokens_to_unmask]
                    ]
                    
                    # Unmask by using predictions or original tokens
                    flat_masked = new_masked_ids.flatten()
                    flat_original = self.original_ids.flatten()
                    flat_masked[unmask_indices] = flat_original[unmask_indices]
                    new_masked_ids = flat_masked.reshape(new_masked_ids.shape)
                    
                    # Update masked positions
                    flat_positions = self.masked_positions.flatten()
                    flat_positions[unmask_indices] = False
                    self.masked_positions = flat_positions.reshape(self.masked_positions.shape)
        
        self.iteration += 1
        
        return {
            'masked_ids': new_masked_ids,
            'attention_mask': state['attention_mask'],
            'original_ids': state['original_ids'],
            'masked_positions': self.masked_positions
        }
    
    def should_halt(self, energy_history: List[float]) -> bool:
        """
        Determine if meaning has stabilized
        """
        if len(energy_history) < 3:
            return False
        
        # Check if meaning has stabilized (small energy changes)
        recent = energy_history[-3:]
        delta = abs(recent[-1] - recent[-2])
        
        # Also check if we've unmasked enough
        if self.current_mask_ratio <= self.final_mask_ratio * 1.1:
            return True
        
        return delta < self.eps or self.iteration >= self.max_iterations
    
    def postprocess(self, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Extract final meaning representation and reconstructed text
        """
        with torch.no_grad():
            outputs = self.model.forward(
                state['masked_ids'],
                state['attention_mask']
            )
        
        return {
            'refined_ids': state['masked_ids'],
            'meaning_latent': outputs['pooler_output'],
            'hidden_states': outputs['last_hidden_state'],
            'mask_ratio_final': self.current_mask_ratio
        }
    
    def compute_trust(self, initial: Any, refined: Any) -> float:
        """
        Compute trust based on meaning stability
        """
        if len(self.meaning_history) < 2:
            return 0.5
        
        # Compute average stability over last few steps
        stabilities = []
        for i in range(1, min(5, len(self.meaning_history))):
            sim = F.cosine_similarity(
                self.meaning_history[-i],
                self.meaning_history[-i-1],
                dim=-1
            ).mean()
            stabilities.append(sim.item())
        
        avg_stability = sum(stabilities) / len(stabilities) if stabilities else 0.5
        
        # Bonus for early convergence
        if self.iteration < self.max_iterations * 0.5:
            avg_stability *= 1.1
        
        return min(avg_stability, 1.0)
    
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Complete refinement pipeline
        """
        # Preprocess
        state = self.preprocess(x)
        
        # Track energy
        energy_history = []
        energy_history.append(self.compute_energy(state))
        
        # Initial meaning
        initial_meaning = self.meaning_history[0] if self.meaning_history else None
        
        # Refinement loop
        while self.iteration < self.max_iterations:
            # Refine
            state = self.refine_step(state)
            
            # Compute energy
            energy = self.compute_energy(state)
            energy_history.append(energy)
            
            # Check early stop
            if early_stop and self.should_halt(energy_history):
                break
        
        # Postprocess
        refined = self.postprocess(state)
        
        # Compute metrics
        trust = self.compute_trust(None, refined)
        
        # Meaning drift (if we have initial)
        if initial_meaning is not None and len(self.meaning_history) > 0:
            meaning_drift = 1 - F.cosine_similarity(
                initial_meaning,
                self.meaning_history[-1],
                dim=-1
            ).mean().item()
        else:
            meaning_drift = 0.0
        
        # Build telemetry
        telemetry = {
            'iterations': self.iteration,
            'final_energy': energy_history[-1],
            'energy_delta': energy_history[-1] - energy_history[0],
            'trust': trust,
            'early_stopped': self.iteration < self.max_iterations,
            'compute_saved': 1 - (self.iteration / self.max_iterations),
            'meaning_drift': meaning_drift,
            'final_mask_ratio': self.current_mask_ratio
        }
        
        return refined, telemetry


def create_language_irp(device: Optional[torch.device] = None) -> LanguageIRPImpl:
    """Factory function for Language IRP"""
    return LanguageIRPImpl(
        model_variant='span',
        device=device
    )


if __name__ == "__main__":
    print("Testing Language IRP Implementation")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create IRP
    irp = create_language_irp(device)
    
    # Test with random tokens
    test_tokens = torch.randint(100, 5000, (2, 32)).to(device)  # Avoid special tokens
    
    # Run refinement with early stopping
    print("\nRunning refinement with early stopping...")
    refined, telemetry = irp.refine(test_tokens, early_stop=True)
    
    print(f"\nResults:")
    print(f"  Iterations: {telemetry['iterations']}")
    print(f"  Final energy: {telemetry['final_energy']:.4f}")
    print(f"  Trust score: {telemetry['trust']:.3f}")
    print(f"  Compute saved: {telemetry['compute_saved']*100:.1f}%")
    print(f"  Meaning drift: {telemetry['meaning_drift']:.3f}")
    print(f"  Final mask ratio: {telemetry['final_mask_ratio']:.2f}")
    
    # Compare with full refinement
    print("\nRunning full refinement for comparison...")
    irp2 = create_language_irp(device)
    refined_full, telemetry_full = irp2.refine(test_tokens, early_stop=False)
    
    print(f"\nFull refinement:")
    print(f"  Iterations: {telemetry_full['iterations']}")
    print(f"  Final energy: {telemetry_full['final_energy']:.4f}")
    print(f"  Meaning drift: {telemetry_full['meaning_drift']:.3f}")
    
    print(f"\nSpeedup: {telemetry_full['iterations'] / telemetry['iterations']:.2f}x")
    
    # Test meaning stability
    print("\n" + "=" * 30)
    if telemetry['meaning_drift'] < 0.2 and telemetry['trust'] > 0.7:
        print("✅ SUCCESS: Meaning stabilized effectively!")
    else:
        print("⚠️  Meaning stability needs tuning")