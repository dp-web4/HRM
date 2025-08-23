"""
Language IRP Plugin Implementation
Version: 1.0 (2025-08-23)

Masked denoising for text understanding and generation.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import random

from .base import IRPPlugin, IRPState


class LanguageIRP(IRPPlugin):
    """
    Language plugin using masked denoising for understanding.
    
    Key features:
    - Span-based masking and denoising
    - Meaning latent extraction
    - Progressive refinement from surface to deep semantics
    - Lightweight alternative to full diffusion LMs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Language IRP with transformer-based denoiser.
        
        Config parameters:
            - vocab_size: Size of vocabulary (default 50000)
            - hidden_dim: Hidden dimension for transformer (default 512)
            - max_seq_len: Maximum sequence length (default 512)
            - meaning_dim: Dimension of meaning latent (default 128)
            - mask_token_id: ID for mask token (default 103)
            - device: Compute device
        """
        super().__init__(config)
        
        self.vocab_size = config.get('vocab_size', 50000)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.max_seq_len = config.get('max_seq_len', 512)
        self.meaning_dim = config.get('meaning_dim', 128)
        self.mask_token_id = config.get('mask_token_id', 103)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build components
        self.embedder = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.denoiser = self._build_denoiser()
        self.meaning_extractor = self._build_meaning_extractor()
        
        # Noise schedules for different refinement depths
        self.noise_schedules = {
            'surface': [0.5, 0.4, 0.3, 0.2, 0.1],
            'syntactic': [0.3, 0.25, 0.2, 0.15, 0.1],
            'semantic': [0.2, 0.15, 0.1, 0.05, 0.0]
        }
        
    def _build_denoiser(self) -> nn.Module:
        """Build transformer-based denoiser."""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
    
    def _build_meaning_extractor(self) -> nn.Module:
        """Build meaning latent extractor."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.meaning_dim),
            nn.Tanh()
        )
    
    def mask_spans(self, tokens: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span-based masking to input tokens.
        
        Args:
            tokens: Input token IDs [B, L]
            mask_ratio: Fraction of tokens to mask
            
        Returns:
            Tuple of (masked_tokens, mask_positions)
        """
        batch_size, seq_len = tokens.shape
        masked_tokens = tokens.clone()
        mask_positions = torch.zeros_like(tokens, dtype=torch.bool)
        
        for b in range(batch_size):
            # Determine number of spans to mask
            num_masks = int(seq_len * mask_ratio)
            if num_masks == 0:
                continue
                
            # Create spans of varying lengths
            span_lengths = np.random.choice([1, 2, 3, 4, 5], size=num_masks // 3 + 1)
            
            mask_indices = []
            for span_len in span_lengths:
                if len(mask_indices) >= num_masks:
                    break
                    
                # Choose random start position
                start = random.randint(0, seq_len - span_len)
                for i in range(span_len):
                    if start + i < seq_len:
                        mask_indices.append(start + i)
            
            # Apply masks
            for idx in mask_indices[:num_masks]:
                masked_tokens[b, idx] = self.mask_token_id
                mask_positions[b, idx] = True
        
        return masked_tokens, mask_positions
    
    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize refinement state from input text.
        
        Args:
            x0: Input tokens or text string
            task_ctx: Task context including mode (understand/generate)
            
        Returns:
            Initial IRPState with masked tokens
        """
        # Convert text to tokens if needed
        if isinstance(x0, str):
            # Simplified tokenization (would use proper tokenizer in practice)
            tokens = torch.tensor([ord(c) % self.vocab_size for c in x0[:self.max_seq_len]])
            tokens = tokens.unsqueeze(0)  # Add batch dimension
        else:
            tokens = x0
        
        tokens = tokens.to(self.device)
        
        # Determine initial mask ratio based on task
        mode = task_ctx.get('mode', 'understand')
        if mode == 'understand':
            # Start with heavy masking for understanding
            mask_ratio = 0.5
        else:
            # Start with full masking for generation
            mask_ratio = 1.0
            tokens = torch.full_like(tokens, self.mask_token_id)
        
        # Apply initial masking
        masked_tokens, mask_positions = self.mask_spans(tokens, mask_ratio)
        
        # Store metadata
        meta = {
            'original_tokens': tokens.clone(),
            'masked_tokens': masked_tokens,
            'mask_positions': mask_positions,
            'task_ctx': task_ctx,
            'mode': mode,
            'refinement_level': 'surface',
            'perplexity_history': []
        }
        
        return IRPState(
            x=masked_tokens,
            step_idx=0,
            energy_val=None,
            meta=meta
        )
    
    def energy(self, state: IRPState) -> float:
        """
        Compute energy as perplexity of current denoising.
        
        Lower perplexity indicates better understanding.
        
        Args:
            state: Current refinement state
            
        Returns:
            Scalar energy (log perplexity)
        """
        tokens = state.x
        
        # Get embeddings
        embeddings = self.embedder(tokens)
        
        # Pass through denoiser
        with torch.no_grad():
            denoised = self.denoiser(embeddings)
            
            # Simple perplexity estimation (would use proper LM head in practice)
            # For now, use reconstruction distance as proxy
            original_embeddings = self.embedder(state.meta['original_tokens'])
            perplexity = nn.functional.mse_loss(denoised, original_embeddings).item()
        
        # Track perplexity
        state.meta['perplexity_history'].append(perplexity)
        
        return float(np.log(perplexity + 1e-6))
    
    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Execute one denoising iteration.
        
        Args:
            state: Current state with masked tokens
            noise_schedule: Optional custom noise schedule
            
        Returns:
            Updated state with refined tokens
        """
        tokens = state.x
        step_idx = state.step_idx
        
        # Determine refinement level
        max_steps = self.config.get('max_iterations', 50)
        if step_idx < max_steps // 3:
            level = 'surface'
        elif step_idx < 2 * max_steps // 3:
            level = 'syntactic'
        else:
            level = 'semantic'
        
        state.meta['refinement_level'] = level
        
        # Get noise schedule for current level
        schedule = noise_schedule or self.noise_schedules[level]
        schedule_idx = min(step_idx % len(schedule), len(schedule) - 1)
        current_mask_ratio = schedule[schedule_idx]
        
        # Get embeddings
        embeddings = self.embedder(tokens)
        
        # Denoise
        with torch.no_grad():
            denoised_embeddings = self.denoiser(embeddings)
            
            # Project back to token space (simplified)
            # In practice, would use proper LM head
            vocab_embeddings = self.embedder.weight  # [V, D]
            
            # Compute similarities
            similarities = torch.matmul(denoised_embeddings, vocab_embeddings.t())  # [B, L, V]
            predicted_tokens = torch.argmax(similarities, dim=-1)  # [B, L]
            
            # Progressively unmask
            mask_positions = state.meta['mask_positions']
            new_tokens = tokens.clone()
            
            # Unmask some positions based on confidence
            confidences = torch.max(torch.softmax(similarities, dim=-1), dim=-1)[0]  # [B, L]
            high_conf_mask = (confidences > 0.7) & mask_positions
            
            # Update high-confidence positions
            new_tokens[high_conf_mask] = predicted_tokens[high_conf_mask]
            
            # Update mask positions
            new_mask_positions = mask_positions & ~high_conf_mask
        
        # Update state
        new_state = IRPState(
            x=new_tokens,
            step_idx=step_idx + 1,
            energy_val=None,
            meta={**state.meta, 'mask_positions': new_mask_positions}
        )
        
        return new_state
    
    def extract_meaning(self, state: IRPState) -> torch.Tensor:
        """
        Extract meaning latent from current state.
        
        Args:
            state: Current refinement state
            
        Returns:
            Meaning latent vector
        """
        tokens = state.x
        embeddings = self.embedder(tokens)
        
        with torch.no_grad():
            # Pass through denoiser
            refined_embeddings = self.denoiser(embeddings)
            
            # Pool over sequence
            pooled = torch.mean(refined_embeddings, dim=1)  # [B, D]
            
            # Extract meaning latent
            meaning = self.meaning_extractor(pooled)
        
        return meaning
    
    def halt(self, history: List[IRPState]) -> bool:
        """
        Determine if refinement should stop.
        
        Halts when:
        - Perplexity stabilizes
        - All masks removed (for understanding)
        - Maximum iterations reached
        
        Args:
            history: List of states from refinement
            
        Returns:
            True if refinement should halt
        """
        # Check base conditions
        if super().halt(history):
            return True
        
        if not history:
            return False
        
        latest_state = history[-1]
        
        # Check if all masks removed (for understanding task)
        if latest_state.meta.get('mode') == 'understand':
            mask_positions = latest_state.meta.get('mask_positions')
            if mask_positions is not None and not mask_positions.any():
                return True
        
        # Check perplexity stabilization
        perp_history = latest_state.meta.get('perplexity_history', [])
        if len(perp_history) >= 5:
            recent_perp = perp_history[-5:]
            perp_variance = np.var(recent_perp)
            if perp_variance < 0.01:
                return True
        
        return False
    
    def get_understanding(self, state: IRPState) -> Dict[str, Any]:
        """
        Extract understanding from refined state.
        
        Args:
            state: Final refined state
            
        Returns:
            Dictionary with understanding metrics
        """
        # Extract meaning latent
        meaning = self.extract_meaning(state)
        
        return {
            'refined_tokens': state.x.cpu().numpy(),
            'meaning_latent': meaning.cpu().numpy(),
            'refinement_level': state.meta['refinement_level'],
            'final_perplexity': state.meta['perplexity_history'][-1] if state.meta['perplexity_history'] else float('inf'),
            'refinement_steps': state.step_idx,
            'masks_remaining': state.meta['mask_positions'].sum().item() if 'mask_positions' in state.meta else 0
        }