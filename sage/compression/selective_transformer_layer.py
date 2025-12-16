#!/usr/bin/env python3
"""
Selective Transformer Layer for SAGE + Qwen3-Omni

Implements full transformer layer with:
- RMSNorm
- Multi-head attention (GQA)
- RoPE positional encoding
- Selective MoE (SAGE's contribution!)
- Residual connections

This demonstrates SAGE's selective resource loading integrated
into a complete transformer architecture.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .selective_expert_loader import SelectiveExpertLoader
except ImportError:
    from selective_expert_loader import SelectiveExpertLoader


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
        Returns:
            normalized: [batch, seq, hidden]
        """
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class MultimodalRotaryEmbedding(nn.Module):
    """
    Multimodal Rotary Position Embedding (mRoPE)

    Q3-Omni uses mRoPE which splits head_dim into multiple sections,
    each with independent position sequences for different modalities.

    For Q3-Omni: mrope_section = [24, 20, 20] (text, image, audio)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 65536,
        base: float = 1000000.0,
        mrope_section: List[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # mRoPE sections specify RoPE half-dims, need doubling for full dims
        # Q3-Omni: mrope_section = [24, 20, 20] -> [48, 40, 40] = 128
        if mrope_section is None:
            # Default to standard RoPE (single section)
            mrope_section = [dim]
        else:
            # Double each section: [24, 20, 20] -> [48, 40, 40]
            mrope_section = [s * 2 for s in mrope_section]

        self.mrope_section = mrope_section

        # Verify sections sum to full dim
        if sum(mrope_section) != dim:
            print(f"⚠️  mRoPE sections {mrope_section} sum to {sum(mrope_section)}, expected {dim}")
            print(f"   Using single section fallback")
            self.mrope_section = [dim]

        # Precompute frequency tensors for each section
        # Each section uses standard RoPE formula with step-by-2
        self.inv_freqs = []
        for section_dim in self.mrope_section:
            # Standard RoPE: step by 2 to get half-dimension freqs
            inv_freq = 1.0 / (self.base ** (torch.arange(0, section_dim, 2).float() / section_dim))
            self.register_buffer(f"inv_freq_{len(self.inv_freqs)}", inv_freq, persistent=False)
            self.inv_freqs.append(inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (for device/dtype)
            seq_len: Sequence length
        Returns:
            (cos, sin) embeddings for mRoPE with shape (3, 1, seq_len, dim)
        """
        # Generate position indices (same for all 3 types in text-only mode)
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        # Compute frequencies for each of 3 position types (temporal, height, width)
        # For text-only, all 3 use the same positions, but we still need 3 copies
        all_cos = []
        all_sin = []

        for position_type_idx in range(3):  # 3 position types
            cos_sections = []
            sin_sections = []

            for inv_freq in self.inv_freqs:
                # Compute frequencies for this section
                freqs = torch.outer(t, inv_freq.to(x.device))

                # Standard RoPE doubling: cat freqs with itself
                emb = torch.cat((freqs, freqs), dim=-1)

                cos_sections.append(emb.cos())
                sin_sections.append(emb.sin())

            # Concatenate sections for this position type
            all_cos.append(torch.cat(cos_sections, dim=-1))  # [seq_len, dim]
            all_sin.append(torch.cat(sin_sections, dim=-1))  # [seq_len, dim]

        # Stack to get (3, seq_len, dim)
        cos = torch.stack(all_cos, dim=0).unsqueeze(1)  # [3, 1, seq_len, dim]
        sin = torch.stack(all_sin, dim=0).unsqueeze(1)  # [3, 1, seq_len, dim]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mRoPE to query and key tensors with interleaved sections

    HuggingFace implementation from Qwen2-VL:
    Splits head_dim by mrope_section and cycles through 3 position types
    """
    # Double sections to get full dimensions [24,20,20] -> [48,40,40]
    mrope_section_doubled = [s * 2 for s in mrope_section]

    # Split cos/sin by sections and interleave using i % 3
    # cos/sin shape: (3, batch, seq_len, dim)
    # After split: list of (3, batch, seq_len, section_dim) tensors
    cos_chunks = cos.split(mrope_section_doubled, dim=-1)
    sin_chunks = sin.split(mrope_section_doubled, dim=-1)

    # Interleave: cycle through 3 position types for each chunk
    # m[i % 3] selects from (3, batch, seq_len, section_dim) -> (batch, seq_len, section_dim)
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1)  # (batch, seq_len, dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin_chunks)], dim=-1)  # (batch, seq_len, dim)

    # Unsqueeze to add head dimension for broadcasting: (batch, 1, seq_len, dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA)

    GQA: Fewer KV heads than Q heads for efficiency
    Qwen3-Omni: 32 Q heads, 4 KV heads
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 65536,
        rope_theta: float = 1000000.0,
        extraction_dir: str = None,
        layer_id: int = None,
        component: str = "thinker",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        # GQA: num_q_heads > num_kv_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Q/K normalization (Q3-Omni uses this!)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # Load REAL attention weights if extraction_dir provided
        if extraction_dir is not None and layer_id is not None:
            import safetensors
            import os
            attn_file = os.path.join(
                extraction_dir,
                "attention",
                f"{component}_attention_layer_{layer_id:02d}.safetensors"
            )
            if os.path.exists(attn_file):
                with safetensors.safe_open(attn_file, framework="pt") as f:
                    prefix = f"{component}.model.layers.{layer_id}.self_attn"

                    # Load Q, K, V, O projections (convert to float32 for CPU compatibility)
                    if f"{prefix}.q_proj.weight" in f.keys():
                        self.q_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.q_proj.weight").float())
                    if f"{prefix}.k_proj.weight" in f.keys():
                        self.k_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.k_proj.weight").float())
                    if f"{prefix}.v_proj.weight" in f.keys():
                        self.v_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.v_proj.weight").float())
                    if f"{prefix}.o_proj.weight" in f.keys():
                        self.o_proj.weight = nn.Parameter(f.get_tensor(f"{prefix}.o_proj.weight").float())

                    # Load Q/K norms (critical for Q3-Omni!)
                    if f"{prefix}.q_norm.weight" in f.keys():
                        self.q_norm.weight = nn.Parameter(f.get_tensor(f"{prefix}.q_norm.weight").float())
                    if f"{prefix}.k_norm.weight" in f.keys():
                        self.k_norm.weight = nn.Parameter(f.get_tensor(f"{prefix}.k_norm.weight").float())

                    print(f"✅ Loaded real attention weights for layer {layer_id}")
            else:
                print(f"⚠️  Attention file not found: {attn_file}, using random weights")

        # mRoPE (Q3-Omni uses multimodal RoPE with section splitting)
        self.rotary_emb = MultimodalRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            mrope_section=[24, 20, 20],  # Q3-Omni config: text, image, audio sections
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: Optional [batch, 1, seq, seq] causal mask
            debug: Enable debug logging
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_length, _ = hidden_states.shape

        if debug:
            print(f"\n🔍 Attention Forward (Layer):")
            print(f"  Input: {hidden_states.shape}, mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")
            print(f"  q_proj weight: {self.q_proj.weight.shape}, mean={self.q_proj.weight.mean():.4f}, std={self.q_proj.weight.std():.4f}")
            print(f"  k_proj weight: {self.k_proj.weight.shape}, mean={self.k_proj.weight.mean():.4f}, std={self.k_proj.weight.std():.4f}")

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if debug:
            print(f"  After projection: Q mean={query_states.mean():.4f}, K mean={key_states.mean():.4f}")

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply Q/K normalization (Q3-Omni critical feature!)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply mRoPE
        cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin,
            mrope_section=[24, 20, 20],  # Q3-Omni config
        )

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_attention_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match Q heads (for GQA)"""
        if n_rep == 1:
            return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SelectiveMoELayer(nn.Module):
    """
    Selective Mixture of Experts with SAGE integration

    Key difference from standard MoE:
    - Loads experts on-demand based on router + SNARC salience
    - Trust-based eviction policy
    - Metabolic state determines expert budget
    """

    def __init__(
        self,
        hidden_size: int,
        expert_loader: SelectiveExpertLoader,
        layer_id: int,
        num_experts_per_tok: int = 8,
        trust_selector=None,  # Optional TrustBasedExpertSelector
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_loader = expert_loader
        self.layer_id = layer_id
        self.num_experts_per_tok = num_experts_per_tok
        self.trust_selector = trust_selector

    def forward(
        self,
        hidden_states: torch.Tensor,
        snarc_salience: Optional[Dict[str, float]] = None,
        metabolic_state: str = "FOCUS",
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            snarc_salience: Optional SNARC scores for expert selection
            metabolic_state: WAKE/FOCUS/CRISIS (determines expert count)
            debug: Enable debug logging
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Select experts PER TOKEN
        # Trust-based selection if available, otherwise SNARC-augmented or standard
        if self.trust_selector is not None:
            # Get router logits first
            router = self.expert_loader.load_router(self.layer_id)
            hidden_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden]
            import torch.nn.functional as F
            router_logits = F.linear(hidden_flat, router)  # [batch*seq, num_experts]

            # Use trust-based selection (per-token)
            # For now, use mean embedding as context (simplified)
            # TODO: Use ContextClassifier for more sophisticated context detection
            mean_embedding = hidden_states.mean(dim=(0, 1)).detach().cpu().numpy()

            # Select experts using trust
            result = self.trust_selector.select_experts(
                router_logits=router_logits[0],  # Use first token as representative
                context=None,  # Will auto-classify if classifier available
                k=self.num_experts_per_tok,
                input_embedding=mean_embedding
            )

            # Convert to tensor format expected by rest of code
            # Repeat selection across all tokens (simplified for now)
            selected_ids = torch.tensor(result.selected_expert_ids, device=hidden_states.device)
            selected_expert_ids = selected_ids.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

            selected_weights = torch.tensor(result.selection_scores, device=hidden_states.device)
            router_weights = selected_weights.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

            if debug:
                print(f"\n🔍 Trust-based selection enabled")
                print(f"  Context: {result.context}")
                print(f"  Selected experts: {result.selected_expert_ids}")
        else:
            # Standard SNARC-augmented selection
            selected_expert_ids, router_weights = self.expert_loader.select_experts_snarc(
                hidden_states,
                self.layer_id,
                num_experts=self.num_experts_per_tok,
                snarc_salience=snarc_salience,
                metabolic_state=metabolic_state
            )
        # selected_expert_ids: [batch, seq, num_experts_per_tok]
        # router_weights: [batch, seq, num_experts_per_tok]

        if debug:
            print(f"\n🔍 MoE Layer {self.layer_id}:")
            print(f"  Expert IDs shape: {selected_expert_ids.shape}")
            print(f"  Sample experts (token 0): {selected_expert_ids[0, 0]}")

        # Process each token with its selected experts
        output = torch.zeros_like(hidden_states)

        for b in range(batch_size):
            for s in range(seq_length):
                token_hidden = hidden_states[b, s:s+1, :]  # [1, 1, hidden]
                token_expert_ids = selected_expert_ids[b, s]  # [num_experts_per_tok]
                token_weights = router_weights[b, s]  # [num_experts_per_tok]

                # Process this token with its experts
                token_output = torch.zeros_like(token_hidden)
                valid_weight_sum = 0.0

                for i, expert_id in enumerate(token_expert_ids):
                    expert_id = int(expert_id.item())

                    # Load expert (from memory or disk)
                    expert_weights = self.expert_loader.load_expert(expert_id, self.layer_id)

                    # Skip if expert doesn't exist (sparse layers)
                    if expert_weights is None:
                        continue

                    # Expert forward pass on this token
                    expert_out = self._expert_forward(token_hidden, expert_weights, debug=False)
                    token_output += token_weights[i] * expert_out
                    valid_weight_sum += token_weights[i].item()

                # Renormalize if some experts were missing
                if valid_weight_sum > 0 and valid_weight_sum != 1.0:
                    token_output = token_output / valid_weight_sum

                output[b, s] = token_output[0]  # Extract [hidden] vector from [1, hidden]

        return output

    def _expert_forward(self, x: torch.Tensor, expert_weights: Dict[str, torch.Tensor], debug: bool = False) -> torch.Tensor:
        """Standard expert MLP: h = down(silu(gate(x)) * up(x))"""
        gate_proj = None
        up_proj = None
        down_proj = None

        for key, weight in expert_weights.items():
            if 'gate_proj' in key:
                gate_proj = weight
            elif 'up_proj' in key:
                up_proj = weight
            elif 'down_proj' in key:
                down_proj = weight

        # Validate all weights are present
        if gate_proj is None or up_proj is None or down_proj is None:
            print(f"⚠️  Expert missing weights! gate:{gate_proj is not None}, up:{up_proj is not None}, down:{down_proj is not None}")
            print(f"   Available keys: {list(expert_weights.keys())}")
            # Return input unchanged if weights are missing
            return x

        if debug:
            print(f"    Expert weights: gate {gate_proj.shape} mean={gate_proj.mean():.4f}, std={gate_proj.std():.4f}")
            print(f"                    up {up_proj.shape} mean={up_proj.mean():.4f}, std={up_proj.std():.4f}")
            print(f"                    down {down_proj.shape} mean={down_proj.mean():.4f}, std={down_proj.std():.4f}")

        gate_output = F.linear(x, gate_proj)
        up_output = F.linear(x, up_proj)
        intermediate = F.silu(gate_output) * up_output
        output = F.linear(intermediate, down_proj)

        if debug:
            print(f"    Expert output: mean={output.mean():.4f}, std={output.std():.4f}")

        return output


class SelectiveTransformerLayer(nn.Module):
    """
    Complete transformer layer with selective expert loading

    Architecture:
    x = x + attention(norm(x))
    x = x + moe(norm(x))

    Where MoE uses SAGE's selective expert loading
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        expert_loader: Optional[SelectiveExpertLoader] = None,
        layer_id: int = 0,
        num_experts_per_tok: int = 8,
        rms_norm_eps: float = 1e-6,
        extraction_dir: str = None,
        trust_selector=None,  # Optional TrustBasedExpertSelector
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.trust_selector = trust_selector

        # Pre-normalization (load real weights if available!)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Load REAL norm weights if extraction_dir provided
        if extraction_dir is not None and layer_id is not None:
            import safetensors
            import os
            norms_file = os.path.join(
                extraction_dir,
                "norms",
                f"thinker_norms_layer_{layer_id:02d}.safetensors"
            )
            if os.path.exists(norms_file):
                with safetensors.safe_open(norms_file, framework="pt") as f:
                    prefix = f"thinker.model.layers.{layer_id}"

                    # Load input layernorm
                    if f"{prefix}.input_layernorm.weight" in f.keys():
                        self.input_layernorm.weight = nn.Parameter(
                            f.get_tensor(f"{prefix}.input_layernorm.weight")
                        )

                    # Load post-attention layernorm
                    if f"{prefix}.post_attention_layernorm.weight" in f.keys():
                        self.post_attention_layernorm.weight = nn.Parameter(
                            f.get_tensor(f"{prefix}.post_attention_layernorm.weight")
                        )

                    print(f"✅ Loaded real layer norms for layer {layer_id}")
            else:
                print(f"⚠️  Norms file not found for layer {layer_id}, using default initialization")

        # Self-attention (with REAL weights if extraction_dir provided!)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            extraction_dir=extraction_dir,
            layer_id=layer_id,
            component="thinker",
        )

        # Selective MoE
        if expert_loader is None:
            raise ValueError("expert_loader required for SelectiveTransformerLayer")

        self.moe = SelectiveMoELayer(
            hidden_size=hidden_size,
            expert_loader=expert_loader,
            layer_id=layer_id,
            num_experts_per_tok=num_experts_per_tok,
            trust_selector=self.trust_selector,  # Pass trust selector to MoE
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        snarc_salience: Optional[Dict[str, float]] = None,
        metabolic_state: str = "FOCUS",
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: Optional causal mask
            snarc_salience: Optional SNARC scores
            metabolic_state: Resource budget
            debug: Enable debug logging
        Returns:
            output: [batch, seq, hidden]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, debug=debug)
        hidden_states = residual + hidden_states

        # MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(
            hidden_states,
            snarc_salience=snarc_salience,
            metabolic_state=metabolic_state,
            debug=debug,
        )
        hidden_states = residual + hidden_states

        return hidden_states


def create_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask (lower triangular)"""
    mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
