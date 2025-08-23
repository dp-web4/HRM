"""
TinyBERT - Ultra-lightweight BERT for Jetson
Target: 10M parameters, <2ms per token inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
import time


class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention for small models"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Precompute scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # Single matrix multiply for Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn
        return out


class FeedForward(nn.Module):
    """Compact feedforward network"""
    
    def __init__(self, hidden_size: int, ff_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        ff_size = ff_size or hidden_size * 4
        
        # Use smaller expansion for TinyBERT
        ff_size = hidden_size * 2  # 2x instead of 4x
        
        self.fc1 = nn.Linear(hidden_size, ff_size)
        self.fc2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (more stable for small models)
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x


class TinyBERT(nn.Module):
    """
    Ultra-lightweight BERT model
    Default config: 6 layers, 256 hidden, 8 heads = ~6M parameters
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output norm
        self.ln_f = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        assert T <= self.max_seq_length, f"Sequence length {T} exceeds maximum {self.max_seq_length}"
        
        # Token and position embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        x = self.embedding_dropout(tok_emb + pos_emb)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to attention mask format [B, 1, T, T]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(B, 1, T, T)
        
        # Apply transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        outputs = {"last_hidden_state": x}
        if return_hidden_states:
            outputs["hidden_states"] = hidden_states
        
        # Add CLS token representation (first token)
        outputs["pooler_output"] = x[:, 0, :]
        
        return outputs
    
    def get_meaning_latent(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract meaning representation (CLS token)"""
        outputs = self.forward(input_ids, attention_mask)
        return outputs["pooler_output"]
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def benchmark(self, device: torch.device = None, batch_size: int = 4, seq_length: int = 64):
        """Benchmark inference performance"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.eval()
        self.to(device)
        
        # Test input
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.forward(input_ids, attention_mask)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                outputs = self.forward(input_ids, attention_mask)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / 100
        
        # Calculate metrics
        tokens_per_batch = batch_size * seq_length
        ms_per_token = (elapsed * 1000) / tokens_per_batch
        tokens_per_second = tokens_per_batch / elapsed
        
        print(f"TinyBERT Benchmark Results:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Total time: {elapsed*1000:.2f}ms")
        print(f"  Per token: {ms_per_token:.3f}ms")
        print(f"  Throughput: {tokens_per_second:.0f} tokens/sec")
        print(f"  Parameters: {self.count_parameters():,}")
        print(f"  Memory: {self.count_parameters() * 4 / 1024 / 1024:.2f}MB")
        
        return {
            'total_ms': elapsed * 1000,
            'ms_per_token': ms_per_token,
            'tokens_per_second': tokens_per_second,
            'parameters': self.count_parameters()
        }


class SpanMaskingModel(TinyBERT):
    """TinyBERT with span masking for IRP"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add MLM head for masked language modeling
        self.mlm_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
    
    def create_span_mask(
        self,
        input_ids: torch.Tensor,
        mask_ratio: float = 0.15,
        span_length: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create span-based masking pattern"""
        B, T = input_ids.shape
        masked_ids = input_ids.clone()
        mask_labels = torch.full_like(input_ids, -100)  # -100 = ignore in loss
        
        for b in range(B):
            # Determine number of spans to mask
            num_masks = int(T * mask_ratio / span_length)
            
            for _ in range(num_masks):
                # Random start position
                start = torch.randint(0, T - span_length, (1,)).item()
                end = min(start + span_length, T)
                
                # Store original tokens
                mask_labels[b, start:end] = input_ids[b, start:end]
                
                # Mask tokens (use token 103 for [MASK] typically)
                masked_ids[b, start:end] = 103
        
        return masked_ids, mask_labels
    
    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with MLM loss"""
        outputs = self.forward(input_ids, attention_mask)
        
        # MLM predictions
        prediction_scores = self.mlm_head(outputs["last_hidden_state"])
        
        outputs["logits"] = prediction_scores
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                prediction_scores.view(-1, self.vocab_size),
                labels.view(-1)
            )
            outputs["loss"] = mlm_loss
        
        return outputs


def create_tiny_bert_for_jetson(variant: str = 'tiny') -> TinyBERT:
    """Factory function to create TinyBERT variants"""
    
    if variant == 'tiny':
        # ~6M parameters
        return TinyBERT(
            hidden_size=256,
            num_layers=6,
            num_heads=8
        )
    elif variant == 'micro':
        # ~2M parameters
        return TinyBERT(
            hidden_size=128,
            num_layers=4,
            num_heads=4
        )
    elif variant == 'nano':
        # ~1M parameters
        return TinyBERT(
            hidden_size=128,
            num_layers=2,
            num_heads=4
        )
    elif variant == 'span':
        # Span masking variant
        return SpanMaskingModel(
            hidden_size=256,
            num_layers=6,
            num_heads=8
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing TinyBERT on {device}")
    print("=" * 50)
    
    # Test different variants
    for variant in ['nano', 'micro', 'tiny']:
        print(f"\n{variant.upper()} variant:")
        model = create_tiny_bert_for_jetson(variant)
        metrics = model.benchmark(device, batch_size=4, seq_length=64)
        
        # Check if we hit our target
        if metrics['ms_per_token'] < 2.0:
            print(f"  ✓ Target achieved: {metrics['ms_per_token']:.3f}ms per token")
        else:
            print(f"  ✗ Too slow: {metrics['ms_per_token']:.3f}ms per token")
    
    # Test span masking
    print(f"\nSPAN MASKING variant:")
    span_model = create_tiny_bert_for_jetson('span')
    span_model = span_model.to(device)
    
    # Create sample input
    input_ids = torch.randint(0, 10000, (2, 32), device=device)
    masked_ids, labels = span_model.create_span_mask(input_ids)
    
    # Forward pass
    outputs = span_model.forward_mlm(masked_ids, labels=labels)
    print(f"  MLM Loss: {outputs['loss'].item():.4f}")
    print(f"  Meaning latent shape: {span_model.get_meaning_latent(input_ids).shape}")