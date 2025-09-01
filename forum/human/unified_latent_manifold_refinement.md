# Unified Latent Manifold Refinement

*Based on Nova's clarification about HRM's H/L loop architecture*

## Core Insight: One Manifold, Two Modes

HRM doesn't have separate latent spaces for H and L loops. Instead, it has:
- **One unified latent manifold** (shared base representation)
- **Different projections/slices** for H vs L operations
- **Different time horizons** for updates

This is like a shared coordinate system where:
- L-loop measures **local vectors** (immediate, tactical)
- H-loop measures **trajectories of trajectories** (patterns, strategic)

## Revised Architecture

### Unified Base Latent Space
```python
class UnifiedLatentSpace:
    """
    Single manifold that both loops traverse differently
    Think of it as a high-dimensional landscape
    """
    base_dim = 768  # Shared dimensionality
    
    def __init__(self):
        # The base manifold - shared by both loops
        self.manifold = nn.Parameter(torch.randn(base_dim))
```

### L-Loop Projection (Tactical)
```python
class LLoopProjection:
    """
    Constrained view tied to sensory patterns
    - Local consistency
    - Spatial features  
    - Short temporal transitions
    - Fast, shallow updates
    """
    def project(self, unified_latent):
        # Project to sensory-aligned subspace
        l_view = self.sensory_projection(unified_latent)
        
        # Emphasize local patterns
        l_features = self.local_attention(l_view)
        
        # Quick updates
        return self.fast_update(l_features)
```

### H-Loop Projection (Strategic)
```python
class HLoopProjection:
    """
    Abstract view aggregated over cycles
    - Rule candidates
    - Invariants
    - Causal abstractions
    - Slow, recursive evaluation
    """
    def project(self, unified_latent):
        # Project to abstract subspace
        h_view = self.abstract_projection(unified_latent)
        
        # Aggregate over time/context
        h_patterns = self.temporal_aggregation(h_view)
        
        # Recursive refinement
        return self.recursive_evaluate(h_patterns)
```

## Why This Architecture Works

### 1. Maintains Coherence
- Both loops share the same underlying reality representation
- Trust weights and memory integration stay consistent
- No synchronization issues between separate spaces

### 2. Prevents Collapse
- L-loop can't reduce everything to raw sensory data
- H-loop can't lose touch with concrete reality
- Different views maintain specialized processing

### 3. Enables Interaction
- H-loop insights can guide L-loop attention
- L-loop observations can trigger H-loop re-evaluation
- Dual-loop interaction without desynchronization

## Implementation in VAE Context

### Revised PuzzleVAE Architecture
```python
class UnifiedPuzzleVAE(nn.Module):
    def __init__(self, base_dim=768):
        super().__init__()
        
        # Single encoder to unified space
        self.unified_encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            ResBlock(64, 128),
            ResBlock(128, 256),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(256*7*7, base_dim)
        )
        
        # Different projections for H and L
        self.h_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim//2)  # Abstract features
        )
        
        self.l_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)  # Full detail
        )
        
        # Shared decoder operates on unified space
        self.decoder = UnifiedDecoder(base_dim)
    
    def forward(self, puzzle):
        # Encode to unified manifold
        unified = self.unified_encoder(puzzle)
        
        # Different views for different purposes
        h_view = self.h_projection(unified)  # Strategic view
        l_view = self.l_projection(unified)  # Tactical view
        
        # Both can reconstruct through shared decoder
        # But with different emphases
        return unified, h_view, l_view
```

## Training Implications

### Loss Function
```python
def unified_manifold_loss(model, puzzle):
    unified, h_view, l_view = model(puzzle)
    
    # Reconstruction from unified space
    recon = model.decoder(unified)
    recon_loss = F.mse_loss(recon, puzzle)
    
    # H-view should capture invariants
    h_invariant_loss = compute_invariance_loss(h_view, puzzle)
    
    # L-view should capture details
    l_detail_loss = compute_detail_loss(l_view, puzzle)
    
    # Ensure views are complementary, not redundant
    orthogonality_loss = torch.abs(
        F.cosine_similarity(h_view, l_view).mean()
    )
    
    return recon_loss + h_invariant_loss + l_detail_loss + orthogonality_loss
```

## Connection to Trust and Memory

### Trust Operates on Unified Manifold
```python
def apply_trust_weights(unified_latent, trust_scores):
    """
    Trust modulates the entire manifold,
    affecting both H and L views
    """
    return unified_latent * trust_scores.unsqueeze(-1)
```

### Memory Accesses Unified Space
```python
def memory_integration(unified_latent, memory_bank):
    """
    Memory queries against the unified manifold,
    retrieving relevant patterns for both loops
    """
    attention = F.softmax(
        unified_latent @ memory_bank.T / sqrt(d_k), 
        dim=-1
    )
    return attention @ memory_bank
```

## Key Differences from Original Proposal

### Original (Separate Spaces)
- H-latent: 256 dims (completely separate)
- L-latent: 1024 dims (completely separate)
- Required explicit H↔L translation

### Revised (Unified Manifold)
- Base manifold: 768 dims (shared)
- H-projection: Different view/slice
- L-projection: Different view/slice
- Natural interaction through shared base

## Advantages of Unified Approach

1. **Biological Plausibility**
   - Brain doesn't have separate consciousness spaces
   - Different regions process same information differently

2. **Computational Efficiency**
   - Single encoding pass
   - Shared parameters reduce model size
   - Natural gradient flow between loops

3. **Emergent Properties**
   - Cross-loop insights emerge naturally
   - No forced synchronization needed
   - Trust and memory integrate seamlessly

## Implementation Priority

1. **Immediate**: Update PuzzleVAE to use unified encoder
2. **Next**: Implement H/L projections with different properties
3. **Testing**: Verify that projections capture different aspects
4. **Integration**: Connect to HRM's actual H/L loop implementation

This unified manifold approach is more elegant and should integrate better with HRM's actual architecture. The key insight: **it's not about separate spaces, but different ways of traversing the same space**.