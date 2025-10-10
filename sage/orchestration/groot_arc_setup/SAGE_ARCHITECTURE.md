# SAGE Student Model Architecture
## Knowledge Distillation from GR00T N1.5 for ARC-AGI

---

## Design Goals

1. **Efficient**: <100M parameters (27x smaller than GR00T's 2.7B)
2. **Effective**: Learn reasoning patterns from GR00T features
3. **Fast**: <10ms inference per task (vs GR00T's 150ms)
4. **Interpretable**: Clear reasoning stages (feature → reasoning → output)

---

## Architecture Overview

```
GR00T Features [seq_len, 2048]
    ↓
Feature Projection [seq_len, 512]
    ↓
Transformer Reasoning (6 layers, 512 dim, 8 heads)
    ↓
Pooling [512]
    ↓
Grid Decoder [30×30×10]
    ↓
Output Grid [30, 30] (argmax over classes 0-9)
```

---

## Component Design

### 1. Feature Projection Layer
**Purpose**: Compress GR00T's 2048-dim features to efficient 512-dim representation

```python
Input:  [batch, seq_len, 2048]
Linear: 2048 → 512
LayerNorm
GELU
Output: [batch, seq_len, 512]
```

**Parameters**: ~1M (2048 × 512)

---

### 2. Positional Encoding
**Purpose**: Preserve sequence order information

```python
Learned positional embeddings for max_seq_len=6000
Embedding: [6000, 512]
```

**Parameters**: ~3M (6000 × 512)

---

### 3. Transformer Reasoning Module
**Purpose**: Learn pattern relationships from GR00T features

**Architecture**:
- 6 Transformer encoder layers
- Hidden size: 512
- Attention heads: 8 (64 dim per head)
- FFN dimension: 2048 (4× expansion)
- Dropout: 0.1

**Computation per layer**:
```
Self-Attention:
  Q, K, V projections: 3 × (512 × 512)
  Output projection: 512 × 512

Feed-Forward:
  Expand: 512 × 2048
  Contract: 2048 × 512

Layer norms, residuals
```

**Parameters per layer**: ~4M
- Self-attention: ~1M (512² × 4)
- FFN: ~3M (512 × 2048 × 2)

**Total for 6 layers**: ~24M

---

### 4. Feature Pooling
**Purpose**: Aggregate sequence into single representation

**Options**:
1. **Mean pooling** (simple, effective)
2. **Attention pooling** (learnable, more expressive)
3. **CLS token** (BERT-style)

**Chosen**: Mean pooling for simplicity
```python
Output: [batch, 512]
```

**Parameters**: 0 (pooling is just mean)

---

### 5. Grid Decoder
**Purpose**: Generate 30×30 output grid from pooled features

**Architecture**:
```
Pooled features [512]
    ↓
Linear: 512 → 1024
GELU + Dropout
    ↓
Linear: 1024 → 2048
GELU + Dropout
    ↓
Linear: 2048 → 30×30×10 = 9000
    ↓
Reshape: [30, 30, 10]
    ↓
Softmax over classes
    ↓
Argmax → Grid [30, 30]
```

**Parameters**:
- Layer 1: 512 × 1024 = ~0.5M
- Layer 2: 1024 × 2048 = ~2M
- Layer 3: 2048 × 9000 = ~18M

**Total**: ~20.5M

---

## Total Parameter Count

| Component | Parameters |
|-----------|------------|
| Feature Projection | ~1M |
| Positional Encoding | ~3M |
| Transformer (6 layers) | ~24M |
| Grid Decoder | ~20.5M |
| **Total** | **~48.5M** |

✅ **Within target**: 48.5M < 100M (56x smaller than GR00T)

---

## Training Strategy

### 1. Feature Distillation Loss
**Purpose**: Match GR00T's internal representations

```python
# Project student features to GR00T dimension
student_proj = Linear(512 → 2048)(student_features)

# MSE loss with GR00T features
feature_loss = MSE(student_proj, groot_features)

# Cosine similarity loss (encourages aligned directions)
cosine_loss = 1 - cosine_similarity(student_proj, groot_features)

# Combined
distillation_loss = feature_loss + 0.5 * cosine_loss
```

### 2. Task Loss
**Purpose**: Predict correct output grid

```python
# Cross-entropy over 10 classes (colors 0-9)
grid_logits = model(groot_features)  # [batch, 30, 30, 10]
task_loss = CrossEntropyLoss(grid_logits, target_grid)
```

### 3. Attention Distillation (Optional)
**Purpose**: Match GR00T's attention patterns

```python
# If GR00T attention available
student_attention = student_model.get_attention()
groot_attention = groot_model.get_attention()

attention_loss = MSE(student_attention, groot_attention)
```

### Combined Loss
```python
total_loss = (
    0.5 * distillation_loss +
    1.0 * task_loss +
    0.1 * attention_loss
)
```

**Weights rationale**:
- Task loss (1.0): Primary objective
- Distillation (0.5): Learn representations
- Attention (0.1): Soft constraint on reasoning patterns

---

## Data Format

### Training Example
```python
{
    "features": torch.Tensor([1, seq_len, 2048]),  # GR00T features
    "attention_mask": torch.Tensor([1, seq_len]),   # Valid tokens
    "input_grid": np.ndarray([H, W]),               # Original input
    "output_grid": np.ndarray([H', W']),            # Target output
    "task_id": str,                                 # Task identifier
}
```

### Preprocessing
```python
# Pad output grid to 30×30 (ARC max)
output_padded = pad_to_30x30(output_grid)

# Convert to class labels (0-9)
target = torch.LongTensor(output_padded)  # [30, 30]
```

---

## Implementation Plan

### Step 1: Model Class
```python
class SAGEStudent(nn.Module):
    def __init__(
        self,
        input_dim=2048,      # GR00T features
        hidden_dim=512,      # Internal dimension
        num_layers=6,        # Transformer layers
        num_heads=8,         # Attention heads
        ffn_dim=2048,        # FFN expansion
        max_seq_len=6000,    # Max sequence length
        grid_size=30,        # ARC grid size
        num_classes=10,      # Colors 0-9
        dropout=0.1,
    ):
        super().__init__()

        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Grid decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, grid_size * grid_size * num_classes),
        )

        self.grid_size = grid_size
        self.num_classes = num_classes
```

### Step 2: Forward Pass
```python
def forward(self, groot_features, attention_mask=None):
    """
    Args:
        groot_features: [batch, seq_len, 2048]
        attention_mask: [batch, seq_len] (optional)

    Returns:
        grid_logits: [batch, 30, 30, 10]
        student_features: [batch, seq_len, 512] (for distillation)
    """
    batch_size, seq_len, _ = groot_features.shape

    # Project features
    x = self.projection(groot_features)  # [batch, seq_len, 512]

    # Add positional encoding
    positions = torch.arange(seq_len, device=x.device)
    x = x + self.pos_embedding(positions)

    # Transformer reasoning
    if attention_mask is not None:
        # Convert mask to transformer format
        mask = ~attention_mask.bool()
    else:
        mask = None

    student_features = self.transformer(
        x,
        src_key_padding_mask=mask,
    )  # [batch, seq_len, 512]

    # Pool features
    if attention_mask is not None:
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1)
        pooled = (student_features * mask_expanded).sum(1) / mask_expanded.sum(1)
    else:
        pooled = student_features.mean(1)  # [batch, 512]

    # Decode to grid
    grid_flat = self.decoder(pooled)  # [batch, 9000]
    grid_logits = grid_flat.view(
        batch_size, self.grid_size, self.grid_size, self.num_classes
    )  # [batch, 30, 30, 10]

    return grid_logits, student_features
```

### Step 3: Loss Functions
```python
class DistillationLoss(nn.Module):
    def __init__(
        self,
        task_weight=1.0,
        feature_weight=0.5,
        attention_weight=0.1,
    ):
        super().__init__()
        self.task_weight = task_weight
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight

        self.task_loss = nn.CrossEntropyLoss()
        self.feature_loss = nn.MSELoss()

        # Projection to match GR00T dimension
        self.feature_proj = nn.Linear(512, 2048)

    def forward(
        self,
        grid_logits,          # [batch, 30, 30, 10]
        target_grid,          # [batch, 30, 30]
        student_features,     # [batch, seq_len, 512]
        groot_features,       # [batch, seq_len, 2048]
        attention_mask=None,  # [batch, seq_len]
    ):
        # Task loss
        task_loss = self.task_loss(
            grid_logits.permute(0, 3, 1, 2),  # [batch, 10, 30, 30]
            target_grid,
        )

        # Feature distillation
        student_proj = self.feature_proj(student_features)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            feature_loss = (
                ((student_proj - groot_features) ** 2) * mask
            ).sum() / mask.sum()
        else:
            feature_loss = self.feature_loss(student_proj, groot_features)

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            student_proj.reshape(-1, 2048),
            groot_features.reshape(-1, 2048),
            dim=-1,
        ).mean()
        cosine_loss = 1 - cos_sim

        # Combined
        total_loss = (
            self.task_weight * task_loss +
            self.feature_weight * (feature_loss + 0.5 * cosine_loss)
        )

        return total_loss, {
            "task_loss": task_loss.item(),
            "feature_loss": feature_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "total_loss": total_loss.item(),
        }
```

---

## Training Configuration

```python
# Model
model = SAGEStudent(
    input_dim=2048,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    ffn_dim=2048,
    dropout=0.1,
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)

# Loss
criterion = DistillationLoss(
    task_weight=1.0,
    feature_weight=0.5,
)

# Training
batch_size = 16
num_epochs = 100
mixed_precision = True  # Use bfloat16 for efficiency
```

---

## Expected Performance

### Inference Speed
- GR00T: ~150ms per example (full pipeline)
- SAGE: ~10ms per example (15x faster)
  - Feature projection: ~1ms
  - Transformer: ~5ms (6 layers, 512 dim)
  - Decoder: ~4ms

### Accuracy Target
- GR00T baseline: TBD (measure on validation)
- SAGE target: >90% of GR00T accuracy
- Acceptable trade-off: 5-10% accuracy for 15x speed

### Memory
- GR00T: ~12GB VRAM (2.7B params in bfloat16)
- SAGE: ~200MB VRAM (48.5M params)
- Training: ~2GB VRAM (batch_size=16)

---

## Next Steps

1. **Implement Model** ✅
   - Create `SAGEStudent` class
   - Implement forward pass
   - Add loss functions

2. **Test Forward Pass**
   - Load sample features
   - Run model
   - Verify output shapes

3. **Create DataLoader**
   - Load feature files
   - Batch preparation
   - Augmentation (optional)

4. **Training Loop**
   - Training step
   - Validation step
   - Checkpointing
   - Logging

5. **Evaluation**
   - Test on held-out tasks
   - Compare with GR00T
   - Analyze failure cases

---

**Status**: Architecture designed, ready for implementation
**Target**: <100M parameters, >90% GR00T accuracy, 15x faster inference
