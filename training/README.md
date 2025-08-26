# TinyVAE Knowledge Distillation Training

Train a lightweight TinyVAE by distilling knowledge from a larger teacher VAE.

## Overview

This training framework implements knowledge distillation to train our efficient TinyVAE model by learning from a more powerful teacher VAE. The approach combines multiple loss functions to ensure the student model captures both the reconstruction quality and the latent space structure of the teacher.

## Architecture

### Teacher VAE (Standard)
- **Latent dimension**: 512
- **Architecture**: 4-layer encoder/decoder
- **Parameters**: ~10M
- **Purpose**: High-quality reference model

### Student VAE (TinyVAE)
- **Latent dimension**: 128 (spatial: 4x4x128)
- **Architecture**: Lightweight 3-layer design
- **Parameters**: ~1.1M (10x smaller!)
- **Purpose**: Efficient edge deployment

## Distillation Strategy

The training uses a multi-component loss function:

```python
Total Loss = α₁·Reconstruction + α₂·KL + α₃·Latent Distillation + α₄·Output Distillation + α₅·Perceptual
```

### Loss Components

1. **Reconstruction Loss** (30%)
   - MSE between student reconstruction and original input
   - Ensures basic VAE functionality

2. **KL Divergence** (10%)
   - Regularizes the latent space
   - Prevents overfitting

3. **Latent Distillation** (30%)
   - MSE between student and teacher latent representations
   - Transfers the latent space structure

4. **Output Distillation** (20%)
   - MSE between student and teacher reconstructions
   - Learns from teacher's output quality

5. **Perceptual Loss** (10%)
   - VGG feature matching
   - Improves visual quality

## Quick Start

### 1. Test the Setup
```bash
cd HRM
python training/test_distillation.py
```

This runs quick tests to verify:
- Models can be instantiated
- Loss calculations work
- One training step completes

### 2. Run Full Training

```bash
# Basic training on CIFAR-10
python training/distill_tinyvae.py --dataset cifar10 --epochs 100

# Custom configuration
python training/distill_tinyvae.py \
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 1e-3 \
    --device cuda \
    --checkpoint-dir ./checkpoints/my_tinyvae
```

### 3. Monitor Training

The training script logs:
- Loss components per batch
- Epoch summaries
- Validation metrics
- Best model checkpoints

## Configuration

Create a custom config JSON:

```json
{
  "dataset": "cifar10",
  "batch_size": 64,
  "epochs": 100,
  "learning_rate": 0.001,
  "teacher_latent_dim": 512,
  "student_latent_dim": 128,
  "recon_weight": 0.3,
  "kl_weight": 0.1,
  "distill_latent_weight": 0.3,
  "distill_recon_weight": 0.2,
  "perceptual_weight": 0.1,
  "temperature": 3.0,
  "checkpoint_dir": "./checkpoints/custom"
}
```

Then run:
```bash
python training/distill_tinyvae.py --config my_config.json
```

## Key Features

### Temperature Scaling
The distillation uses temperature scaling (T=3.0 default) to soften the teacher's outputs, making them easier for the student to learn from.

### Progressive Training
1. **Phase 1**: Teacher pre-training (if needed)
2. **Phase 2**: Distillation training
3. **Phase 3**: Fine-tuning (optional)

### Automatic Checkpointing
- Saves every 5 epochs
- Keeps best model based on validation loss
- Stores training history for analysis

## Results Expected

After successful distillation:
- **Size reduction**: 10x smaller model
- **Speed improvement**: 5-10x faster inference
- **Quality preservation**: >95% of teacher's reconstruction quality
- **Latent alignment**: Student learns similar latent structure

## Deployment

After training, use the distilled model:

```python
import torch
from models.vision.lightweight_vae import MinimalVAE

# Load the trained model
model = MinimalVAE(latent_dim=128, base_channels=32)
checkpoint = torch.load('checkpoints/tinyvae_best.pth')
model.load_state_dict(checkpoint['student_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    latent = model.encode(image)
    reconstruction = model.decode(latent)
```

## Tips for Best Results

1. **Pre-train the teacher** well before distillation
2. **Use temperature scaling** (T=3-5 works well)
3. **Balance loss weights** - start with defaults, adjust based on validation
4. **Monitor all loss components** - not just total loss
5. **Use perceptual loss** for better visual quality
6. **Fine-tune on target domain** after distillation

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (fp16)

### Poor Reconstruction Quality
- Increase reconstruction weight
- Add more perceptual loss
- Train teacher longer first

### Latent Space Mismatch
- Adjust temperature parameter
- Increase latent distillation weight
- Use gradual unfreezing

## Next Steps

After successful distillation:
1. Deploy to Jetson for testing
2. Integrate with IRP framework
3. Test with camera pipeline
4. Optimize for TensorRT

## References

- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network"
- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes"
- Perceptual Loss: Johnson et al., "Perceptual Losses for Real-Time Style Transfer"