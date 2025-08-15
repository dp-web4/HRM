# Complete HRM Setup Guide for Jetson Orin Nano

## Overview
This guide provides multiple approaches to set up HRM (Hierarchical Reasoning Model) on Jetson Orin Nano with JetPack 6.2.1.

## System Requirements
- **Device**: Jetson Orin Nano (8GB RAM, 40 TOPS)
- **JetPack**: 6.2.1 (L4T R36.4.4)
- **Python**: 3.10.12
- **CUDA**: 12.6
- **Storage**: ~5GB free space

## Setup Approaches

### Option 1: Docker (RECOMMENDED) 🐳
The most reliable method using NVIDIA's pre-built containers:

```bash
# 1. Run the setup script
chmod +x setup_hrm_docker.sh
./setup_hrm_docker.sh

# 2. Start container
./run_hrm_docker.sh

# 3. Test inside container
python3 test_docker.py
```

**Advantages:**
- Pre-installed PyTorch optimized for Jetson
- Isolated environment
- No dependency conflicts

### Option 2: Direct Installation
Install PyTorch and dependencies directly:

```bash
# 1. Try automated installation
chmod +x install_pytorch_final.sh
./install_pytorch_final.sh

# 2. Install HRM dependencies
pip3 install einops tqdm coolname pydantic wandb huggingface_hub argdantic
pip3 install omegaconf==2.3.0 hydra-core==1.3.2 --no-deps
```

### Option 3: Manual Wheel Installation
Download specific Jetson wheels:

1. Visit [NVIDIA PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
2. Download wheel for JetPack 6.x
3. Install:
   ```bash
   pip3 install torch-*.whl
   ```

## Modifications for Jetson

### 1. Flash Attention Alternative
HRM uses Flash Attention which isn't available on Jetson. Use our modified layers:

```bash
# Replace flash attention with standard attention
cp models/layers_jetson.py models/layers.py
```

### 2. Memory Optimization
Configure for 8GB RAM:

```python
# In config/jetson_sudoku_demo.yaml
batch_size: 8  # Reduced from 16
num_workers: 2  # Reduced from 4
```

### 3. Performance Settings
Maximize Jetson performance:

```bash
# Set to max performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Testing HRM

### Quick Test
```bash
python3 test_hrm_simple.py
```

### Mini Sudoku Demo
```bash
python3 pretrain.py --config-path config --config-name jetson_sudoku_demo
```

### Monitor Performance
```bash
# In another terminal
tegrastats
```

## Expected Performance

- **Model Size**: 27M parameters (~110MB)
- **Memory Usage**: ~350MB for training
- **Training Speed**: ~50-100 examples/sec
- **Inference Speed**: ~200-500 examples/sec

## Troubleshooting

### CUDA Errors
If you see "no kernel image available":
- PyTorch version doesn't match Jetson's compute capability
- Use Docker or Jetson-specific wheels

### Out of Memory
- Reduce batch size in config
- Close other applications
- Enable swap (already configured)

### Import Errors
- Flash Attention: Use layers_jetson.py
- adam-atan2: Not critical, can skip
- antlr4: Version conflict, install with --no-deps

## Integration with Your Projects

HRM can enhance your existing work:

1. **Binocular Vision**: Visual reasoning for object understanding
2. **IMU Integration**: Predict motion patterns
3. **Edge AI**: Proves complex reasoning on embedded devices

## Next Steps

1. Complete PyTorch installation (Docker recommended)
2. Run test scripts to verify setup
3. Train on mini datasets
4. Integrate with vision/IMU systems
5. Experiment with custom reasoning tasks

## Resources

- [HRM Paper](https://arxiv.org/abs/2410.13080)
- [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/)
- [PyTorch Jetson Wheels](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

---

Remember: The goal is efficient experimentation, not just minimal testing. Choose the setup method that gives you the most flexibility for future work!