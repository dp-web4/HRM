# GPU Setup Summary

## Current Status

✅ **PyTorch installed**: Version 2.4.0+cu121
✅ **HRM model working**: Runs successfully on CPU
✅ **Environment configured**: Conda environment 'hrm' with all dependencies
❌ **GPU not accessible**: CUDA initialization returns "unknown error"

## The Issue

The RTX 4090 GPU is detected by `nvidia-smi` but PyTorch cannot initialize CUDA:
- Driver Version: 570.169 (supports CUDA 12.8)
- CUDA Runtime: 12.6 installed
- Error: `CUDA unknown error` (code 999)

This is a system-level CUDA initialization issue, not a PyTorch or HRM problem.

## Immediate Workarounds

### 1. Run on CPU (Currently Working)
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hrm
python run_hrm_safe.py
```

### 2. Use CPU-patched version
The `run_hrm_safe.py` script includes CPU-compatible attention mechanisms.

## To Fix GPU Access

Try these solutions in order:

### 1. Enable Persistence Mode
```bash
sudo nvidia-smi -pm 1
```

### 2. Reload NVIDIA Modules
```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
sudo systemctl restart nvidia-persistenced
```

### 3. Check System Logs
```bash
# Check for NVIDIA errors
sudo dmesg | grep -i nvidia

# Check for CUDA errors
journalctl -xe | grep -i cuda
```

### 4. Reset GPU State
```bash
# Reset GPU without reboot
sudo nvidia-smi --gpu-reset
```

### 5. Update/Reinstall NVIDIA Driver
The driver version 570.169 is from 2025 (future date), which might indicate a beta/experimental driver.
Consider downgrading to a stable version:
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall
```

## Training HRM

Once GPU is working:
```bash
# Build dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku

# Train model
python pretrain.py data_path=data/sudoku
```

For now, you can experiment with the CPU version for development and testing.

## Environment Variables

Add these to `~/.bashrc` for permanent setup:
```bash
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

## Next Steps

1. **Immediate**: Continue development using CPU mode
2. **Short-term**: Debug GPU access with system administrator
3. **Long-term**: Consider container-based deployment (Docker with --gpus all)