# CUDA Error 999 Diagnosis

## Issue Summary
- **Error**: CUDA unknown error (code 999)
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **Driver**: 570.169 (appears to be beta/future version)
- **CUDA Version**: 12.8
- **PyTorch**: 2.4.0+cu121
- **OS**: Ubuntu with kernel 6.8.0-64-generic
- **Display Server**: Wayland ⚠️

## Root Cause
The CUDA error 999 occurs even with direct CUDA C code, indicating a system-level issue. Key factors:

1. **Wayland Session**: Running under Wayland can cause CUDA initialization issues
2. **Beta Driver**: Version 570.169 appears to be from the future (2025 date)
3. **Persistence Mode**: Already enabled but doesn't help

## Solutions (in order)

### 1. Switch to X11 (Recommended)
Log out and select "Ubuntu on Xorg" at the login screen instead of the default Wayland session.

### 2. Simple Reboot
Often fixes CUDA initialization issues:
```bash
sudo reboot
```

### 3. Downgrade NVIDIA Driver
The 570.x driver series might be experimental. Downgrade to stable 535.x:
```bash
# Check available drivers
ubuntu-drivers devices

# Install stable driver
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

### 4. Force Module Reload (if X11 doesn't help)
```bash
# Stop display manager
sudo systemctl stop gdm3

# Unload all NVIDIA modules
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

# Reload them
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# Restart display manager
sudo systemctl start gdm3
```

## Current Workaround
HRM works perfectly on CPU mode:
```bash
cd /home/dp/ai-workspace/HRM
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hrm
python run_hrm_safe.py
```

## What We Tried
- ✅ Enabled persistence mode
- ✅ Created device nodes with nvidia-modprobe
- ✅ Installed correct PyTorch version
- ✅ Fixed nn.Buffer issues in HRM code
- ❌ Module reload (blocked by gnome-shell)
- ❌ GPU reset (blocked by active processes)

## Next Steps
1. **Immediate**: Continue using CPU mode for development
2. **Quick Fix**: Log out and switch to X11 session
3. **Permanent Fix**: Downgrade to stable NVIDIA driver

The issue is not with PyTorch or HRM, but with the system-level CUDA initialization, likely due to Wayland + beta driver combination.