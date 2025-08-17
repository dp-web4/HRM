#!/bin/bash

# HRM Startup Script for Jetson Orin Nano
# This script sets up the environment for running HRM with locally built dependencies

echo "Setting up HRM environment for Jetson Orin Nano..."

# Add Flash Attention to Python path
export PYTHONPATH="/home/sprout/flash-attention:$PYTHONPATH"

# CUDA environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Suppress cpuinfo warning
export OPENBLAS_CORETYPE=ARMV8

echo "Environment setup complete!"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "Flash Attention: $(python3 -c 'import sys; sys.path.insert(0, \"/home/sprout/flash-attention\"); import flash_attn; print(flash_attn.__version__)' 2>/dev/null)"
echo "CUDA: $(python3 -c 'import torch; print(f\"Available: {torch.cuda.is_available()}\")' 2>/dev/null)"

# Run command passed as arguments, or start interactive shell
if [ $# -eq 0 ]; then
    echo ""
    echo "Starting interactive Python shell with HRM environment..."
    python3
else
    echo ""
    echo "Running: $@"
    "$@"
fi