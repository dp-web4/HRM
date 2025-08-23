#!/bin/bash
# Run HRM with Docker container that has pre-built PyTorch for Jetson

echo "=== Running HRM with Docker PyTorch Container ==="
echo "Container: dustynv/l4t-pytorch:r36.2.0"
echo ""

# Check if container exists
if ! sudo docker images | grep -q "dustynv/l4t-pytorch.*r36.2.0"; then
    echo "Docker container not found. Pulling..."
    sudo docker pull dustynv/l4t-pytorch:r36.2.0
fi

echo "Starting container with GPU access..."
echo ""

# Run container with:
# - GPU access (--runtime nvidia)
# - Mount current HRM directory
# - Interactive terminal
sudo docker run --runtime nvidia \
    -it \
    --rm \
    --network host \
    -v /home/sprout/ai-workspace/HRM:/workspace/HRM \
    -v /home/sprout/ai-workspace/datasets:/workspace/datasets \
    -w /workspace/HRM \
    dustynv/l4t-pytorch:r36.2.0 \
    bash -c "
        echo 'PyTorch version in container:'
        python3 -c 'import torch; print(torch.__version__)'
        echo ''
        echo 'CUDA available:'
        python3 -c 'import torch; print(torch.cuda.is_available())'
        echo ''
        echo 'Installing HRM dependencies...'
        pip3 install transformers einops
        echo ''
        echo 'Testing HRM...'
        python3 test_hrm_minimal.py
    "

echo ""
echo "=== Docker run complete ==="