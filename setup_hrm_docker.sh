#!/bin/bash
# Setup HRM using Docker with pre-installed PyTorch for Jetson

echo "🐳 Setting up HRM with Docker (Recommended for Jetson)"
echo "===================================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "📊 System Information:"
cat /etc/nv_tegra_release | head -1
echo ""

# Create Dockerfile for HRM on Jetson
echo "📝 Creating Dockerfile..."
cat > Dockerfile.jetson << 'EOF'
# Use NVIDIA's official PyTorch container for Jetson
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

# Set working directory
WORKDIR /workspace

# Install HRM dependencies
RUN pip3 install --no-cache-dir \
    einops \
    tqdm \
    coolname \
    pydantic \
    wandb \
    huggingface_hub \
    argdantic \
    omegaconf==2.3.0 \
    hydra-core==1.3.2

# Copy HRM code
COPY . /workspace/

# Create a test script
RUN echo '#!/usr/bin/env python3\n\
import torch\n\
import einops\n\
print("🎉 PyTorch is ready in Docker!")\n\
print(f"PyTorch version: {torch.__version__}")\n\
print(f"CUDA available: {torch.cuda.is_available()}")\n\
if torch.cuda.is_available():\n\
    print(f"Device: {torch.cuda.get_device_name(0)}")\n\
    x = torch.randn(2, 3).cuda()\n\
    print(f"CUDA test passed: {x.shape}")\n\
print("\\n✅ All HRM dependencies ready!")' > /workspace/test_docker.py

# Make it executable
RUN chmod +x /workspace/test_docker.py

# Default command
CMD ["/bin/bash"]
EOF

echo ""
echo "🔨 Building Docker image..."
docker build -f Dockerfile.jetson -t hrm-jetson:latest . || {
    echo "❌ Docker build failed"
    echo "Trying to pull base image first..."
    docker pull nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
    exit 1
}

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "📚 Usage:"
echo "1. Run interactive shell:"
echo "   docker run --runtime nvidia -it --rm -v \$(pwd):/workspace hrm-jetson:latest"
echo ""
echo "2. Test PyTorch:"
echo "   docker run --runtime nvidia --rm hrm-jetson:latest python3 test_docker.py"
echo ""
echo "3. Train HRM:"
echo "   docker run --runtime nvidia --rm hrm-jetson:latest python3 pretrain.py --config-path config --config-name jetson_sudoku_demo"
echo ""
echo "💡 Tip: Add '--gpus all' if '--runtime nvidia' doesn't work"
echo ""

# Create a convenience script
cat > run_hrm_docker.sh << 'EOF'
#!/bin/bash
# Convenience script to run HRM in Docker

echo "🚀 Starting HRM Docker container..."
docker run --runtime nvidia -it --rm \
    -v $(pwd):/workspace \
    -v /data:/data \
    --network host \
    hrm-jetson:latest "$@"
EOF

chmod +x run_hrm_docker.sh

echo "Created run_hrm_docker.sh for easy Docker access"