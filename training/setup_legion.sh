#!/bin/bash
# Quick setup script for Legion RTX 4090 ARC training

echo "🚀 Setting up HRM ARC training on Legion..."

# Create virtual environment if it doesn't exist
if [ ! -d "arc_training_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv arc_training_env
fi

# Activate environment
source arc_training_env/bin/activate

echo "📦 Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installing dependencies..."
pip install numpy pandas tqdm tensorboard argdantic pydantic einops accelerate

echo "✅ Verifying GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "📊 Preparing directories..."
mkdir -p data/arc-aug-1000
mkdir -p checkpoints
mkdir -p logs

echo "⚠️  Next steps:"
echo "1. Download ARC dataset:"
echo "   mkdir -p dataset/raw-data && cd dataset/raw-data"
echo "   git clone https://github.com/fchollet/ARC-AGI.git"
echo "   cd ../.."
echo ""
echo "2. Generate augmented dataset:"
echo "   python dataset/build_arc_dataset.py --output-dir data/arc-aug-1000 --num-aug 1000"
echo ""
echo "3. Start training:"
echo "   python training/train_arc_legion.py"
echo ""
echo "✅ Setup complete!"