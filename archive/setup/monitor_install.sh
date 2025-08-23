#!/bin/bash
# Monitor PyTorch installation

echo "📊 Monitoring PyTorch Installation"
echo "=================================="
echo ""

while true; do
    if pgrep -f "pip.*torch" > /dev/null; then
        echo "⏳ Installation in progress..."
        tail -5 pytorch_install_direct.log 2>/dev/null | grep -v "^$"
    else
        if python3 -c "import torch" 2>/dev/null; then
            echo ""
            echo "✅ PyTorch installed successfully!"
            python3 -c "
import torch
print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
            break
        else
            echo "❌ Installation not running. Check pytorch_install_direct.log"
            tail -10 pytorch_install_direct.log 2>/dev/null
            break
        fi
    fi
    
    echo ""
    echo "Checking again in 30 seconds... (Press Ctrl+C to stop)"
    sleep 30
    clear
done