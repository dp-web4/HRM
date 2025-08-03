#!/bin/bash
# Monitor PyTorch installation progress

echo "🔍 Monitoring PyTorch Installation"
echo "=================================="
echo ""

while true; do
    # Clear screen for clean output
    clear
    
    echo "🔍 PyTorch Installation Monitor"
    echo "=================================="
    echo "Time: $(date)"
    echo ""
    
    # Check if pip process is running
    if pgrep -f "pip.*torch" > /dev/null; then
        echo "✅ Installation is running..."
        echo ""
        
        # Show download progress from log
        if [ -f pytorch_install.log ]; then
            echo "📊 Progress:"
            tail -5 pytorch_install.log | grep -E "(Downloading|Installing|Successfully|MB)" || echo "Downloading..."
            echo ""
        fi
        
        # Show disk usage
        echo "💾 Disk Usage:"
        df -h / | grep -v Filesystem
        echo ""
        
        # Estimate based on file size if downloading
        if [ -f pytorch_install.log ]; then
            LAST_LINE=$(tail -1 pytorch_install.log)
            if echo "$LAST_LINE" | grep -q "Downloading.*MB"; then
                echo "📦 Download size: 2462.4 MB"
                # Try to extract progress if shown
                echo "$LAST_LINE"
            fi
        fi
    else
        # Check if installation completed
        if python3 -c "import torch" 2>/dev/null; then
            echo "🎉 PyTorch is installed!"
            echo ""
            python3 -c "
import torch
print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
            break
        else
            echo "❌ Installation not running"
            echo ""
            echo "Check pytorch_install.log for errors"
            if [ -f pytorch_install.log ]; then
                echo ""
                echo "Last 5 lines of log:"
                tail -5 pytorch_install.log
            fi
        fi
    fi
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 30 seconds..."
    sleep 30
done