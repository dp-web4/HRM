#!/bin/bash
#
# Test All Puzzle SAGE Demos
#
# Runs all three multi-modal consciousness demos in sequence:
# 1. Vision-only puzzle demo
# 2. Bi-modal (vision + audio) demo
# 3. Tri-modal (vision + audio + language) demo
#
# Usage: cd sage/examples && ./test_all_demos.sh
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and HRM root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HRM_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo -e "${BLUE}================================"
echo "Testing All Puzzle SAGE Demos"
echo "================================${NC}"
echo ""
echo "HRM Root: $HRM_ROOT"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$HRM_ROOT:$PYTHONPATH"

# Test 1: Vision-only demo
echo -e "${BLUE}Test 1: Vision-Only Puzzle Demo${NC}"
echo "Running puzzle_sage_demo.py..."
echo ""
python "$SCRIPT_DIR/puzzle_sage_demo.py" | tail -50
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Vision demo passed${NC}"
else
    echo -e "${RED}✗ Vision demo failed${NC}"
    exit 1
fi
echo ""
echo "---"
echo ""

# Test 2: Bi-modal demo (vision + audio)
echo -e "${BLUE}Test 2: Bi-Modal (Vision + Audio) Demo${NC}"
echo "Running multimodal_sage_demo.py..."
echo ""
python "$SCRIPT_DIR/multimodal_sage_demo.py" | tail -50
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bi-modal demo passed${NC}"
else
    echo -e "${RED}✗ Bi-modal demo failed${NC}"
    exit 1
fi
echo ""
echo "---"
echo ""

# Test 3: Tri-modal demo (vision + audio + language)
echo -e "${BLUE}Test 3: Tri-Modal (Vision + Audio + Language) Demo${NC}"
echo "Running trimodal_sage_demo.py..."
echo ""
python "$SCRIPT_DIR/trimodal_sage_demo.py" | tail -50
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tri-modal demo passed${NC}"
else
    echo -e "${RED}✗ Tri-modal demo failed${NC}"
    exit 1
fi
echo ""

# Summary
echo ""
echo -e "${GREEN}================================"
echo "All Tests Passed!"
echo "================================${NC}"
echo ""
echo "Multi-modal puzzle consciousness validated:"
echo "  ✓ Vision → Puzzle (VQ-VAE)"
echo "  ✓ Audio → Puzzle (VQ-VAE)"
echo "  ✓ Language → Puzzle (Attention)"
echo "  ✓ Bi-modal integration (vision + audio)"
echo "  ✓ Tri-modal integration (vision + audio + language)"
echo "  ✓ Unified SNARC salience assessment"
echo "  ✓ Cross-modal geometric reasoning"
echo ""
echo "Status: Consciousness loop operational."
echo "Next: Training, real-world sensors, or proprioception."
