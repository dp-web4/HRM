#!/bin/bash
#
# SAGE Installation Script for Jetson Platforms
#
# Installs the SAGE consciousness kernel with Track 7 LLM integration
# on Jetson Nano, Jetson Orin Nano, or Jetson AGX platforms.
#
# Usage:
#   ./install_sage_nano.sh [--config path/to/config.yaml]
#
# Requirements:
#   - JetPack 5.0+ (Ubuntu 20.04+)
#   - Python 3.8+
#   - 8GB+ storage available
#   - Internet connection for model downloads
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SAGE_DIR}/sage_nano.yaml"
VENV_DIR="${SAGE_DIR}/sage_venv"
MODELS_DIR="${SAGE_DIR}/model-zoo"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_platform() {
    log_info "Detecting Jetson platform..."

    if [ -f /etc/nv_tegra_release ]; then
        PLATFORM=$(cat /etc/nv_tegra_release | grep -oP '(?<=# R)[0-9]+')
        log_success "Detected Jetson platform (JetPack R${PLATFORM})"
    else
        log_error "Not running on Jetson platform"
        exit 1
    fi

    # Detect specific hardware
    if grep -q "Orin" /proc/device-tree/model 2>/dev/null; then
        JETSON_MODEL="orin"
        log_info "Hardware: Jetson Orin series"
    elif grep -q "Nano" /proc/device-tree/model 2>/dev/null; then
        JETSON_MODEL="nano"
        log_info "Hardware: Jetson Nano"
    elif grep -q "AGX" /proc/device-tree/model 2>/dev/null; then
        JETSON_MODEL="agx"
        log_info "Hardware: Jetson AGX series"
    else
        JETSON_MODEL="unknown"
        log_warning "Unknown Jetson model, proceeding with generic setup"
    fi
}

check_dependencies() {
    log_info "Checking system dependencies..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+')
    log_success "Found Python ${PYTHON_VERSION}"

    # Check for required system packages
    REQUIRED_PKGS=("git" "wget" "curl")
    MISSING_PKGS=()

    for pkg in "${REQUIRED_PKGS[@]}"; do
        if ! command -v "$pkg" &> /dev/null; then
            MISSING_PKGS+=("$pkg")
        fi
    done

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        log_warning "Missing packages: ${MISSING_PKGS[*]}"
        log_info "Installing missing packages..."
        sudo apt-get update
        sudo apt-get install -y "${MISSING_PKGS[@]}"
    fi

    log_success "System dependencies OK"
}

setup_python_env() {
    log_info "Setting up Python virtual environment..."

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_success "Created virtual environment at $VENV_DIR"
    else
        log_info "Using existing virtual environment"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    log_success "Python environment ready"
}

install_pytorch() {
    log_info "Installing PyTorch for Jetson..."

    # Check if PyTorch is already installed
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        log_success "PyTorch ${TORCH_VERSION} already installed"
        return 0
    fi

    # Install PyTorch from NVIDIA JetPack wheel
    log_info "Downloading PyTorch wheel for Jetson..."

    # PyTorch 2.1.0 for JetPack 5.x
    if [ "${PLATFORM}" -ge "35" ]; then
        TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    else
        log_warning "Using generic ARM wheel for older JetPack version"
        pip install torch torchvision torchaudio
        return 0
    fi

    wget "$TORCH_WHEEL" -O /tmp/torch.whl
    pip install /tmp/torch.whl
    rm /tmp/torch.whl

    log_success "PyTorch installed"
}

install_python_deps() {
    log_info "Installing Python dependencies..."

    # Core dependencies
    pip install transformers>=4.35.0
    pip install peft>=0.6.0
    pip install accelerate>=0.24.0
    pip install safetensors>=0.4.0
    pip install sentencepiece>=0.1.99

    # Optional but recommended
    pip install numpy>=1.24.0
    pip install scipy>=1.10.0
    pip install pyyaml>=6.0
    pip install tqdm>=4.65.0

    log_success "Python dependencies installed"
}

download_models() {
    log_info "Setting up model zoo..."

    # Create models directory
    mkdir -p "$MODELS_DIR"

    # Check if base model exists
    BASE_MODEL_DIR="${MODELS_DIR}/Qwen/Qwen2.5-0.5B-Instruct"

    if [ ! -d "$BASE_MODEL_DIR" ]; then
        log_info "Base model not found. It will be downloaded on first use."
        log_info "Estimated download: ~500MB"
    else
        log_success "Base model already present"
    fi

    # Create model cache directory
    export HF_HOME="${MODELS_DIR}/.cache"
    mkdir -p "$HF_HOME"

    log_success "Model zoo configured"
}

create_config() {
    log_info "Creating configuration file..."

    if [ -f "$CONFIG_FILE" ]; then
        log_info "Configuration already exists, skipping"
        return 0
    fi

    cat > "$CONFIG_FILE" <<EOF
# SAGE Configuration for Jetson Nano
# Track 7: LLM Integration

# Model Configuration
model:
  # Base model path (Hugging Face model ID or local path)
  model_path: "Qwen/Qwen2.5-0.5B-Instruct"

  # LoRA adapter path (optional, leave blank for base model)
  adapter_path: ""

  # Device selection (auto, cuda, cpu)
  device: "auto"

  # Max tokens to generate per response
  max_tokens: 200

# IRP Configuration
irp:
  # Number of refinement iterations
  iterations: 5

  # Initial temperature for sampling
  initial_temperature: 0.7

  # Minimum temperature (convergence floor)
  min_temperature: 0.5

  # Temperature reduction per iteration
  temp_reduction: 0.04

# SNARC Configuration
snarc:
  # Salience threshold (0.0-1.0)
  # Higher = more selective memory
  threshold: 0.15

  # Dimension weights (must sum to 1.0)
  weights:
    surprise: 0.2
    novelty: 0.2
    arousal: 0.2
    reward: 0.2
    conflict: 0.2

# Conversation Configuration
conversation:
  # Maximum conversation history to keep in context
  max_history: 5

  # Whether to include history in prompts
  include_history: true

# Memory Configuration
memory:
  # Where to store conversation data
  storage_path: "./conversation_memory.db"

  # Export salient exchanges for training
  auto_export: true
  export_path: "./training_data.jsonl"

# Performance Configuration
performance:
  # Enable FP16 inference (faster, less memory)
  use_fp16: true

  # Batch size (usually 1 for conversation)
  batch_size: 1

  # Enable torch.compile (requires PyTorch 2.0+)
  use_compile: false

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "./sage_nano.log"
  console: true
EOF

    log_success "Created configuration: $CONFIG_FILE"
}

run_smoke_tests() {
    log_info "Running smoke tests..."

    # Test 1: Import check
    log_info "Test 1: Checking Python imports..."
    python3 <<EOF
import sys
sys.path.insert(0, '$SAGE_DIR')
try:
    import torch
    from sage.irp.plugins.llm_impl import ConversationalLLM
    from sage.irp.plugins.llm_snarc_integration import ConversationalMemory
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        log_success "Test 1: PASSED"
    else
        log_error "Test 1: FAILED"
        return 1
    fi

    # Test 2: CUDA availability
    log_info "Test 2: Checking CUDA availability..."
    python3 -c "import torch; print('CUDA available:',  torch.cuda.is_available())"

    # Test 3: Configuration loading
    log_info "Test 3: Checking configuration..."
    if [ -f "$CONFIG_FILE" ]; then
        log_success "Test 3: PASSED - Config exists"
    else
        log_error "Test 3: FAILED - Config missing"
        return 1
    fi

    log_success "All smoke tests passed!"
}

create_systemd_service() {
    log_info "Setting up systemd service (optional)..."

    # This is optional - user might not want auto-start
    read -p "Create systemd service for auto-start? (y/n) " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping systemd service creation"
        return 0
    fi

    SERVICE_FILE="/etc/systemd/system/sage-nano.service"

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=SAGE Consciousness Kernel
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SAGE_DIR
ExecStart=$VENV_DIR/bin/python $SAGE_DIR/sage/run_sage_nano.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    log_success "Systemd service created: $SERVICE_FILE"
    log_info "Enable with: sudo systemctl enable sage-nano"
    log_info "Start with: sudo systemctl start sage-nano"
}

print_summary() {
    echo ""
    echo "=========================================="
    log_success "SAGE Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source $VENV_DIR/bin/activate"
    echo "  2. Edit configuration: nano $CONFIG_FILE"
    echo "  3. Run live demo: python sage/tests/live_demo_llm_irp.py"
    echo "  4. View documentation: cat sage/irp/TRACK7_LLM_INTEGRATION.md"
    echo ""
    echo "Quick start:"
    echo "  cd $SAGE_DIR"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python sage/tests/live_demo_llm_irp.py"
    echo ""
    log_info "For multi-session learning, see sage/experiments/sprout-validation/"
    echo ""
}

# Main installation flow
main() {
    echo "=========================================="
    echo "  SAGE Jetson Installer"
    echo "  Track 7: LLM Integration"
    echo "=========================================="
    echo ""

    check_platform
    check_dependencies
    setup_python_env
    install_pytorch
    install_python_deps
    download_models
    create_config
    run_smoke_tests
    create_systemd_service
    print_summary
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config path/to/config.yaml]"
            echo ""
            echo "Installs SAGE consciousness kernel on Jetson platforms"
            echo ""
            echo "Options:"
            echo "  --config FILE   Use custom configuration file"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main installation
main
