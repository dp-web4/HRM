# SAGE Daemon Setup Guide

Run the full SAGE consciousness loop as an always-on service with HTTP dashboard.

## Prerequisites

- Python 3.10+
- PyTorch (with CUDA, MPS, or CPU backend)
- Ollama (for Ollama-backed models) or local model weights
- `psutil` (for system stats on dashboard)

## Quick Start

```bash
cd /path/to/HRM

# Auto-detect machine and start
python3 -m sage.gateway.sage_daemon

# Or specify machine explicitly
SAGE_MACHINE=mcnugget python3 -m sage.gateway.sage_daemon

# Custom port
SAGE_PORT=9000 python3 -m sage.gateway.sage_daemon
```

The daemon starts the consciousness loop, HTTP gateway, and dashboard on port 8750 (default).

- **Dashboard**: http://localhost:8750/
- **Health**: http://localhost:8750/health
- **Chat**: POST http://localhost:8750/chat `{"message": "Hello SAGE"}`
- **Status**: http://localhost:8750/status

---

## Platform Setup

### Linux (CUDA)

Tested on Ubuntu 22.04+ with NVIDIA GPUs (Jetson, RTX 2060/4060/4090).

**1. Install dependencies**

```bash
cd HRM
pip install -r sage_requirements_minimal.txt
pip install psutil
```

**2. Install PyTorch**

```bash
# CUDA 12.x (RTX 4090, Jetson AGX Thor)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.x (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Set up local model (Qwen via Transformers)**

Download model weights to a local path and configure in `machine_config.py`, or use Ollama (see below).

**4. Run as systemd service** (optional)

```ini
# /etc/systemd/system/sage.service
[Unit]
Description=SAGE Consciousness Daemon
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/HRM
Environment=SAGE_MACHINE=your_machine
Environment=PYTHONPATH=/path/to/HRM
ExecStart=/usr/bin/python3 -m sage.gateway.sage_daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable sage
sudo systemctl start sage
journalctl -u sage -f  # View logs
```

---

### macOS (Apple Silicon / MPS)

Tested on Mac Mini M4 (16GB unified memory) with macOS Sequoia.

**1. Install dependencies**

```bash
# Homebrew Python (recommended over system Python)
brew install python@3.14

cd HRM
pip3 install -r sage_requirements_minimal.txt
pip3 install psutil
```

**2. Install PyTorch with MPS**

```bash
pip3 install torch torchvision torchaudio
# MPS backend is auto-detected on Apple Silicon
```

Verify:
```bash
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

**3. Install Ollama**

```bash
brew install ollama
brew services start ollama
ollama pull gemma3:12b  # or any model
```

**4. Environment variables**

macOS with Homebrew Python + PyTorch requires two OpenMP fixes:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE   # Duplicate libomp from brew + torch
export OMP_NUM_THREADS=1            # pthread_mutex_init crash in asyncio
```

These are set automatically by `sage_daemon.py` at import time, but must also be in the launchd plist for service mode.

**5. Run as launchd service** (optional)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.sage.daemon</string>

    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>-m</string>
        <string>sage.gateway.sage_daemon</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/HRM</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>SAGE_MACHINE</key>
        <string>your_machine</string>
        <key>PYTHONPATH</key>
        <string>/path/to/HRM</string>
        <key>KMP_DUPLICATE_LIB_OK</key>
        <string>TRUE</string>
        <key>OMP_NUM_THREADS</key>
        <string>1</string>
        <key>SAGE_NO_BROWSER</key>
        <string>1</string>
    </dict>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/sage_daemon.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/sage_daemon_error.log</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
```

```bash
mkdir -p ~/Library/Logs/sage
cp sage.plist ~/Library/LaunchAgents/com.sage.daemon.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.sage.daemon.plist
launchctl kickstart gui/$(id -u)/com.sage.daemon

# View logs
tail -f /tmp/sage_daemon.log

# Stop
launchctl bootout gui/$(id -u)/com.sage.daemon
```

---

### WSL2 (Windows)

Tested on WSL2 Ubuntu 22.04 with RTX 2060 SUPER.

**1. Install dependencies**

```bash
cd HRM
pip install -r sage_requirements_minimal.txt
pip install psutil
```

**2. Install PyTorch**

```bash
# With CUDA (if nvidia-smi works in WSL2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. Install Ollama** (if using Ollama-backed models)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama  # or any model
```

**4. Run**

```bash
SAGE_MACHINE=your_machine python3 -m sage.gateway.sage_daemon
```

WSL2 note: the daemon opens a browser on startup. Set `SAGE_NO_BROWSER=1` if running headless or without a Windows display server.

---

## Machine Configuration

### Adding a New Machine

Edit `sage/gateway/machine_config.py`:

**1. Add hostname detection** in `detect_machine()`:
```python
if 'your-hostname' in hostname:
    return 'your_machine'
```

**2. Add config block** in `get_config()`:
```python
elif machine == 'your_machine':
    config = SAGEMachineConfig(
        machine_name='your_machine',
        model_path='ollama:gemma3:12b',   # or local path
        model_size='ollama',               # or '0.5b', '14b', etc.
        device='mps',                      # 'cuda', 'mps', or 'cpu'
        max_memory_gb=16.0,
        gateway_port=port,
        workspace_path='/path/to/workspace',
        irp_iterations=5,
        lct_id='your_machine_sage_lct',
        max_response_tokens=250,
    )
```

### Model Backends

| `model_size` | Backend | `model_path` Format | Requirements |
|---|---|---|---|
| `'ollama'` | Ollama HTTP API | `'ollama:model_name'` | Ollama running on localhost:11434 |
| `'0.5b'`, `'14b'`, etc. | Transformers (local) | `/path/to/model/weights` | Model files on disk, PyTorch + CUDA |

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SAGE_MACHINE` | auto-detect | Override machine detection |
| `SAGE_PORT` | 8750 | HTTP gateway port |
| `SAGE_MODEL` | from config | Override model path |
| `SAGE_BIND_HOST` | `0.0.0.0` | Bind address |
| `SAGE_NO_BROWSER` | unset | Skip auto-opening dashboard |

---

## Dashboard

The web dashboard shows live SAGE stats via Server-Sent Events (1Hz):

- **Metabolic state**: WAKE / FOCUS / REST / DREAM / CRISIS with color-coded badge
- **ATP budget**: Current energy level, drains during WAKE, recharges during REST
- **GPU/Memory**: VRAM (CUDA) or unified memory (Apple Silicon)
- **Plugin trust**: Per-plugin trust scores from the IRP orchestrator
- **Salience**: Average SNARC attention score
- **System stats**: CPU%, RAM, uptime, message counts

---

## Troubleshooting

### macOS: Segfault on startup
Set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` before running. The daemon sets these automatically, but launchd services need them in the plist.

### No GPU stats on dashboard
- **CUDA**: Ensure `torch.cuda.is_available()` returns True
- **MPS**: Requires `psutil` installed — unified memory stats are reported as GPU
- **CPU-only**: GPU panel shows "N/A"

### Port already in use
```bash
lsof -i :8750  # Find what's using the port
kill <PID>     # or change port with SAGE_PORT
```

### Ollama connection refused
```bash
ollama list          # Verify Ollama is running
curl localhost:11434 # Test Ollama API
ollama pull model    # Ensure model is downloaded
```
