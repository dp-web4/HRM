# Dropbox Integration for HRM

This directory manages large file synchronization across machines using Dropbox.

**Current Method**: `rclone` (simpler, no API keys needed)

## Quick Start with rclone

### 1. Install rclone
```bash
sudo apt-get update && sudo apt-get install -y rclone
```

### 2. Configure Dropbox
```bash
rclone config
```

**Interactive setup:**
- Choose `n` for new remote
- Name: `dropbox`
- Storage type: Select `dropbox` from list (usually #12)
- Leave `client_id` and `client_secret` **blank** (uses rclone's app)
- Advanced config: `n`
- Auto config: `y` (opens browser for OAuth)
- Authorize in browser
- Confirm: `y`

### 3. Verify Connection
```bash
rclone listremotes  # Should show: dropbox:
rclone ls dropbox:  # List your Dropbox contents
```

## Usage

### Upload Model to Dropbox
```bash
# From model-zoo
cd /home/dp/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism
rclone copy . dropbox:HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism -P

# Or using the Python wrapper
python3 /home/dp/ai-workspace/HRM/dropbox/rclone_sync.py upload /path/to/file
```

### Download Model from Dropbox
```bash
# To model-zoo on another machine
rclone copy dropbox:HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism \
  /home/dp/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism -P
```

### List Files in Dropbox
```bash
rclone ls dropbox:HRM/model-zoo/
rclone size dropbox:HRM/model-zoo/  # Show total size
```

### Sync Entire Directory
```bash
# Two-way sync (careful! can delete files)
rclone sync local-dir dropbox:HRM/path

# One-way copy (safer)
rclone copy local-dir dropbox:HRM/path -P
```

## Python Wrapper

Simple Python interface for common operations:

```bash
cd /home/dp/ai-workspace/HRM/dropbox

# Upload file
python3 rclone_sync.py upload /path/to/file

# Download file
python3 rclone_sync.py download HRM/path/to/file

# Sync all directories
python3 rclone_sync.py sync

# List remote files
python3 rclone_sync.py list checkpoints
```

## Dropbox Directory Structure

```
Dropbox:/HRM/
├── model-zoo/                    # Trained models (NEW)
│   └── sage/
│       └── epistemic-stances/
│           └── qwen2.5-0.5b/
│               └── epistemic-pragmatism/  # 1.855 GB
│                   ├── model.safetensors
│                   ├── tokenizer.json
│                   ├── metadata.json
│                   └── ... (11 files total)
│
├── checkpoints/                  # Training checkpoints
│   ├── hrm_arc_best.pt
│   └── ...
│
├── datasets/                     # Large datasets
├── logs/                        # Training logs
└── configs/                     # Shared configurations
```

## Machine Roles

- **Legion (RTX 4090)**: Primary training machine (uploads)
- **CBP (RTX 2060)**: Development machine (bidirectional)
- **Jetson (Orin Nano)**: Inference machine (downloads)

## Recent Uploads (2025-10-27)

✅ **qwen2.5-0.5b-epistemic-pragmatism** - 1.855 GB
- 99.5% reduction in performative denial
- 37x increase in epistemic pragmatism
- Fine-tuned from Qwen/Qwen2.5-0.5B-Instruct
- Ready for deployment on Jetson

## Legacy Method (dbxcli)

Old scripts using Dropbox API still available:
- `setup_dropbox.sh` - Install dbxcli
- `sync_dropbox.py` - Python API wrapper

**Note**: rclone is simpler and doesn't require API keys.

## Troubleshooting

### Check rclone config
```bash
rclone config show
```

### Test connection
```bash
rclone lsd dropbox:  # List top-level directories
```

### Re-authenticate
```bash
rclone config reconnect dropbox:
```

### Check bandwidth
```bash
rclone copy source dest -P --bwlimit 5M  # Limit to 5MB/s
```
