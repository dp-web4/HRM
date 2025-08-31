# Dropbox Integration for HRM Cross-Machine Sharing

## Overview
This system enables automatic synchronization of large model checkpoints, datasets, and logs across Legion (training), CBP (development), and Jetson (inference) machines.

## Quick Setup

### 1. Install Dropbox CLI
```bash
cd HRM/dropbox
chmod +x setup_dropbox.sh
./setup_dropbox.sh
```

### 2. Authenticate
```bash
dbxcli account
# Follow the OAuth flow in your browser
```

### 3. Create HRM folder in Dropbox
```bash
dbxcli mkdir /HRM
dbxcli mkdir /HRM/checkpoints
dbxcli mkdir /HRM/datasets
dbxcli mkdir /HRM/logs
dbxcli mkdir /HRM/configs
```

## Machine-Specific Paths

### Legion (Training Machine)
```bash
# Upload checkpoint after training
python dropbox/sync_dropbox.py upload --file checkpoints/hrm_arc_best.pt

# Auto-upload all checkpoints
python dropbox/sync_dropbox.py sync --type checkpoints

# Watch mode - auto-upload every 60 seconds
python dropbox/sync_dropbox.py watch
```

### CBP/WSL2 (Development Machine)
```bash
# Download latest checkpoint
python dropbox/sync_dropbox.py download --file checkpoints/hrm_arc_best.pt

# Auto-sync all checkpoints
python dropbox/sync_dropbox.py auto

# List available checkpoints
python dropbox/sync_dropbox.py list --path /HRM/checkpoints
```

### Jetson (Inference Machine)
```bash
# Download optimized model
python dropbox/sync_dropbox.py download --file checkpoints/hrm_jetson.pt

# Download minimal inference configs
python dropbox/sync_dropbox.py sync --type configs
```

## Directory Structure

```
Dropbox/
└── HRM/
    ├── checkpoints/       # Model weights
    │   ├── hrm_arc_best.pt        # Best validation model
    │   ├── hrm_arc_step_1000.pt   # Checkpoint at step 1000
    │   └── hrm_jetson.pt          # Optimized for edge
    │
    ├── datasets/          # Shared datasets
    │   ├── arc-aug-1000.tar.gz    # Augmented ARC dataset
    │   └── arc-test.json           # Test set
    │
    ├── logs/             # Training logs
    │   ├── training_20241231.log
    │   └── tensorboard/
    │
    └── configs/          # Shared configurations
        ├── model_config.json
        └── training_config.json
```

## Automatic Sync Configuration

The `sync_config.json` defines machine roles:

- **Legion**: `auto_upload: true` - Automatically uploads new checkpoints
- **CBP**: `auto_download: true` - Automatically downloads new checkpoints
- **Jetson**: `auto_download: true` - Downloads optimized models

## Common Commands

### Upload checkpoint from Legion
```bash
# After training completes
python dropbox/sync_dropbox.py upload --file checkpoints/hrm_arc_best.pt
```

### Download on other machines
```bash
# Get the trained model
python dropbox/sync_dropbox.py download --file checkpoints/hrm_arc_best.pt
```

### Auto-sync based on machine role
```bash
# Legion will upload, others will download
python dropbox/sync_dropbox.py auto
```

### Watch mode for continuous sync
```bash
# Check every 60 seconds for changes
python dropbox/sync_dropbox.py watch --interval 60
```

### List remote files
```bash
# See what's available
python dropbox/sync_dropbox.py list --path /HRM/checkpoints
```

## File Size Limits

Configured in `sync_config.json`:
- Checkpoints: 5GB max
- Datasets: 10GB max
- Logs: 100MB max
- Configs: 10MB max

## Integration with Training

### Legion Training Script Addition
Add to `train_arc_legion.py`:
```python
# After saving checkpoint
if checkpoint_saved:
    os.system(f"python dropbox/sync_dropbox.py upload --file {checkpoint_path}")
```

### CBP/Jetson Evaluation Script
```python
# Auto-download latest checkpoint
os.system("python dropbox/sync_dropbox.py download --file checkpoints/hrm_arc_best.pt")

# Load model
model = torch.load("checkpoints/hrm_arc_best.pt")
```

## Sync Logs

Sync operations are logged to `dropbox/sync_logs/`:
```
2024-12-31T10:30:45 | upload   | Legion-Pro-7    | hrm_arc_best.pt | 108.5 MB | /HRM/checkpoints/hrm_arc_best.pt
2024-12-31T10:31:02 | download | CBP             | hrm_arc_best.pt | 108.5 MB | /HRM/checkpoints/hrm_arc_best.pt
```

## Troubleshooting

### Authentication Issues
```bash
# Re-authenticate
dbxcli account
dbxcli revoke  # If needed
dbxcli account  # Re-auth
```

### Check Connection
```bash
# Test Dropbox connection
dbxcli account
dbxcli ls /
```

### Manual Upload/Download
```bash
# Using dbxcli directly
dbxcli put local_file.pt /HRM/checkpoints/
dbxcli get /HRM/checkpoints/file.pt local_file.pt
```

## Security Notes

- The `.dropbox` folder and auth tokens are git-ignored
- Each machine authenticates independently
- Use separate app folders if needed for isolation
- Consider encryption for sensitive models

## Next Steps

1. Run `setup_dropbox.sh` on each machine
2. Authenticate with `dbxcli account`
3. Test with a small file upload/download
4. Set up watch mode on Legion for automatic uploads
5. Configure auto-download on CBP and Jetson