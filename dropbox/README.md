# Dropbox Integration for HRM

This directory manages large file synchronization across machines using Dropbox.

## Setup

1. **Install Dropbox CLI**:
   ```bash
   ./setup_dropbox.sh
   ```

2. **Authenticate**:
   ```bash
   dbxcli account
   # Follow the OAuth flow
   ```

3. **Test connection**:
   ```bash
   dbxcli ls /
   ```

## Usage

### Upload checkpoint from Legion:
```bash
python sync_dropbox.py upload --file checkpoints/hrm_arc_best.pt
```

### Download checkpoint on CBP/Jetson:
```bash
python sync_dropbox.py download --file checkpoints/hrm_arc_best.pt
```

### Auto-sync all checkpoints:
```bash
python sync_dropbox.py auto
```

## Directory Structure

```
dropbox/
├── checkpoints/     # Model weights (.pt, .pth)
├── datasets/        # Large datasets
├── logs/           # Training logs
└── configs/        # Shared configurations
```

## Machine Roles

- **Legion**: Primary training machine (uploads)
- **CBP**: Development machine (downloads)
- **Jetson**: Inference machine (downloads)
