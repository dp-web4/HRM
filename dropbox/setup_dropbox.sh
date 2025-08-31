#!/bin/bash
# Dropbox integration setup for HRM model sharing across machines

echo "📦 Setting up Dropbox integration for HRM..."

# Install Dropbox CLI if not present
if ! command -v dbxcli &> /dev/null; then
    echo "Installing Dropbox CLI..."
    
    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    if [ "$ARCH" = "x86_64" ]; then
        ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ]; then
        ARCH="arm64"
    fi
    
    # Download appropriate binary
    DBXCLI_URL="https://github.com/dropbox/dbxcli/releases/latest/download/dbxcli-${OS}-${ARCH}"
    
    echo "Downloading from: $DBXCLI_URL"
    wget -O /tmp/dbxcli "$DBXCLI_URL"
    chmod +x /tmp/dbxcli
    sudo mv /tmp/dbxcli /usr/local/bin/dbxcli
    
    echo "✅ Dropbox CLI installed"
    echo "Run 'dbxcli account' to authenticate"
else
    echo "✅ Dropbox CLI already installed"
fi

# Create Dropbox sync directories
echo "Creating Dropbox sync structure..."
mkdir -p dropbox/checkpoints
mkdir -p dropbox/datasets
mkdir -p dropbox/logs
mkdir -p dropbox/configs

# Create sync configuration
cat > dropbox/sync_config.json << 'EOF'
{
  "sync_paths": {
    "checkpoints": {
      "local": "./checkpoints",
      "dropbox": "/HRM/checkpoints",
      "patterns": ["*.pt", "*.pth", "*.ckpt"],
      "max_size_mb": 5000
    },
    "datasets": {
      "local": "./data",
      "dropbox": "/HRM/datasets",
      "patterns": ["*.json", "*.npz", "*.pkl"],
      "max_size_mb": 10000
    },
    "logs": {
      "local": "./logs",
      "dropbox": "/HRM/logs",
      "patterns": ["*.log", "*.txt", "events.out.tfevents.*"],
      "max_size_mb": 100
    },
    "configs": {
      "local": "./configs",
      "dropbox": "/HRM/configs",
      "patterns": ["*.json", "*.yaml", "*.txt"],
      "max_size_mb": 10
    }
  },
  "machines": {
    "legion": {
      "hostname": "Legion-Pro-7",
      "role": "training",
      "auto_upload": true
    },
    "cbp": {
      "hostname": "CBP",
      "role": "development",
      "auto_download": true
    },
    "jetson": {
      "hostname": "jetson-orin",
      "role": "inference",
      "auto_download": true
    }
  }
}
EOF

echo "✅ Sync configuration created"

# Create .gitignore for Dropbox files
cat > dropbox/.gitignore << 'EOF'
# Ignore synced files
checkpoints/*
datasets/*
logs/*
!*.md
!.gitkeep

# Dropbox metadata
.dropbox
.dropbox.cache
.dropbox.attr
EOF

# Create README
cat > dropbox/README.md << 'EOF'
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
EOF

echo "📝 Creating .gitkeep files..."
touch dropbox/checkpoints/.gitkeep
touch dropbox/datasets/.gitkeep
touch dropbox/logs/.gitkeep
touch dropbox/configs/.gitkeep

echo "✅ Dropbox structure created"
echo ""
echo "Next steps:"
echo "1. Authenticate: dbxcli account"
echo "2. Create HRM folder in Dropbox: dbxcli mkdir /HRM"
echo "3. Run sync script: python dropbox/sync_dropbox.py"