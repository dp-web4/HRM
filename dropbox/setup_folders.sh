#!/bin/bash
# Create Dropbox folders for HRM

echo "📁 Creating Dropbox folders for HRM..."

# Check if authenticated
if ! dbxcli account > /dev/null 2>&1; then
    echo "❌ Not authenticated. Run: ./authenticate.sh"
    exit 1
fi

# Create folders
echo "Creating /HRM..."
dbxcli mkdir /HRM 2>/dev/null || echo "  /HRM already exists"

echo "Creating /HRM/checkpoints..."
dbxcli mkdir /HRM/checkpoints 2>/dev/null || echo "  /HRM/checkpoints already exists"

echo "Creating /HRM/datasets..."
dbxcli mkdir /HRM/datasets 2>/dev/null || echo "  /HRM/datasets already exists"

echo "Creating /HRM/logs..."
dbxcli mkdir /HRM/logs 2>/dev/null || echo "  /HRM/logs already exists"

echo "Creating /HRM/configs..."
dbxcli mkdir /HRM/configs 2>/dev/null || echo "  /HRM/configs already exists"

echo ""
echo "✅ Folders created. Testing access..."
dbxcli ls /HRM

echo ""
echo "🎉 Setup complete! You can now use:"
echo "  python3 dropbox/sync_dropbox.py upload --file <file>"
echo "  python3 dropbox/sync_dropbox.py download --file <file>"
echo "  python3 dropbox/sync_dropbox.py auto"
echo "  python3 dropbox/sync_dropbox.py watch"