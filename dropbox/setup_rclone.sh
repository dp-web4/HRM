#!/bin/bash
# Setup rclone for Dropbox - much simpler!

echo "=== Setting up Dropbox with rclone ==="
echo ""
echo "This is MUCH simpler than the Dropbox API!"
echo ""
echo "1. Run: rclone config"
echo "2. Choose 'n' for new remote"
echo "3. Name it: dropbox"
echo "4. Choose 'dropbox' from the list"
echo "5. Leave client_id and client_secret blank (uses rclone's)"
echo "6. Choose 'n' for advanced config"
echo "7. Choose 'y' for auto config"
echo "8. It will open a browser - authorize it"
echo "9. Choose 'y' to confirm"
echo ""
echo "After setup, test with:"
echo "  rclone ls dropbox:"
echo "  rclone mkdir dropbox:HRM"
echo ""
echo "Ready to run setup? (y/n)"
read -r response

if [[ "$response" == "y" ]]; then
    rclone config
    echo ""
    echo "Testing connection..."
    rclone mkdir dropbox:HRM
    rclone mkdir dropbox:HRM/checkpoints
    rclone mkdir dropbox:HRM/datasets
    rclone mkdir dropbox:HRM/logs
    rclone mkdir dropbox:HRM/configs
    echo ""
    echo "Listing Dropbox contents:"
    rclone ls dropbox:HRM
    echo ""
    echo "✅ Setup complete!"
fi