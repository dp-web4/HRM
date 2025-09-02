#!/usr/bin/env python3
"""
Simple Dropbox sync using rclone - works with ANY Dropbox account!
No API keys needed - rclone handles everything.
"""

import subprocess
import sys
import os
from pathlib import Path
import json

class RcloneSync:
    def __init__(self):
        self.remote = "dropbox"
        self.base_path = "HRM"
        
    def check_rclone(self):
        """Check if rclone is configured"""
        try:
            result = subprocess.run(['rclone', 'listremotes'], 
                                  capture_output=True, text=True)
            if f"{self.remote}:" in result.stdout:
                return True
            else:
                print(f"❌ Rclone remote '{self.remote}' not configured")
                print("Run: ./dropbox/setup_rclone.sh")
                return False
        except FileNotFoundError:
            print("❌ rclone not installed")
            print("Run: sudo apt-get install rclone")
            return False
    
    def upload(self, local_file, remote_path=None):
        """Upload file to Dropbox"""
        if not self.check_rclone():
            return False
        
        local_file = Path(local_file)
        if not local_file.exists():
            print(f"❌ File not found: {local_file}")
            return False
        
        if remote_path is None:
            # Auto-determine remote path based on file type
            if local_file.suffix in ['.pt', '.pth', '.ckpt']:
                remote_path = f"{self.base_path}/checkpoints/{local_file.name}"
            elif local_file.suffix in ['.json', '.yaml']:
                remote_path = f"{self.base_path}/configs/{local_file.name}"
            elif local_file.suffix in ['.log', '.txt']:
                remote_path = f"{self.base_path}/logs/{local_file.name}"
            else:
                remote_path = f"{self.base_path}/{local_file.name}"
        
        cmd = ['rclone', 'copy', str(local_file), 
               f"{self.remote}:{Path(remote_path).parent}", '-P']
        
        print(f"📤 Uploading {local_file} to dropbox:{remote_path}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"✅ Uploaded successfully")
            return True
        else:
            print(f"❌ Upload failed")
            return False
    
    def download(self, remote_file, local_path=None):
        """Download file from Dropbox"""
        if not self.check_rclone():
            return False
        
        if not remote_file.startswith(self.base_path):
            remote_file = f"{self.base_path}/{remote_file}"
        
        if local_path is None:
            # Auto-determine local path
            filename = Path(remote_file).name
            if 'checkpoints' in remote_file:
                local_path = f"checkpoints/{filename}"
            elif 'configs' in remote_file:
                local_path = f"configs/{filename}"
            elif 'logs' in remote_file:
                local_path = f"logs/{filename}"
            else:
                local_path = filename
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ['rclone', 'copy', f"{self.remote}:{remote_file}", 
               str(local_path.parent), '-P']
        
        print(f"📥 Downloading dropbox:{remote_file} to {local_path}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"✅ Downloaded successfully")
            return True
        else:
            print(f"❌ Download failed")
            return False
    
    def sync_all(self):
        """Sync all directories"""
        if not self.check_rclone():
            return False
        
        dirs = ['checkpoints', 'configs', 'logs', 'datasets']
        
        for dir_name in dirs:
            local_dir = Path(dir_name)
            if local_dir.exists():
                print(f"🔄 Syncing {dir_name}...")
                cmd = ['rclone', 'sync', str(local_dir), 
                       f"{self.remote}:{self.base_path}/{dir_name}", 
                       '-P', '--exclude', '*.tmp']
                subprocess.run(cmd)
        
        print("✅ Sync complete")
        return True
    
    def list_files(self, path=""):
        """List files in Dropbox"""
        if not self.check_rclone():
            return False
        
        full_path = f"{self.base_path}/{path}" if path else self.base_path
        cmd = ['rclone', 'ls', f"{self.remote}:{full_path}"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to list files")
            return False

def main():
    sync = RcloneSync()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 rclone_sync.py upload <file>")
        print("  python3 rclone_sync.py download <remote_file>")
        print("  python3 rclone_sync.py sync")
        print("  python3 rclone_sync.py list")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "upload" and len(sys.argv) > 2:
        sync.upload(sys.argv[2])
    elif command == "download" and len(sys.argv) > 2:
        sync.download(sys.argv[2])
    elif command == "sync":
        sync.sync_all()
    elif command == "list":
        path = sys.argv[2] if len(sys.argv) > 2 else ""
        sync.list_files(path)
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()