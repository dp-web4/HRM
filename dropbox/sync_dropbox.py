#!/usr/bin/env python3
"""
Dropbox synchronization for HRM checkpoints and large files
Handles upload/download across Legion, CBP, and Jetson machines
"""

import os
import sys
import json
import subprocess
import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import socket

class DropboxSync:
    def __init__(self, config_path: str = "dropbox/sync_config.json"):
        """Initialize Dropbox sync with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        self.hostname = socket.gethostname()
        self.machine_config = self.get_machine_config()
        
    def load_config(self):
        """Load sync configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def get_machine_config(self) -> Dict:
        """Get configuration for current machine"""
        for machine_name, machine_config in self.config['machines'].items():
            if self.hostname.lower().startswith(machine_config['hostname'].lower()):
                print(f"🖥️  Detected machine: {machine_name}")
                return machine_config
        
        print(f"⚠️  Unknown machine: {self.hostname}, using default settings")
        return {'role': 'unknown', 'auto_upload': False, 'auto_download': True}
    
    def check_dbxcli(self) -> bool:
        """Check if dbxcli is installed and authenticated"""
        try:
            result = subprocess.run(['dbxcli', 'account'], 
                                  capture_output=True, text=True)
            if 'Email:' in result.stdout or 'Logged in as' in result.stdout:
                return True
            else:
                print("❌ dbxcli not authenticated. Run: dbxcli account")
                return False
        except FileNotFoundError:
            print("❌ dbxcli not found. Run: ./setup_dropbox.sh")
            return False
    
    def get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file for comparison"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def upload_file(self, local_path: Path, dropbox_path: str, force: bool = False):
        """Upload file to Dropbox"""
        if not local_path.exists():
            print(f"❌ File not found: {local_path}")
            return False
        
        # Check file size
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"📤 Uploading {local_path.name} ({size_mb:.1f} MB)")
        
        # Check if file already exists in Dropbox
        if not force:
            result = subprocess.run(['dbxcli', 'ls', dropbox_path],
                                  capture_output=True, text=True)
            if local_path.name in result.stdout:
                print(f"⚠️  File already exists in Dropbox. Use --force to overwrite")
                return False
        
        # Upload file
        cmd = ['dbxcli', 'put', str(local_path), dropbox_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Uploaded successfully to {dropbox_path}")
            
            # Log upload
            self.log_sync('upload', local_path, dropbox_path, size_mb)
            return True
        else:
            print(f"❌ Upload failed: {result.stderr}")
            return False
    
    def download_file(self, dropbox_path: str, local_path: Path, force: bool = False):
        """Download file from Dropbox"""
        # Check if local file exists
        if local_path.exists() and not force:
            print(f"⚠️  Local file already exists: {local_path}")
            print("   Use --force to overwrite")
            return False
        
        # Create directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading {dropbox_path}")
        
        # Download file
        cmd = ['dbxcli', 'get', dropbox_path, str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✅ Downloaded successfully ({size_mb:.1f} MB)")
            
            # Log download
            self.log_sync('download', local_path, dropbox_path, size_mb)
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
    
    def sync_directory(self, sync_type: str):
        """Sync entire directory based on configuration"""
        if sync_type not in self.config['sync_paths']:
            print(f"❌ Unknown sync type: {sync_type}")
            return
        
        sync_config = self.config['sync_paths'][sync_type]
        local_dir = Path(sync_config['local'])
        dropbox_dir = sync_config['dropbox']
        patterns = sync_config['patterns']
        
        print(f"🔄 Syncing {sync_type}...")
        
        # Get list of files matching patterns
        files_to_sync = []
        if local_dir.exists():
            for pattern in patterns:
                files_to_sync.extend(local_dir.glob(pattern))
        
        # Upload or download based on machine role
        if self.machine_config.get('auto_upload', False):
            # Upload files (training machine)
            for file in files_to_sync:
                dropbox_path = f"{dropbox_dir}/{file.name}"
                self.upload_file(file, dropbox_path)
        
        elif self.machine_config.get('auto_download', False):
            # Download files (inference/dev machines)
            # List files in Dropbox directory
            result = subprocess.run(['dbxcli', 'ls', dropbox_dir],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse file list
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            filename = parts[3]
                            # Check if matches pattern
                            for pattern in patterns:
                                if filename.endswith(pattern.replace('*', '')):
                                    local_file = local_dir / filename
                                    dropbox_file = f"{dropbox_dir}/{filename}"
                                    if not local_file.exists():
                                        self.download_file(dropbox_file, local_file)
    
    def auto_sync(self):
        """Automatically sync all configured directories"""
        print(f"🤖 Starting auto-sync for {self.hostname}")
        
        for sync_type in self.config['sync_paths']:
            self.sync_directory(sync_type)
        
        print("✅ Auto-sync complete")
    
    def watch_and_sync(self, interval: int = 60):
        """Watch for changes and sync periodically"""
        print(f"👁️  Watching for changes every {interval} seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.auto_sync()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 Stopping watch mode")
    
    def list_remote(self, path: str = "/HRM"):
        """List files in Dropbox"""
        print(f"📂 Listing {path}")
        result = subprocess.run(['dbxcli', 'ls', path],
                              capture_output=True, text=True)
        print(result.stdout)
    
    def log_sync(self, action: str, local_path: Path, dropbox_path: str, size_mb: float):
        """Log sync operations"""
        log_dir = Path("dropbox/sync_logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"sync_{datetime.now().strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} | {action:8} | "
                   f"{self.hostname:15} | {local_path.name:30} | "
                   f"{size_mb:8.1f} MB | {dropbox_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Dropbox sync for HRM")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload file to Dropbox')
    upload_parser.add_argument('--file', required=True, help='File to upload')
    upload_parser.add_argument('--dest', help='Dropbox destination (optional)')
    upload_parser.add_argument('--force', action='store_true', help='Overwrite if exists')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download file from Dropbox')
    download_parser.add_argument('--file', required=True, help='File to download')
    download_parser.add_argument('--dest', help='Local destination (optional)')
    download_parser.add_argument('--force', action='store_true', help='Overwrite if exists')
    
    # Auto sync
    auto_parser = subparsers.add_parser('auto', help='Auto-sync based on machine role')
    
    # Watch mode
    watch_parser = subparsers.add_parser('watch', help='Watch and sync periodically')
    watch_parser.add_argument('--interval', type=int, default=60, help='Sync interval in seconds')
    
    # List remote
    list_parser = subparsers.add_parser('list', help='List Dropbox contents')
    list_parser.add_argument('--path', default='/HRM', help='Dropbox path to list')
    
    # Sync specific directory
    sync_parser = subparsers.add_parser('sync', help='Sync specific directory')
    sync_parser.add_argument('--type', required=True, 
                            choices=['checkpoints', 'datasets', 'logs', 'configs'],
                            help='Type of data to sync')
    
    args = parser.parse_args()
    
    # Initialize sync
    sync = DropboxSync()
    
    # Check dbxcli
    if not sync.check_dbxcli():
        sys.exit(1)
    
    # Execute command
    if args.command == 'upload':
        local_path = Path(args.file)
        if args.dest:
            dropbox_path = args.dest
        else:
            # Auto-determine based on file type
            if local_path.suffix in ['.pt', '.pth', '.ckpt']:
                dropbox_path = f"/HRM/checkpoints/{local_path.name}"
            elif local_path.suffix in ['.json', '.npz', '.pkl']:
                dropbox_path = f"/HRM/datasets/{local_path.name}"
            else:
                dropbox_path = f"/HRM/{local_path.name}"
        
        sync.upload_file(local_path, dropbox_path, args.force)
    
    elif args.command == 'download':
        if args.file.startswith('/'):
            dropbox_path = args.file
        else:
            # Assume it's in HRM folder
            dropbox_path = f"/HRM/{args.file}"
        
        if args.dest:
            local_path = Path(args.dest)
        else:
            # Auto-determine local path
            filename = os.path.basename(dropbox_path)
            if 'checkpoint' in dropbox_path:
                local_path = Path('checkpoints') / filename
            elif 'dataset' in dropbox_path:
                local_path = Path('data') / filename
            else:
                local_path = Path(filename)
        
        sync.download_file(dropbox_path, local_path, args.force)
    
    elif args.command == 'auto':
        sync.auto_sync()
    
    elif args.command == 'watch':
        sync.watch_and_sync(args.interval)
    
    elif args.command == 'list':
        sync.list_remote(args.path)
    
    elif args.command == 'sync':
        sync.sync_directory(args.type)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()