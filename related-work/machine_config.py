#!/usr/bin/env python3
"""
Machine-agnostic configuration for SAGE-Totality integration.
Detects machine capabilities and configures accordingly.
"""

import os
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

class MachineConfig:
    """Detect and configure for different machine environments."""
    
    @staticmethod
    def detect_machine() -> Dict[str, Any]:
        """Detect current machine capabilities."""
        config = {
            "hostname": platform.node(),
            "os": platform.system(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cuda_available": False,
            "gpu_name": None,
            "cpu_count": os.cpu_count(),
            "memory_gb": None,
            "machine_profile": "unknown"
        }
        
        # Check for CUDA/GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            config["cuda_available"] = True
            config["gpu_name"] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check for Jetson
        if Path("/etc/nv_tegra_release").exists():
            config["machine_profile"] = "jetson"
            config["cuda_available"] = True  # Jetson has integrated GPU
            config["gpu_name"] = "Jetson Integrated GPU"
        
        # Try to get memory info
        try:
            if config["os"] == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            config["memory_gb"] = round(kb / (1024 * 1024), 1)
                            break
        except:
            pass
        
        # Determine machine profile
        hostname = config["hostname"].lower()
        if "legion" in hostname:
            config["machine_profile"] = "legion"
        elif "jetson" in hostname or "ubuntu" in hostname:
            if config.get("gpu_name") and "Jetson" in config.get("gpu_name", ""):
                config["machine_profile"] = "jetson"
        elif "cbp" in hostname or "desktop" in hostname:
            config["machine_profile"] = "windows_wsl"
        elif "laptop" in hostname:
            config["machine_profile"] = "laptop"
        
        return config
    
    @staticmethod
    def get_optimal_settings(machine_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal settings based on machine capabilities."""
        
        # Default settings (conservative)
        settings = {
            "batch_size": 1,
            "num_workers": 0,
            "use_cuda": False,
            "device": "cpu",
            "sleep_cycle_count": 10,
            "augmentation_count": 5,
            "h_module_lr": 0.01,
            "l_module_lr": 0.001,
            "max_memory_items": 1000,
            "service_port": 8080,
            "service_host": "localhost",
            "enable_profiling": False
        }
        
        # Adjust based on machine profile
        profile = machine_config.get("machine_profile", "unknown")
        
        if profile == "legion":
            # High-performance Linux with RTX 4090
            settings.update({
                "batch_size": 32,
                "num_workers": 4,
                "use_cuda": True,
                "device": "cuda",
                "sleep_cycle_count": 100,
                "augmentation_count": 50,
                "max_memory_items": 10000,
                "enable_profiling": True
            })
        elif profile == "jetson":
            # Edge device with integrated GPU
            settings.update({
                "batch_size": 8,
                "num_workers": 2,
                "use_cuda": True,
                "device": "cuda",
                "sleep_cycle_count": 20,
                "augmentation_count": 10,
                "max_memory_items": 5000
            })
        elif profile == "windows_wsl":
            # Windows WSL, may have GPU passthrough
            if machine_config.get("cuda_available"):
                settings.update({
                    "batch_size": 16,
                    "num_workers": 2,
                    "use_cuda": True,
                    "device": "cuda",
                    "sleep_cycle_count": 50,
                    "augmentation_count": 20
                })
            else:
                settings.update({
                    "batch_size": 4,
                    "num_workers": 1,
                    "sleep_cycle_count": 20,
                    "augmentation_count": 10
                })
        elif profile == "laptop":
            # Mobile device, conservative settings
            settings.update({
                "batch_size": 2,
                "num_workers": 1,
                "sleep_cycle_count": 10,
                "augmentation_count": 5,
                "max_memory_items": 2000
            })
        
        # Adjust based on available memory
        memory_gb = machine_config.get("memory_gb", 0)
        if memory_gb and memory_gb > 32:
            settings["max_memory_items"] *= 2
            settings["batch_size"] = min(settings["batch_size"] * 2, 64)
        elif memory_gb and memory_gb < 8:
            settings["max_memory_items"] //= 2
            settings["batch_size"] = max(settings["batch_size"] // 2, 1)
        
        return settings

    @staticmethod
    def save_config(config: Dict[str, Any], settings: Dict[str, Any], 
                    filepath: str = "machine_config.json"):
        """Save configuration to file."""
        full_config = {
            "machine": config,
            "settings": settings,
            "timestamp": str(Path(filepath).stat().st_mtime if Path(filepath).exists() else "new")
        }
        with open(filepath, "w") as f:
            json.dump(full_config, f, indent=2)
        return filepath
    
    @staticmethod
    def load_config(filepath: str = "machine_config.json") -> Optional[Dict[str, Any]]:
        """Load configuration from file if it exists."""
        if Path(filepath).exists():
            with open(filepath) as f:
                return json.load(f)
        return None


def setup_for_current_machine():
    """Main setup function for current machine."""
    print("🔍 Detecting machine configuration...")
    
    config = MachineConfig.detect_machine()
    print(f"📦 Machine Profile: {config['machine_profile']}")
    print(f"🖥️  Hostname: {config['hostname']}")
    print(f"🐍 Python: {config['python_version']}")
    print(f"🎮 GPU: {config.get('gpu_name', 'Not detected')}")
    print(f"💾 Memory: {config.get('memory_gb', 'Unknown')} GB")
    
    print("\n⚙️  Determining optimal settings...")
    settings = MachineConfig.get_optimal_settings(config)
    
    print(f"📊 Settings:")
    print(f"  - Device: {settings['device']}")
    print(f"  - Batch Size: {settings['batch_size']}")
    print(f"  - Sleep Cycles: {settings['sleep_cycle_count']}")
    print(f"  - Augmentations: {settings['augmentation_count']}")
    
    # Save configuration
    config_file = MachineConfig.save_config(config, settings)
    print(f"\n💾 Configuration saved to: {config_file}")
    
    return config, settings


if __name__ == "__main__":
    setup_for_current_machine()