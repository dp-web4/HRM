"""
Machine auto-detection and configuration for SAGE daemon.

Detects which machine SAGE is running on and loads the appropriate
configuration for model paths, memory limits, gateway ports, and
federation identity.

Fleet — default models (per-machine, override with SAGE_MODEL env var):

  Machine    | Hardware               | Default Model          | Device
  -----------|------------------------|------------------------|--------
  thor       | Jetson AGX Thor        | Qwen 2.5 14B (local)  | cuda
  sprout     | Jetson Orin Nano 8GB   | Qwen 2.5 0.5B (local) | cuda
  legion     | RTX 4090 desktop       | Qwen 2.5 14B (local)  | cuda
  mcnugget   | Mac Mini M4            | gemma3:12b (Ollama)    | mps
  nomad      | RTX 4060 laptop        | gemma3:4b (Ollama)     | cuda
  cbp        | RTX 2060S WSL2         | tinyllama:latest       | cpu

Each machine+model pairing maintains its own experience buffer.
SAGE_MODEL env var overrides the default for any machine.
"""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SAGEMachineConfig:
    """Configuration for a specific SAGE deployment machine."""
    machine_name: str           # "thor", "sprout", "cbp", "legion"
    model_path: str             # Path to Qwen model directory
    model_size: str             # "0.5b", "14b", "30b"
    device: str                 # "cuda", "cpu"
    max_memory_gb: float        # GPU/unified memory budget
    gateway_port: int           # HTTP port (default 8750)
    workspace_path: str         # Root workspace directory
    identity_state_path: str    # Path to state/identity.json
    experience_buffer_path: str # Path to state/experience_buffer.json
    irp_iterations: int         # IRP refinement iterations (3 for Sprout, 5 for Thor)
    federation_port: int        # Existing federation service port
    ed25519_key_path: str       # Path to platform Ed25519 signing key
    lct_id: str                 # Federation LCT identity
    system_prompt_mode: str     # "creative", "balanced", "honest"
    cycle_sleep_ms: int         # Consciousness loop cycle time in ms
    max_response_tokens: int    # Max tokens for LLM response generation


def _read_device_tree_model() -> str:
    """Read Jetson device tree model string (empty on non-Jetson)."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return f.read().strip('\x00').strip()
    except (FileNotFoundError, PermissionError):
        return ""


def _has_cuda() -> bool:
    """Check if CUDA is available without importing torch."""
    return Path('/usr/local/cuda/bin/nvcc').exists() or \
           Path('/usr/bin/nvcc').exists() or \
           os.path.exists('/proc/driver/nvidia/version')


def detect_machine() -> str:
    """
    Detect which machine we're running on.

    Detection order:
    1. SAGE_MACHINE environment variable (explicit override)
    2. /proc/device-tree/model (Jetson identification)
    3. Hostname pattern matching
    4. Workspace path existence checks
    """
    # Explicit override
    env_machine = os.environ.get('SAGE_MACHINE', '').lower()
    if env_machine in ('thor', 'sprout', 'cbp', 'legion', 'nomad', 'mcnugget'):
        return env_machine

    # Jetson device tree
    dt_model = _read_device_tree_model()
    if 'AGX' in dt_model or 'Thor' in dt_model:
        return 'thor'
    if 'Orin Nano' in dt_model or 'p3768' in dt_model:
        return 'sprout'

    # Hostname
    hostname = socket.gethostname().lower()
    if 'thor' in hostname:
        return 'thor'
    if hostname == 'ubuntu':
        # Sprout's default hostname — disambiguate by checking workspace
        if Path('/home/sprout/ai-workspace').exists():
            return 'sprout'
    if 'cbp' in hostname:
        return 'cbp'
    if 'legion' in hostname:
        return 'legion'
    if 'nomad' in hostname or 'desktop-9e6hcao' in hostname:
        return 'nomad'
    if 'mcnugget' in hostname:
        return 'mcnugget'

    # Workspace path fallback
    if Path('/home/dp/ai-workspace/HRM/sage/core').exists():
        # Could be Thor or Legion — check for Jetson indicators
        if Path('/etc/nv_tegra_release').exists():
            return 'thor'
        return 'legion'
    if Path('/home/sprout/ai-workspace').exists():
        return 'sprout'
    if Path('/mnt/c/exe/projects/ai-agents').exists():
        return 'cbp'

    return 'unknown'


def get_config(machine_name: Optional[str] = None) -> SAGEMachineConfig:
    """
    Get SAGE configuration for a machine.

    Args:
        machine_name: Machine name, or None to auto-detect

    Returns:
        SAGEMachineConfig for the detected/specified machine
    """
    if machine_name is None:
        machine_name = detect_machine()

    port = int(os.environ.get('SAGE_PORT', '8750'))

    # ── Per-machine configs ──────────────────────────────────────────
    # SAGE_MODEL env var overrides the default model on ANY machine.
    # For Ollama machines the value is an Ollama tag (e.g. "gemma3:4b").
    # For local-weight machines the value is a filesystem path.
    # Each machine+model pairing gets its own experience buffer file
    # (see sage_daemon.py _load_experience_collector).

    model_override = os.environ.get('SAGE_MODEL')

    if machine_name == 'thor':
        workspace = '/home/dp/ai-workspace'
        default_model = f'{workspace}/HRM/model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct'
        model = model_override or default_model
        is_ollama = model_override and not model_override.startswith('/')
        return SAGEMachineConfig(
            machine_name='thor',
            model_path=f'ollama:{model}' if is_ollama else model,
            model_size='ollama' if is_ollama else '14b',
            device='cuda',
            max_memory_gb=100.0,
            gateway_port=port,
            workspace_path=workspace,
            identity_state_path=f'{workspace}/HRM/sage/raising/state/identity.json',
            experience_buffer_path=f'{workspace}/HRM/sage/raising/state/experience_buffer.json',
            irp_iterations=5,
            federation_port=50051,
            ed25519_key_path=f'{workspace}/HRM/sage/data/keys/Thor_ed25519.key',
            lct_id='thor_sage_lct',
            system_prompt_mode='creative',
            cycle_sleep_ms=100,
            max_response_tokens=250,
        )

    elif machine_name == 'sprout':
        workspace = '/home/sprout/ai-workspace'
        default_model = f'{workspace}/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged'
        model = model_override or default_model
        is_ollama = model_override and not model_override.startswith('/')
        return SAGEMachineConfig(
            machine_name='sprout',
            model_path=f'ollama:{model}' if is_ollama else model,
            model_size='ollama' if is_ollama else '0.5b',
            device='cuda',
            max_memory_gb=6.0,
            gateway_port=port,
            workspace_path=workspace,
            identity_state_path=f'{workspace}/HRM/sage/raising/state/identity.json',
            experience_buffer_path=f'{workspace}/HRM/sage/raising/state/experience_buffer.json',
            irp_iterations=3,
            federation_port=50051,
            ed25519_key_path=f'{workspace}/HRM/sage/data/keys/Sprout_ed25519.key',
            lct_id='sprout_sage_lct',
            system_prompt_mode='creative',
            cycle_sleep_ms=100,
            max_response_tokens=150,
        )

    elif machine_name == 'legion':
        workspace = '/home/dp/ai-workspace'
        default_model = f'{workspace}/HRM/model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct'
        model = model_override or default_model
        is_ollama = model_override and not model_override.startswith('/')
        return SAGEMachineConfig(
            machine_name='legion',
            model_path=f'ollama:{model}' if is_ollama else model,
            model_size='ollama' if is_ollama else '14b',
            device='cuda',
            max_memory_gb=14.0,
            gateway_port=port,
            workspace_path=workspace,
            identity_state_path=f'{workspace}/HRM/sage/raising/state/identity.json',
            experience_buffer_path=f'{workspace}/HRM/sage/raising/state/experience_buffer.json',
            irp_iterations=5,
            federation_port=50051,
            ed25519_key_path=f'{workspace}/HRM/sage/data/keys/Legion_ed25519.key',
            lct_id='legion_sage_lct',
            system_prompt_mode='creative',
            cycle_sleep_ms=100,
            max_response_tokens=250,
        )

    elif machine_name == 'mcnugget':
        # McNugget: Mac Mini M4, 16GB unified
        workspace = '/Users/dennispalatov/repos'
        model = model_override or 'gemma3:12b'
        return SAGEMachineConfig(
            machine_name='mcnugget',
            model_path=f'ollama:{model}',
            model_size='ollama',
            device='mps',
            max_memory_gb=16.0,
            gateway_port=port,
            workspace_path=workspace,
            identity_state_path=f'{workspace}/HRM/sage/raising/state/identity.json',
            experience_buffer_path=f'{workspace}/HRM/sage/raising/state/experience_buffer.json',
            irp_iterations=5,
            federation_port=50051,
            ed25519_key_path=f'{workspace}/HRM/sage/data/keys/McNugget_ed25519.key',
            lct_id='mcnugget_sage_lct',
            system_prompt_mode='creative',
            cycle_sleep_ms=100,
            max_response_tokens=250,
        )

    elif machine_name == 'nomad':
        # Nomad: Legion laptop, RTX 4060 8GB
        workspace = '/mnt/c/projects/ai-agents'
        model = model_override or 'gemma3:4b'
        return SAGEMachineConfig(
            machine_name='nomad',
            model_path=f'ollama:{model}',
            model_size='ollama',
            device='cuda',
            max_memory_gb=8.0,
            gateway_port=port,
            workspace_path=workspace,
            identity_state_path=f'{workspace}/HRM/sage/raising/state/identity.json',
            experience_buffer_path=f'{workspace}/HRM/sage/raising/state/experience_buffer.json',
            irp_iterations=5,
            federation_port=50051,
            ed25519_key_path='',
            lct_id='nomad_sage_lct',
            system_prompt_mode='creative',
            cycle_sleep_ms=100,
            max_response_tokens=250,
        )

    elif machine_name == 'cbp':
        # CBP: WSL2 desktop, RTX 2060 SUPER
        workspace = '/mnt/c/exe/projects/ai-agents'
        model = model_override or 'tinyllama:latest'
        return SAGEMachineConfig(
            machine_name='cbp',
            model_path=f'ollama:{model}',
            model_size='ollama',
            device='cpu',
            max_memory_gb=8.0,
            gateway_port=port,
            workspace_path=workspace,
            identity_state_path=f'{workspace}/HRM/sage/raising/state/identity.json',
            experience_buffer_path=f'{workspace}/HRM/sage/raising/state/experience_buffer.json',
            irp_iterations=3,
            federation_port=0,
            ed25519_key_path='',
            lct_id='cbp_sage_lct',
            system_prompt_mode='creative',
            cycle_sleep_ms=100,
            max_response_tokens=500,
        )

    else:
        raise ValueError(
            f"Unknown machine: {machine_name}. "
            f"Set SAGE_MACHINE env var to one of: thor, sprout, cbp, legion, nomad, mcnugget"
        )


if __name__ == "__main__":
    machine = detect_machine()
    print(f"Detected machine: {machine}")

    config = get_config(machine)
    print(f"\nConfiguration:")
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        print(f"  {field_name}: {value}")
