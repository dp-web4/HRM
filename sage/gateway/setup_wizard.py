"""
SAGE Setup Wizard — interactive first-launch configuration.

Detects the environment, guides through missing components (Ollama, models),
and produces a working ~/.sage/config.yaml so sage-daemon can start on any
machine — not just the fleet.

Usage:
    sage-setup              # Full interactive wizard
    sage-setup --check      # Detect and report only, no changes
    sage-setup --auto       # Non-interactive, pick defaults, write config
    sage-setup --reset      # Delete existing config, start fresh
"""

import json
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # checked at runtime


# ── Constants ────────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / '.sage'
CONFIG_PATH = CONFIG_DIR / 'config.yaml'
OLLAMA_HOST = 'http://localhost:11434'

# VRAM thresholds → (model tag, param description, approx download size)
_MODEL_SUGGESTIONS: List[Tuple[float, str, str, str]] = [
    (24.0, 'qwen2.5:32b',  '32B params', '~20 GB'),
    (12.0, 'phi4:14b',     '14B params', '~9 GB'),
    (6.0,  'gemma3:12b',   '12B params', '~7 GB'),
    (4.0,  'gemma3:4b',    '4B params',  '~3 GB'),
    (0.0,  'tinyllama:latest', '1.1B params', '~1 GB'),
]

_ADJECTIVES = [
    'nimble', 'swift', 'keen', 'bright', 'calm', 'bold', 'sharp', 'warm',
    'clear', 'quiet', 'steady', 'eager', 'gentle', 'lucid', 'witty',
]
_NOUNS = [
    'fox', 'owl', 'hawk', 'wolf', 'bear', 'lynx', 'tern', 'wren',
    'hare', 'dove', 'elk', 'moth', 'lark', 'newt', 'crow',
]


def _random_name() -> str:
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


def _sage_version() -> str:
    try:
        from sage import __version__
        return __version__
    except Exception:
        return 'unknown'


# ── Phase 1: Environment Detection ──────────────────────────────────────────

def _detect_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _detect_os() -> str:
    system = platform.system()
    release = platform.release()
    wsl = ''
    if system == 'Linux':
        try:
            proc_version = Path('/proc/version').read_text()
            if 'microsoft' in proc_version.lower():
                wsl = ' (WSL2)'
        except Exception:
            pass
    return f"{system} {release}{wsl}"


def _detect_ram_gb() -> float:
    try:
        if platform.system() == 'Linux':
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
        elif platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / 1024**3, 1)
    except Exception:
        pass
    return 0.0


def _detect_disk_free_gb() -> float:
    try:
        usage = shutil.disk_usage(Path.home())
        return round(usage.free / 1024**3, 1)
    except Exception:
        return 0.0


def _detect_ollama_installed() -> bool:
    return shutil.which('ollama') is not None


def _detect_ollama_running() -> Optional[List[Dict[str, Any]]]:
    """Probe Ollama /api/tags. Returns list of models if running, else None."""
    try:
        req = urllib.request.Request(f'{OLLAMA_HOST}/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get('models', [])
    except Exception:
        return None


def _detect_gpu() -> Tuple[str, str, float]:
    """Detect GPU. Returns (device, gpu_name, vram_gb).

    Tries torch first, then nvidia-smi fallback.
    """
    # Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            return 'cuda', name, vram
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps', 'Apple Silicon', 0.0
    except ImportError:
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            parts = [p.strip() for p in line.split(',')]
            name = parts[0]
            vram = round(float(parts[1]) / 1024, 1) if len(parts) > 1 else 0.0
            return 'cuda', name, vram
    except Exception:
        pass

    # macOS Metal (no torch)
    if platform.system() == 'Darwin':
        return 'mps', 'Apple Silicon (no torch)', 0.0

    return 'cpu', '', 0.0


def _detect_pytorch() -> Optional[str]:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None


def detect_environment() -> Dict[str, Any]:
    """Run all environment detections. Never raises."""
    ollama_installed = _detect_ollama_installed()
    ollama_models_raw = _detect_ollama_running() if ollama_installed else None
    ollama_running = ollama_models_raw is not None
    device, gpu_name, vram_gb = _detect_gpu()
    pytorch_ver = _detect_pytorch()

    # Parse model list
    models = []
    if ollama_models_raw:
        for m in ollama_models_raw:
            name = m.get('name', m.get('model', ''))
            size_bytes = m.get('size', 0)
            size_gb = round(size_bytes / 1024**3, 1) if size_bytes else 0.0
            models.append({'name': name, 'size_gb': size_gb})

    existing_config = None
    if CONFIG_PATH.exists():
        try:
            existing_config = yaml.safe_load(CONFIG_PATH.read_text()) if yaml else None
        except Exception:
            existing_config = None

    return {
        'python': _detect_python(),
        'os': _detect_os(),
        'ram_gb': _detect_ram_gb(),
        'disk_free_gb': _detect_disk_free_gb(),
        'ollama_installed': ollama_installed,
        'ollama_running': ollama_running,
        'models': models,
        'device': device,
        'gpu_name': gpu_name,
        'vram_gb': vram_gb,
        'pytorch': pytorch_ver,
        'sage_version': _sage_version(),
        'existing_config': existing_config,
    }


def print_environment(env: Dict[str, Any]) -> None:
    """Pretty-print the environment scan results."""
    v = _sage_version()
    print(f"\n  SAGE Setup Wizard v{v}\n")
    print("  Scanning environment...\n")

    print(f"  Python       {env['python']}")
    print(f"  OS           {env['os']}")
    print(f"  RAM          {env['ram_gb']} GB")
    print(f"  Disk (free)  {env['disk_free_gb']} GB")
    print(f"  ---")

    # Ollama status
    if not env['ollama_installed']:
        print(f"  Ollama       not installed")
    elif not env['ollama_running']:
        print(f"  Ollama       installed, not running")
    else:
        print(f"  Ollama       installed, running")

    # Models
    if env['models']:
        for i, m in enumerate(env['models']):
            prefix = "  Models       " if i == 0 else "               "
            sz = f"({m['size_gb']} GB)" if m['size_gb'] else ""
            print(f"{prefix}{m['name']} {sz}")
    else:
        print(f"  Models       none")

    print(f"  ---")

    # GPU
    if env['gpu_name']:
        vram_str = f" ({env['vram_gb']} GB VRAM)" if env['vram_gb'] else ""
        print(f"  GPU          {env['gpu_name']}{vram_str}")
    else:
        print(f"  GPU          none detected")

    if env['pytorch']:
        print(f"  PyTorch      {env['pytorch']}")
    else:
        print(f"  PyTorch      not installed")

    print(f"  ---")
    print(f"  SAGE         {env['sage_version']}")
    if env['existing_config']:
        inst = env['existing_config'].get('instance', {}).get('name', '?')
        print(f"  Config       {CONFIG_PATH} ({inst})")
    else:
        print(f"  Config       none (first run)")
    print()


# ── Phase 2: Interactive Prompts ─────────────────────────────────────────────

def _input_default(prompt: str, default: str) -> str:
    """Prompt with a default value in brackets."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    return raw if raw else default


def _input_yn(prompt: str, default: bool = True) -> bool:
    """Yes/no prompt."""
    hint = 'Y/n' if default else 'y/N'
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


def _suggest_model(vram_gb: float) -> Tuple[str, str, str]:
    """Pick a model suggestion based on VRAM."""
    for threshold, tag, desc, size in _MODEL_SUGGESTIONS:
        if vram_gb >= threshold:
            return tag, desc, size
    return _MODEL_SUGGESTIONS[-1][1], _MODEL_SUGGESTIONS[-1][2], _MODEL_SUGGESTIONS[-1][3]


def prompt_ollama_install(env: Dict[str, Any]) -> Dict[str, Any]:
    """Handle Ollama not installed / not running. Returns updated env."""
    if not env['ollama_installed']:
        print("  Ollama is required for SAGE. Install it:\n")
        print("    Linux/WSL:  curl -fsSL https://ollama.com/install.sh | sh")
        print("    macOS:      brew install ollama")
        print("    Manual:     https://ollama.com/download")
        print()
        print("  After installing, run: ollama serve\n")

        while True:
            raw = input("  Press Enter when ready (or 'skip' to continue without)... ").strip()
            if raw.lower() == 'skip':
                return env
            # Re-probe
            env['ollama_installed'] = _detect_ollama_installed()
            if env['ollama_installed']:
                env['models'] = _detect_ollama_running() or []
                env['ollama_running'] = bool(env['models'])
                if env['ollama_running']:
                    print("  Ollama detected and running!\n")
                    # Re-parse model list
                    raw_models = _detect_ollama_running()
                    if raw_models:
                        env['models'] = [
                            {'name': m.get('name', m.get('model', '')),
                             'size_gb': round(m.get('size', 0) / 1024**3, 1)}
                            for m in raw_models
                        ]
                    return env
                else:
                    print("  Ollama installed but not running. Start it: ollama serve\n")
            else:
                print("  Ollama not found yet. Install and try again.\n")

    if env['ollama_installed'] and not env['ollama_running']:
        print("  Ollama is installed but not running.")
        print("  Start it with: ollama serve\n")
        while True:
            raw = input("  Press Enter when ready (or 'skip' to continue without)... ").strip()
            if raw.lower() == 'skip':
                return env
            raw_models = _detect_ollama_running()
            if raw_models is not None:
                env['ollama_running'] = True
                env['models'] = [
                    {'name': m.get('name', m.get('model', '')),
                     'size_gb': round(m.get('size', 0) / 1024**3, 1)}
                    for m in raw_models
                ]
                print("  Ollama is running!\n")
                return env
            print("  Still not running. Check: ollama serve\n")

    return env


def prompt_model_selection(env: Dict[str, Any]) -> str:
    """Interactive model selection. Returns chosen model tag."""
    models = env.get('models', [])

    if not models:
        # No models — suggest one
        tag, desc, size = _suggest_model(env.get('vram_gb', 0.0))
        hw_info = f"{env['vram_gb']} GB VRAM" if env['vram_gb'] else "no GPU detected"
        print(f"  No models found. Recommended for your hardware ({hw_info}):")
        print(f"    {tag} — {desc}, {size}\n")

        if _input_yn("Pull now?", default=True):
            print(f"  Running: ollama pull {tag}\n")
            try:
                subprocess.run(['ollama', 'pull', tag], check=False)
                print()
            except Exception as e:
                print(f"  [WARN] Pull failed: {e}")
                print(f"  Pull manually: ollama pull {tag}\n")
            return tag
        else:
            # Let them type a custom model
            custom = input("  Model tag to use (e.g. gemma3:4b): ").strip()
            return custom if custom else tag
    else:
        # Models exist — let them pick
        print("  Available models:")
        for i, m in enumerate(models, 1):
            sz = f"({m['size_gb']} GB)" if m['size_gb'] else ""
            print(f"    [{i}] {m['name']} {sz}")
        print(f"    [{len(models) + 1}] Pull a different model...")
        print()

        raw = input(f"  Select [1]: ").strip()
        if not raw:
            idx = 0
        else:
            try:
                idx = int(raw) - 1
            except ValueError:
                idx = 0

        if 0 <= idx < len(models):
            return models[idx]['name']
        else:
            # Pull different model
            tag, desc, size = _suggest_model(env.get('vram_gb', 0.0))
            custom = input(f"  Model tag to pull (default: {tag}): ").strip()
            tag = custom if custom else tag
            print(f"  Running: ollama pull {tag}\n")
            try:
                subprocess.run(['ollama', 'pull', tag], check=False)
                print()
            except Exception as e:
                print(f"  [WARN] Pull failed: {e}")
            return tag


def prompt_port_and_name(env: Dict[str, Any]) -> Tuple[int, str]:
    """Prompt for dashboard port and instance name."""
    default_name = _random_name()
    # Preserve existing name/port if re-running
    existing = env.get('existing_config') or {}
    existing_name = existing.get('instance', {}).get('name', default_name)
    existing_port = str(existing.get('gateway', {}).get('port', 8750))

    port_str = _input_default("Dashboard port", existing_port)
    try:
        port = int(port_str)
    except ValueError:
        port = 8750

    name = _input_default("Instance name", existing_name)
    return port, name


# ── Phase 3: Config Write + Probe + Summary ─────────────────────────────────

def _count_available_tools() -> int:
    """Count registered built-in tools (best-effort)."""
    try:
        from sage.tools.builtin import BUILTIN_TOOLS
        return len(BUILTIN_TOOLS)
    except Exception:
        return 0


def probe_and_write_config(
    env: Dict[str, Any],
    model: str,
    port: int,
    name: str,
) -> Dict[str, Any]:
    """Probe model capabilities, write config, return config dict."""
    # Probe tool capability
    tier = 'T3'
    grammar = 'intent_heuristic'
    tool_count = _count_available_tools()

    print(f"  Probing {model} for tool capabilities...")
    try:
        from sage.tools.tool_capability import ToolCapability
        cap = ToolCapability.detect(model, OLLAMA_HOST)
        tier = cap.tier
        grammar = cap.grammar_id
        print(f"    Tier:    {tier} ({_tier_label(tier)})")
        print(f"    Grammar: {grammar}")
        if tool_count:
            print(f"    Tools:   {tool_count} available")
    except Exception as e:
        print(f"    [WARN] Probe failed: {e}")
        print(f"    Using defaults: {tier}, {grammar}")
    print()

    now = datetime.now().isoformat(timespec='seconds')

    # Build config dict
    config: Dict[str, Any] = {
        'instance': {
            'name': name,
        },
        'ollama': {
            'host': OLLAMA_HOST,
            'model': model,
        },
        'gateway': {
            'port': port,
        },
        'hardware': {
            'device': env['device'],
            'gpu_name': env.get('gpu_name', ''),
            'vram_gb': env.get('vram_gb', 0.0),
            'ram_gb': env.get('ram_gb', 0.0),
        },
        'tools': {
            'tier': tier,
            'grammar': grammar,
        },
        'setup': {
            'version': _sage_version(),
            'created_at': now,
            'updated_at': now,
        },
    }

    # Preserve created_at on re-run
    existing = env.get('existing_config') or {}
    if existing.get('setup', {}).get('created_at'):
        config['setup']['created_at'] = existing['setup']['created_at']

    # Write config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if yaml:
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: write YAML by hand (simple flat structure)
        _write_yaml_manual(CONFIG_PATH, config)

    print(f"  Config saved to {CONFIG_PATH}\n")
    return config


def _write_yaml_manual(path: Path, data: Dict[str, Any], indent: int = 0) -> None:
    """Write nested dict as YAML without pyyaml (fallback)."""
    lines: List[str] = []

    def _dump(d: Dict[str, Any], level: int) -> None:
        prefix = '  ' * level
        for key, val in d.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}{key}:")
                _dump(val, level + 1)
            elif isinstance(val, bool):
                lines.append(f"{prefix}{key}: {'true' if val else 'false'}")
            elif isinstance(val, (int, float)):
                lines.append(f"{prefix}{key}: {val}")
            else:
                lines.append(f'{prefix}{key}: "{val}"')

    _dump(data, 0)
    path.write_text('\n'.join(lines) + '\n')


def _tier_label(tier: str) -> str:
    return {
        'T1': 'native tool calling',
        'T2': 'grammar-guided',
        'T3': 'heuristic',
    }.get(tier, tier)


def _device_label(env: Dict[str, Any]) -> str:
    device = env.get('device', 'cpu')
    gpu = env.get('gpu_name', '')
    vram = env.get('vram_gb', 0.0)
    if device == 'cpu':
        return 'cpu'
    parts = [device]
    if gpu:
        parts.append(f"({gpu}")
        if vram:
            parts.append(f"{vram} GB)")
        else:
            parts[-1] += ')'
    return ' '.join(parts)


def print_summary(env: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Print the final summary banner."""
    model = config['ollama']['model']
    tier = config['tools']['tier']
    name = config['instance']['name']
    port = config['gateway']['port']

    print(f"  {'━' * 40}\n")
    print(f"  Instance     {name}")
    print(f"  Model        {model} ({tier})")
    print(f"  Device       {_device_label(env)}")
    print(f"  Port         {port}")
    print()
    print(f"  Start SAGE:  sage-daemon")
    print(f"  Dashboard:   http://localhost:{port}/")
    print(f"  Chat (CLI):  sage-cli \"Hello SAGE\"")
    print()


# ── Config Loader (used by sage_daemon.py) ───────────────────────────────────

def load_config() -> Optional[Dict[str, Any]]:
    """Load ~/.sage/config.yaml. Returns None if missing or invalid."""
    if not CONFIG_PATH.exists():
        return None
    try:
        if yaml:
            return yaml.safe_load(CONFIG_PATH.read_text())
        else:
            return _parse_yaml_manual(CONFIG_PATH)
    except Exception:
        return None


def _parse_yaml_manual(path: Path) -> Dict[str, Any]:
    """Minimal YAML parser for the simple nested-dict config (no pyyaml)."""
    result: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, result)]

    for line in path.read_text().splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())
        # Pop stack to correct level
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        parent = stack[-1][1]

        if ':' in stripped:
            key, _, val = stripped.partition(':')
            key = key.strip()
            val = val.strip().strip('"').strip("'")

            if not val:
                # Nested dict
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                # Scalar value
                if val == 'true':
                    parent[key] = True
                elif val == 'false':
                    parent[key] = False
                else:
                    try:
                        parent[key] = int(val)
                    except ValueError:
                        try:
                            parent[key] = float(val)
                        except ValueError:
                            parent[key] = val

    return result


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for sage-setup."""
    args = sys.argv[1:]

    # --reset: delete config and start fresh
    if '--reset' in args:
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
            print(f"  Deleted {CONFIG_PATH}")
        args = [a for a in args if a != '--reset']
        if not args:
            # Continue into full wizard
            pass

    # Phase 1: detect
    env = detect_environment()
    print_environment(env)

    # --check: detect only, no changes
    if '--check' in args:
        return

    # Check pyyaml
    if yaml is None:
        print("  [WARN] pyyaml not installed. Using built-in YAML writer.")
        print("  Install for best results: pip install pyyaml\n")

    # --auto: non-interactive, pick defaults
    if '--auto' in args:
        _run_auto(env)
        return

    # Phase 2: interactive prompts
    env = prompt_ollama_install(env)

    if env.get('ollama_running') and not env.get('models'):
        # Ollama is running but no models — re-probe to refresh
        raw_models = _detect_ollama_running()
        if raw_models:
            env['models'] = [
                {'name': m.get('name', m.get('model', '')),
                 'size_gb': round(m.get('size', 0) / 1024**3, 1)}
                for m in raw_models
            ]

    if env.get('ollama_running') or env.get('ollama_installed'):
        model = prompt_model_selection(env)
    else:
        print("  Continuing without Ollama. You can configure a model later.\n")
        model = input("  Model tag (or press Enter to skip): ").strip() or 'tinyllama:latest'

    port, name = prompt_port_and_name(env)

    # Phase 3: probe, write, summarize
    config = probe_and_write_config(env, model, port, name)
    print_summary(env, config)


def _run_auto(env: Dict[str, Any]) -> None:
    """Non-interactive mode: pick best defaults, write config."""
    # Pick model
    models = env.get('models', [])
    if models:
        model = models[0]['name']
    else:
        model, _, _ = _suggest_model(env.get('vram_gb', 0.0))

    # Preserve existing name/port or generate new
    existing = env.get('existing_config') or {}
    name = existing.get('instance', {}).get('name', _random_name())
    port = existing.get('gateway', {}).get('port', 8750)

    config = probe_and_write_config(env, model, port, name)
    print_summary(env, config)


if __name__ == '__main__':
    main()
