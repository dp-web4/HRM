#!/usr/bin/env python3
"""
config_resolver.py — load YAML config, apply CLI overrides, and dump a resolved config
into the run logdir with a stable SHA256 hash.

Features
- Reads YAML (via PyYAML). If PyYAML isn't installed, falls back to a minimal parser
  that supports "key: value" and basic nested dicts via dot-keys in overrides.
- CLI overrides with dotted paths, e.g. --set training.batch_size=8 eval.fast_every=2000
- Type coercion: "true/false" → bool, int/float literals, and "null/none" → None
- Deep-merge strategy: CLI overrides take precedence over file values
- Writes both YAML and JSON to logdir: resolved_config.yaml / .json
- Emits a config_hash to stdout for easy logging

Usage
------
python config_resolver.py \
  --config path/to/config.yaml \
  --logdir runs/EXP-01_2025-09-02_1012 \
  --set training.batch_size=8 --set grad_accum_steps=5 --set eval_frequency=2000

Inside your train script:
-------------------------
from pathlib import Path
import json

# at startup, after creating LOGDIR:
#   subprocess.run([...,"config_resolver.py","--config",cfg,"--logdir",LOGDIR, ...])
# or call functions in this file directly if imported as a module.

"""

import os, sys, json, hashlib
from pathlib import Path

# ---------- YAML loader with fallback ----------
def _coerce_scalar(val: str):
    s = val.strip()
    low = s.lower()
    if low in ("true","false"):
        return low == "true"
    if low in ("null","none"):
        return None
    # int
    try:
        if re.match(r"^[+-]?\d+$", s):
            return int(s)
    except Exception:
        pass
    # float
    try:
        if re.match(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$", s):
            return float(s)
    except Exception:
        pass
    return s

def _load_yaml_fallback(text: str):
    # very naive: only supports "key: value" at top-level, ignores lists
    data = {}
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = _coerce_scalar(v)
    return data

def load_yaml(path: str):
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        with open(path, "r") as f:
            txt = f.read()
        return _load_yaml_fallback(txt)

# ---------- Deep merge & dotted override helpers ----------
def deep_merge(a, b):
    if not isinstance(b, dict):
        return b
    out = dict(a) if isinstance(a, dict) else {}
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def set_dotted(d, dotted_key, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def parse_overrides(pairs):
    ov = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Invalid override (missing '='): {p}")
        k, v = p.split("=", 1)
        set_dotted(ov, k.strip(), _coerce_scalar(v))
    return ov

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def dump_yaml(path: Path, obj):
    # Try to use PyYAML. If unavailable, write a simple key: value flat dump.
    try:
        import yaml  # type: ignore
        with open(path, "w") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
    except Exception:
        # naive flatten for top-level
        with open(path, "w") as f:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(obj))

def resolve_config(config_path: str, overrides: dict):
    base = load_yaml(config_path) if config_path else {}
    resolved = deep_merge(base, overrides)
    return resolved

def main():
    import argparse, re
    ap = argparse.ArgumentParser(description="Resolve config with overrides and write to logdir.")
    ap.add_argument("--config", type=str, default="", help="Path to YAML config file (optional).")
    ap.add_argument("--logdir", type=str, required=True, help="Directory to write resolved config.")
    ap.add_argument("--set", dest="sets", action="append", help="Override in form key=val (supports dotted keys).", default=[])
    ap.add_argument("--json_only", action="store_true", help="Write JSON only (skip YAML).")
    ap.add_argument("--yaml_only", action="store_true", help="Write YAML only (skip JSON).")
    args = ap.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.sets)
    resolved = resolve_config(args.config, overrides)

    # dump JSON
    if not args.yaml_only:
        jpath = Path(args.logdir) / "resolved_config.json"
        with open(jpath, "w") as f:
            json.dump(resolved, f, indent=2)
    # dump YAML
    if not args.json_only:
        ypath = Path(args.logdir) / "resolved_config.yaml"
        dump_yaml(ypath, resolved)

    # hash
    blob = json.dumps(resolved, sort_keys=True).encode("utf-8")
    h = sha256_of_bytes(blob)
    print(f"[config] hash={h}")
    print(f"[config] wrote: {Path(args.logdir) / 'resolved_config.json'}")
    print(f"[config] wrote: {Path(args.logdir) / 'resolved_config.yaml'}")

if __name__ == "__main__":
    main()
