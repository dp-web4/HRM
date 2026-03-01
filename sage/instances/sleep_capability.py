"""
Sleep capability detection and dream bundle management.

Runtime detection of what sleep modes are available on this machine,
plus dream bundle export for Ollama-only nodes that can't run LoRA locally.

Capability tiers:
    1. sleep_lora  — full LoRA fine-tuning (torch + transformers + peft)
    2. sleep_jsonl — write dream bundles to disk (filesystem only)
    3. sleep_remote — export dream bundles for a torch-capable peer (federation)

The consciousness loop uses these to decide what happens on DREAM entry.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class SleepCapability:
    """Runtime sleep capability for this SAGE instance."""
    sleep_lora: bool = False     # torch + transformers + peft available
    sleep_jsonl: bool = False    # can write dream bundles to disk
    sleep_remote: bool = False   # can export to a peer for remote training

    # Tracking fields (Nova's identity-drift guard)
    last_sleep_mode: Optional[str] = None        # 'lora', 'jsonl', 'remote', None
    last_consolidation_at: Optional[str] = None   # ISO timestamp
    consolidation_count: int = 0

    @classmethod
    def detect(cls, instance_dir: Optional[Path] = None) -> 'SleepCapability':
        """Detect available sleep capabilities on this machine."""
        cap = cls()

        # Tier 1: LoRA (torch + transformers + peft)
        try:
            import torch
            from transformers import AutoModelForCausalLM
            from peft import get_peft_model, LoraConfig
            cap.sleep_lora = True
        except ImportError:
            pass

        # Tier 2: JSONL (always available if we have a writable dir)
        if instance_dir and instance_dir.exists():
            try:
                dream_dir = instance_dir / "dream_bundles"
                dream_dir.mkdir(exist_ok=True)
                cap.sleep_jsonl = True
            except OSError:
                pass
        else:
            # Even without instance dir, we can write to cwd
            cap.sleep_jsonl = True

        # Tier 3: Remote (available if federation is configured)
        # For now, always True — the peer client handles availability at send time
        cap.sleep_remote = True

        return cap

    @property
    def best_mode(self) -> str:
        """Return the best available sleep mode."""
        if self.sleep_lora:
            return 'lora'
        if self.sleep_jsonl:
            return 'jsonl'
        if self.sleep_remote:
            return 'remote'
        return 'none'

    def record_consolidation(self, mode: str):
        """Record that a consolidation happened (identity-drift guard)."""
        self.last_sleep_mode = mode
        self.last_consolidation_at = datetime.now().isoformat()
        self.consolidation_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sleep_lora': self.sleep_lora,
            'sleep_jsonl': self.sleep_jsonl,
            'sleep_remote': self.sleep_remote,
            'best_mode': self.best_mode,
            'last_sleep_mode': self.last_sleep_mode,
            'last_consolidation_at': self.last_consolidation_at,
            'consolidation_count': self.consolidation_count,
        }


def write_dream_bundle(
    instance_dir: Path,
    experiences: List[Dict[str, Any]],
    machine: str = 'unknown',
    model: str = 'unknown',
    lora_hash: Optional[str] = None,
) -> Path:
    """Write a dream bundle to the instance's dream_bundles/ directory.

    A dream bundle is a portable artifact containing high-salience experiences
    ready for consolidation. On Ollama-only nodes, this is the DREAM output.
    A torch-capable peer can pick it up and run LoRA training.

    Args:
        instance_dir: Instance root directory.
        experiences: List of high-salience SNARC experiences.
        machine: Machine name (for provenance).
        model: Model identifier (for provenance).
        lora_hash: Hash of current LoRA adapter (if any).

    Returns:
        Path to the written bundle file.
    """
    bundle_dir = instance_dir / "dream_bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle_path = bundle_dir / f"dream_{timestamp}.jsonl"

    # Sort by salience descending
    sorted_exp = sorted(
        experiences,
        key=lambda x: x.get('salience', 0),
        reverse=True,
    )

    # Write bundle header + experiences
    header = {
        '_type': 'dream_bundle_header',
        'machine': machine,
        'model': model,
        'lora_hash': lora_hash,
        'experience_count': len(sorted_exp),
        'created_at': datetime.now().isoformat(),
        'format_version': 1,
    }

    written = 0
    with open(bundle_path, 'w') as f:
        f.write(json.dumps(header) + '\n')
        for exp in sorted_exp:
            record = {
                '_type': 'experience',
                'salience': exp.get('salience', 0),
                'cycle': exp.get('cycle', 0),
                'plugin': exp.get('plugin', ''),
                'timestamp': exp.get('ts', time.time()),
                'source': exp.get('source', ''),
            }
            # Extract response preview if available
            result = exp.get('result')
            if hasattr(result, 'final_state') and isinstance(result.final_state, dict):
                resp = result.final_state.get('response', '')
                if resp:
                    record['response_preview'] = resp[:500]
            # Include context/outcome for remote training
            if isinstance(exp.get('context'), dict):
                record['context'] = exp['context']
            if isinstance(exp.get('outcome'), dict):
                record['outcome'] = exp['outcome']

            f.write(json.dumps(record) + '\n')
            written += 1

    return bundle_path
