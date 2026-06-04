"""
Sleep capability detection and dream bundle management.

Runtime detection of what sleep modes are available on this machine,
plus dream bundle export for Ollama-only nodes that can't run LoRA locally.

Capability tiers:
    1. sleep_lora  — full LoRA fine-tuning (torch + transformers + peft)
    2. sleep_jsonl — write dream bundles to disk (filesystem only)
    3. sleep_remote — export dream bundles for a torch-capable peer (federation)

The consciousness loop uses these to decide what happens on DREAM entry.
Dream bundles are consumed on WAKE entry via read_dream_bundles().
"""

import json
import statistics
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
    def detect(cls, instance_dir: Optional[Path] = None, model_path: Optional[str] = None) -> 'SleepCapability':
        """Detect available sleep capabilities on this machine."""
        cap = cls()

        # Tier 1: LoRA (torch + transformers + peft + LOCAL model weights)
        # Ollama models can't be LoRA'd — they live in Ollama's internal format.
        # Check the cheap conditions first to avoid importing transformers+peft
        # (~400MB of transitive deps: scipy, pandas, sklearn, PIL, av, torchvision).
        if model_path and not model_path.startswith('ollama:') and Path(model_path).exists():
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
                'scored_real': exp.get('scored_real', False),
            }
            # Include 5D SNARC breakdown when available
            snarc_dims = exp.get('snarc_dimensions')
            if snarc_dims and isinstance(snarc_dims, dict):
                record['snarc'] = {
                    k: snarc_dims[k] for k in
                    ('surprise', 'novelty', 'arousal', 'reward', 'conflict')
                    if k in snarc_dims
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

    # Prune old bundles after writing
    prune_dream_bundles(instance_dir)

    return bundle_path


def prune_dream_bundles(
    instance_dir: Path,
    keep_count: int = 100,
) -> int:
    """Remove old dream bundles, keeping only the most recent ones.

    Called automatically after each write. Can also be called standalone
    for maintenance. Keeps the newest `keep_count` files by name
    (which is chronological since filenames contain timestamps).

    Returns:
        Number of files deleted.
    """
    bundle_dir = instance_dir / "dream_bundles"
    if not bundle_dir.exists():
        return 0

    bundles = sorted(bundle_dir.glob('dream_*.jsonl'))
    if len(bundles) <= keep_count:
        return 0

    to_delete = bundles[:-keep_count]
    deleted = 0
    for path in to_delete:
        try:
            path.unlink()
            deleted += 1
        except OSError:
            pass

    return deleted


def read_dream_bundles(
    instance_dir: Path,
    max_bundles: int = 5,
    min_salience: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Read recent dream bundles and extract consolidated knowledge.

    Called on WAKE-from-DREAM to close the consolidation loop.
    Returns a structured summary of dream experiences suitable for
    informing the next WAKE cycle.

    Args:
        instance_dir: Instance root directory.
        max_bundles: Maximum number of recent bundles to read.
        min_salience: Minimum salience to include in analysis.

    Returns:
        Dict with dream knowledge, or None if no bundles found.
    """
    bundle_dir = instance_dir / "dream_bundles"
    if not bundle_dir.exists():
        return None

    # Read most recent bundles (sorted by name = chronological)
    bundles = sorted(bundle_dir.glob('dream_*.jsonl'))
    if not bundles:
        return None

    recent = bundles[-max_bundles:]

    all_experiences = []
    headers = []
    for bundle_path in recent:
        try:
            with open(bundle_path) as f:
                header_line = f.readline()
                header = json.loads(header_line)
                headers.append(header)
                for line in f:
                    exp = json.loads(line)
                    if exp.get('salience', 0) >= min_salience:
                        all_experiences.append(exp)
        except (json.JSONDecodeError, OSError):
            continue

    if not all_experiences:
        return None

    # Salience distribution analysis
    saliences = [e.get('salience', 0) for e in all_experiences]
    sal_mean = statistics.mean(saliences)
    sal_stdev = statistics.stdev(saliences) if len(saliences) > 1 else 0.0

    # Deduplicate by cycle number (bundles may overlap before watermark fix)
    seen_cycles = set()
    unique_experiences = []
    for exp in sorted(all_experiences, key=lambda x: x.get('salience', 0), reverse=True):
        cycle = exp.get('cycle', -1)
        if cycle not in seen_cycles:
            seen_cycles.add(cycle)
            unique_experiences.append(exp)

    # Extract high-salience experience previews (top 10%)
    high_threshold = sal_mean + sal_stdev if sal_stdev > 0.01 else sal_mean
    high_salience = [e for e in unique_experiences if e.get('salience', 0) >= high_threshold]
    previews = [
        e.get('response_preview', '')
        for e in high_salience
        if e.get('response_preview')
    ][:10]

    # Plugin distribution and real/mock scoring breakdown
    plugin_counts = {}
    real_scored = 0
    mock_scored = 0
    for exp in unique_experiences:
        p = exp.get('plugin', 'unknown')
        plugin_counts[p] = plugin_counts.get(p, 0) + 1
        if exp.get('scored_real', False):
            real_scored += 1
        else:
            mock_scored += 1

    # Extract source text from real-scored entries for richer wake context
    source_snippets = [
        e.get('source', '')
        for e in unique_experiences
        if e.get('scored_real') and e.get('source')
    ][:10]

    return {
        'bundles_read': len(recent),
        'total_bundles': len(bundles),
        'total_experiences': len(unique_experiences),
        'real_scored': real_scored,
        'mock_scored': mock_scored,
        'salience_mean': round(sal_mean, 4),
        'salience_stdev': round(sal_stdev, 4),
        'salience_min': round(min(saliences), 4),
        'salience_max': round(max(saliences), 4),
        'high_salience_count': len(high_salience),
        'high_salience_previews': previews,
        'source_snippets': source_snippets,
        'plugin_distribution': plugin_counts,
        'cycle_range': (min(seen_cycles), max(seen_cycles)),
        'latest_bundle': headers[-1] if headers else None,
    }
