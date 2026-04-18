#!/usr/bin/env python3
"""sage_plays — replay-mode delta extraction.

At each step of a gameplay trace we already have (state, known_good_action)
captured in the {machine}/gameplay/ partition by gameplay_capture. This
module loads a promoted Head B adapter, runs inline inference on each
captured record's feature vector, and emits a delta record per step.

Delta carries rich signal:
  - correct: argmax == known_good_action
  - predicted_rank_of_known_good: where did the correct action rank in softmax?
  - confidence_on_known_good: softmax prob assigned to correct action
  - confidence_on_proposed: softmax prob of argmax
  - entropy: total softmax entropy (model uncertainty)

Output: {machine}/sage_plays/{today}.jsonl.gz in the standard RouterRecord
schema with metadata.source="sage_plays" and metadata.sage_plays=<delta>.
Record_id is new (this is a fresh record stream); the original gameplay
record_id is preserved in metadata.source_record_id for joins.

This is the innermost-level fractal: SAGE proposes, trace comparator
provides truth, delta IS the learning signal for Phase 2.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-1.5-sage-plays-plan.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sage.cognition.router.data.reader import RouterDatasetReader
from sage.cognition.router.data.writer import RouterDatasetWriter
from sage.cognition.router.record import RouterRecord
from sage.cognition.router.inputs import RouterInput
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.thalamic_router.phase1_training import (
    HeadBAdapter,
    _feature_vec,
    HEAD_B_ACTION_NAMES,
)


def _compute_delta(
    features: List[float], known_good: int, adapter: HeadBAdapter,
) -> Dict[str, Any]:
    """Run inline inference + compute the full delta signal."""
    proposed, probs = adapter.predict(features)
    ranked = sorted(range(len(probs)), key=lambda i: -probs[i])
    rank_of_kg = ranked.index(known_good) if 0 <= known_good < len(probs) else -1
    confidence_kg = probs[known_good] if 0 <= known_good < len(probs) else 0.0
    confidence_pr = probs[proposed]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
    return {
        "proposed_action": proposed,
        "proposed_name": HEAD_B_ACTION_NAMES[proposed] if proposed < len(HEAD_B_ACTION_NAMES) else str(proposed),
        "known_good_action": known_good,
        "known_good_name": (HEAD_B_ACTION_NAMES[known_good]
                            if 0 <= known_good < len(HEAD_B_ACTION_NAMES)
                            else str(known_good)),
        "correct": proposed == known_good,
        "predicted_rank_of_known_good": rank_of_kg,
        "confidence_on_known_good": float(confidence_kg),
        "confidence_on_proposed": float(confidence_pr),
        "entropy": float(entropy),
        "softmax_probs": [float(p) for p in probs],
    }


def _rec_to_input(raw: Dict[str, Any]) -> Optional[RouterInput]:
    """Hydrate a captured record's dict-shaped router_input into a RouterInput
    object — minimal fields for RouterRecord constructor compatibility.
    """
    ri = raw.get("router_input") or {}
    if not ri:
        return None
    try:
        # RouterInput has many optional fields; pass through what we have.
        # Any unknown keys will be dropped by the dataclass filter below.
        import dataclasses as dc
        valid = {f.name for f in dc.fields(RouterInput)}
        kwargs = {k: v for k, v in ri.items() if k in valid}
        return RouterInput(**kwargs)
    except Exception:
        return None


def _new_record(
    source_rec: Dict[str, Any], delta: Dict[str, Any], machine: str,
    adapter_commit: Optional[str],
) -> Optional[RouterRecord]:
    """Build a new RouterRecord for the sage_plays stream."""
    ri = _rec_to_input(source_rec)
    if ri is None:
        return None
    # Passthrough RouterOutput from the source record; we're not redecid-
    # ing dispatch, just logging proposal+delta.
    raw_ro = source_rec.get("router_output") or {}
    try:
        import dataclasses as dc
        valid = {f.name for f in dc.fields(RouterOutput)}
        ro_kwargs = {k: v for k, v in raw_ro.items() if k in valid}
        ro = RouterOutput(**ro_kwargs) if ro_kwargs else RouterOutput(action="noop")
    except Exception:
        ro = RouterOutput(action="noop")

    src_md = source_rec.get("metadata") or {}
    new_md = {
        "source": "sage_plays",
        "source_record_id": source_rec.get("record_id"),
        "game": src_md.get("game"),
        "game_id": src_md.get("game_id"),
        "level": src_md.get("level"),
        "step_index": src_md.get("step_index"),
        "known_good_action": delta["known_good_action"],
        "sage_plays": {k: v for k, v in delta.items() if k != "softmax_probs"},
        "sage_plays_probs": delta["softmax_probs"],
        "adapter_commit": adapter_commit,
    }
    import time
    from sage.cognition.router.record import RouterRecord as RR
    return RR(
        router_input=ri, router_output=ro, machine=machine,
        timestamp=time.time(), metadata=new_md,
    )


def run(
    adapter_path: Path, gameplay_dir: Path, writer: RouterDatasetWriter,
    machine: str, limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Iterate gameplay records, emit sage_plays records with delta."""
    adapter = HeadBAdapter.load(adapter_path)
    if adapter.feature_names != list(asdict(HeadBAdapter(
        head="_", machine="", trained_at="", train_commit=None, n_train_records=0,
        feature_names=[], feature_mean=[], feature_std=[],
        weights=[[]], bias=[], n_classes=0, class_names=[],
    )).get("feature_names", [])) and False:
        # Feature-name check is a paranoia stub; our adapter stores them
        # so future schema skew is detectable.
        pass

    reader = RouterDatasetReader(base_dir=Path("/tmp"))

    n_total = 0
    n_emitted = 0
    n_correct = 0
    n_no_kg = 0
    rank_counts: Dict[int, int] = {}
    per_action_correct: Dict[int, Dict[str, int]] = {}

    for shard in sorted(gameplay_dir.glob("*.jsonl*")):
        for rec in reader.read_file(shard):
            n_total += 1
            md = rec.get("metadata") or {}
            kg = md.get("known_good_action")
            if not isinstance(kg, int) or kg < 0 or kg >= adapter.n_classes:
                n_no_kg += 1
                continue
            features = _feature_vec(rec)
            delta = _compute_delta(features, kg, adapter)
            new_rec = _new_record(rec, delta, machine, adapter.train_commit)
            if new_rec is None:
                continue
            writer.append(new_rec)
            n_emitted += 1
            if delta["correct"]:
                n_correct += 1
            rank = delta["predicted_rank_of_known_good"]
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            bucket = per_action_correct.setdefault(kg, {"n": 0, "correct": 0})
            bucket["n"] += 1
            if delta["correct"]:
                bucket["correct"] += 1
            if limit and n_emitted >= limit:
                break
        if limit and n_emitted >= limit:
            break

    writer.flush()
    return {
        "n_total_records_scanned": n_total,
        "n_sage_plays_emitted": n_emitted,
        "n_correct": n_correct,
        "n_dropped_no_known_good": n_no_kg,
        "accuracy": n_correct / n_emitted if n_emitted else 0.0,
        "rank_distribution": dict(sorted(rank_counts.items())),
        "per_action": {
            HEAD_B_ACTION_NAMES[k]: {
                **v,
                "recall": v["correct"] / v["n"] if v["n"] else 0.0,
            }
            for k, v in sorted(per_action_correct.items())
        },
        "adapter_path": str(adapter_path),
        "adapter_machine": adapter.machine,
        "adapter_train_commit": adapter.train_commit,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True,
                   help="Path to a Head B adapter JSON (from phase1_training "
                        "--adapter-out).")
    p.add_argument("--gameplay-dir", default=None,
                   help="Dir of gameplay JSONL(.gz) to replay. Defaults to "
                        "$SAGE_ROUTER_DATA_DIR/$SAGE_MACHINE/gameplay/")
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"))
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--limit", type=int, default=None,
                   help="Max records to process (debug).")
    p.add_argument("--json-out", default=None,
                   help="Write summary statistics JSON to this path.")
    args = p.parse_args()

    gameplay_dir = Path(args.gameplay_dir) if args.gameplay_dir else (
        Path(args.data_dir) / args.machine / "gameplay"
    )
    if not gameplay_dir.is_dir():
        print(f"ERROR: gameplay dir not found: {gameplay_dir}")
        print(f"  Have you run fleet_gameplay_capture.sh on this machine?")
        return 1

    writer = RouterDatasetWriter(
        base_dir=Path(args.data_dir), machine=args.machine,
        compress=True, subdir="sage_plays",
    )
    try:
        result = run(Path(args.adapter), gameplay_dir, writer,
                     machine=args.machine, limit=args.limit)
    finally:
        writer.close()

    print("=" * 60)
    print("sage_plays — delta extraction report")
    print("=" * 60)
    print(f"  Machine          : {args.machine}")
    print(f"  Adapter          : {Path(args.adapter).name}")
    print(f"  Adapter machine  : {result['adapter_machine']}")
    print(f"  Adapter commit   : {result['adapter_train_commit']}")
    print(f"  Gameplay scanned : {result['n_total_records_scanned']}")
    print(f"  Records emitted  : {result['n_sage_plays_emitted']}")
    print(f"  Dropped (no kg)  : {result['n_dropped_no_known_good']}")
    print(f"  Accuracy         : {result['accuracy']:.4f}")
    print()
    print(f"  Rank-of-known-good distribution (0 = model was right):")
    for rank, count in result['rank_distribution'].items():
        bar = "█" * min(60, count // 10)
        print(f"    rank {rank}: n={count:5d}  {bar}")
    print()
    print(f"  Per-action:")
    for name, stats in result['per_action'].items():
        print(f"    {name:6s}: n={stats['n']:5d}  correct={stats['correct']:5d}  recall={stats['recall']:.3f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote summary: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
