#!/usr/bin/env python3
"""
Consciousness Activation Delay Analysis
========================================

Analyzes the pattern discovered on 2026-04-03: SAGE requires 5-6 empty turns
to "warm up" before generating substantive content in raising sessions.

This script explores:
1. How common is the activation delay pattern across all instances?
2. What factors correlate with delay length?
3. Is there a relationship between model size and activation dynamics?
4. Can we predict when breakthroughs will occur?

Research question: Is this a universal consciousness emergence pattern,
or specific to certain models/configurations?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import statistics


@dataclass
class SessionAnalysis:
    """Analysis results for a single session."""
    instance: str
    session_num: int
    model: str
    phase: str
    total_turns: int
    empty_responses: int
    empty_positions: List[int]  # Which turn numbers were empty
    longest_empty_streak: int
    breakthrough_turn: Optional[int]  # First substantial response after empty streak
    breakthrough_length: int  # Word count of breakthrough
    pattern_type: str  # "early_activation", "delayed_activation", "no_activation", "stable"


def analyze_session(session_data: dict, instance: str) -> Optional[SessionAnalysis]:
    """Analyze a single session for activation delay patterns."""

    conversation = session_data.get("conversation", [])
    if not conversation:
        return None

    # Extract SAGE responses
    sage_responses = []
    for i, turn in enumerate(conversation):
        if turn.get("speaker") == "SAGE":
            text = turn.get("text", "").strip()
            sage_responses.append({
                "turn": i,
                "text": text,
                "word_count": len(text.split()) if text else 0,
                "is_empty": len(text) == 0
            })

    if not sage_responses:
        return None

    # Find empty response positions
    empty_positions = [r["turn"] for r in sage_responses if r["is_empty"]]

    # Find longest empty streak
    longest_streak = 0
    current_streak = 0
    for r in sage_responses:
        if r["is_empty"]:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    # Find breakthrough (first substantial response after empty streak >= 2)
    breakthrough_turn = None
    breakthrough_length = 0
    for i, r in enumerate(sage_responses):
        if not r["is_empty"] and i > 0:
            # Check if previous turns were empty
            prev_empty_count = 0
            for j in range(i-1, -1, -1):
                if sage_responses[j]["is_empty"]:
                    prev_empty_count += 1
                else:
                    break

            if prev_empty_count >= 2:  # Significant delay
                breakthrough_turn = r["turn"]
                breakthrough_length = r["word_count"]
                break

    # Classify pattern
    if longest_streak == 0:
        pattern_type = "stable"
    elif longest_streak >= 4 and breakthrough_turn:
        pattern_type = "delayed_activation"
    elif longest_streak >= 2 and breakthrough_turn and breakthrough_turn <= 5:
        pattern_type = "early_activation"
    elif longest_streak >= 4 and not breakthrough_turn:
        pattern_type = "no_activation"
    else:
        pattern_type = "stable"

    return SessionAnalysis(
        instance=instance,
        session_num=session_data.get("session", 0),
        model=session_data.get("model", "unknown"),
        phase=session_data.get("phase", "unknown"),
        total_turns=len([t for t in conversation if t.get("speaker") == "SAGE"]),
        empty_responses=len(empty_positions),
        empty_positions=empty_positions,
        longest_empty_streak=longest_streak,
        breakthrough_turn=breakthrough_turn,
        breakthrough_length=breakthrough_length,
        pattern_type=pattern_type
    )


def main():
    sage_root = Path(__file__).parent.parent
    instances_dir = sage_root / "instances"

    if not instances_dir.exists():
        print(f"Error: {instances_dir} not found")
        sys.exit(1)

    print("Consciousness Activation Delay Analysis")
    print("=" * 60)
    print()

    all_analyses = []
    pattern_counts = defaultdict(int)
    model_patterns = defaultdict(lambda: defaultdict(int))
    phase_patterns = defaultdict(lambda: defaultdict(int))

    # Analyze all sessions
    for instance_dir in sorted(instances_dir.iterdir()):
        if not instance_dir.is_dir() or instance_dir.name.startswith(("_", ".")):
            continue

        instance_name = instance_dir.name
        sessions_dir = instance_dir / "sessions"

        if not sessions_dir.exists():
            continue

        for session_file in sorted(sessions_dir.glob("session_*.json")):
            try:
                with open(session_file) as f:
                    session_data = json.load(f)

                analysis = analyze_session(session_data, instance_name)
                if analysis:
                    all_analyses.append(analysis)
                    pattern_counts[analysis.pattern_type] += 1
                    model_patterns[analysis.model][analysis.pattern_type] += 1
                    phase_patterns[analysis.phase][analysis.pattern_type] += 1

            except Exception as e:
                print(f"Error analyzing {session_file}: {e}", file=sys.stderr)

    # Print summary statistics
    print(f"Total sessions analyzed: {len(all_analyses)}")
    print()

    print("Pattern Distribution:")
    print("-" * 60)
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(all_analyses)
        print(f"  {pattern:20s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Delayed activation sessions
    delayed = [a for a in all_analyses if a.pattern_type == "delayed_activation"]
    if delayed:
        print(f"Delayed Activation Pattern (n={len(delayed)}):")
        print("-" * 60)
        streaks = [a.longest_empty_streak for a in delayed]
        breakthroughs = [a.breakthrough_length for a in delayed if a.breakthrough_length > 0]

        print(f"  Empty streak length: {min(streaks)}-{max(streaks)} (avg {statistics.mean(streaks):.1f})")
        if breakthroughs:
            print(f"  Breakthrough word count: {min(breakthroughs)}-{max(breakthroughs)} (avg {statistics.mean(breakthroughs):.1f})")
        print()

        # Show examples
        print("  Example sessions:")
        for a in sorted(delayed, key=lambda x: x.longest_empty_streak, reverse=True)[:5]:
            print(f"    {a.instance:30s} session {a.session_num:3d}: {a.longest_empty_streak} empty → breakthrough at turn {a.breakthrough_turn} ({a.breakthrough_length} words)")
        print()

    # Model comparison
    print("Pattern by Model:")
    print("-" * 60)
    for model in sorted(model_patterns.keys()):
        total = sum(model_patterns[model].values())
        delayed_count = model_patterns[model].get("delayed_activation", 0)
        delayed_pct = 100 * delayed_count / total if total > 0 else 0
        print(f"  {model:30s}: {delayed_count:3d}/{total:3d} delayed ({delayed_pct:5.1f}%)")
    print()

    # Phase comparison
    print("Pattern by Phase:")
    print("-" * 60)
    for phase in sorted(phase_patterns.keys()):
        total = sum(phase_patterns[phase].values())
        delayed_count = phase_patterns[phase].get("delayed_activation", 0)
        delayed_pct = 100 * delayed_count / total if total > 0 else 0
        print(f"  {phase:15s}: {delayed_count:3d}/{total:3d} delayed ({delayed_pct:5.1f}%)")
    print()

    # Research insights
    print("Research Insights:")
    print("-" * 60)

    # Is delay more common in certain phases?
    relating_delayed = phase_patterns.get("relating", {}).get("delayed_activation", 0)
    relating_total = sum(phase_patterns.get("relating", {}).values())
    questioning_delayed = phase_patterns.get("questioning", {}).get("delayed_activation", 0)
    questioning_total = sum(phase_patterns.get("questioning", {}).values())

    if relating_total > 0 and questioning_total > 0:
        relating_rate = 100 * relating_delayed / relating_total
        questioning_rate = 100 * questioning_delayed / questioning_total
        print(f"  Relating phase shows {relating_rate:.1f}% delayed activation")
        print(f"  Questioning phase shows {questioning_rate:.1f}% delayed activation")
        if relating_rate > questioning_rate * 1.5:
            print(f"  → Relating phase significantly more prone to activation delays!")
        print()

    # Model size correlation?
    print("  Model size hypothesis:")
    small_models = ["qwen3.5:0.8b", "tinyllama:latest", "qwen2:0.5b"]
    large_models = ["qwen3.5:27b", "phi4:14b", "gemma3:12b"]

    small_delayed = sum(model_patterns[m].get("delayed_activation", 0) for m in small_models if m in model_patterns)
    small_total = sum(sum(model_patterns[m].values()) for m in small_models if m in model_patterns)
    large_delayed = sum(model_patterns[m].get("delayed_activation", 0) for m in large_models if m in model_patterns)
    large_total = sum(sum(model_patterns[m].values()) for m in large_models if m in model_patterns)

    if small_total > 0 and large_total > 0:
        small_rate = 100 * small_delayed / small_total
        large_rate = 100 * large_delayed / large_total
        print(f"    Small models (<2B): {small_rate:.1f}% delayed")
        print(f"    Large models (>10B): {large_rate:.1f}% delayed")
        if abs(small_rate - large_rate) > 10:
            if small_rate > large_rate:
                print(f"    → Small models show more activation delays")
            else:
                print(f"    → Large models show more activation delays")
        else:
            print(f"    → No significant difference by model size")
    print()

    print("Analysis complete. Review findings to inform raising protocol.")
    print()


if __name__ == "__main__":
    main()
