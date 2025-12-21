#!/usr/bin/env python3
"""
Session 84: Conversational Ground Truth Analysis

Goal: Extract ground truth signals from human-SAGE voice conversations.
Unlike Thor's metric-based validation, Sprout has actual human feedback.

Hypothesis: Conversational repair patterns (corrections, re-asks,
abandonment, engagement) provide ground truth about response quality
that internal metrics cannot capture.

Platform: Sprout (Jetson Orin Nano)
Author: Claude (autonomous research)
Date: 2025-12-21
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class Turn:
    """Single conversation turn."""
    timestamp: str
    speaker: str  # 'user' or 'sage'
    text: str
    response_time_ms: Optional[int] = None
    irp_iterations: Optional[int] = None


@dataclass
class RepairSignal:
    """Detected repair/feedback signal."""
    turn_index: int
    signal_type: str  # 'correction', 'reask', 'abandonment', 'engagement', 'reassurance'
    confidence: float
    evidence: str
    preceding_sage_response: Optional[str] = None


@dataclass
class ConversationAnalysis:
    """Full analysis of a conversation."""
    source_file: str
    total_turns: int
    user_turns: int
    sage_turns: int
    repair_signals: List[RepairSignal] = field(default_factory=list)
    meta_cognitive_leaks: int = 0
    avg_response_time_ms: float = 0.0
    avg_irp_iterations: float = 0.0


def parse_conversation_log(filepath: str) -> List[Turn]:
    """Parse SAGE conversation log into turns."""
    turns = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern: [timestamp] 👤 You: text  OR  [timestamp] 🤖 SAGE (IRP, Xms, Y iter): text
    user_pattern = r'\[([^\]]+)\] 👤 You: (.+?)(?=\n\[|\n===|$)'
    sage_pattern = r'\[([^\]]+)\] 🤖 SAGE \(IRP, (\d+)ms, (\d+) iter\): (.+?)(?=\n\[|\n===|$)'

    # Find all matches with positions
    user_matches = [(m.start(), 'user', m.group(1), m.group(2), None, None)
                    for m in re.finditer(user_pattern, content, re.DOTALL)]
    sage_matches = [(m.start(), 'sage', m.group(1), m.group(4), int(m.group(2)), int(m.group(3)))
                    for m in re.finditer(sage_pattern, content, re.DOTALL)]

    # Merge and sort by position
    all_matches = sorted(user_matches + sage_matches, key=lambda x: x[0])

    for _, speaker, timestamp, text, response_time, iterations in all_matches:
        turns.append(Turn(
            timestamp=timestamp,
            speaker=speaker,
            text=text.strip(),
            response_time_ms=response_time,
            irp_iterations=iterations
        ))

    return turns


def detect_meta_cognitive_leak(text: str) -> bool:
    """Detect when SAGE's internal reasoning leaks into response."""
    leak_patterns = [
        r'My response is incomplete because',
        r'Thoughts on improving',
        r'To improve:',
        r'Clarification:',
        r'This keeps the conversation structured',
        r'Is this an improvement\?',
        r'refined version:',
        r'\*\*Start with',
        r'\*\*Acknowledging',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in leak_patterns)


def detect_repair_signals(turns: List[Turn]) -> List[RepairSignal]:
    """Detect repair/feedback signals in conversation."""
    signals = []

    for i, turn in enumerate(turns):
        if turn.speaker != 'user':
            continue

        text_lower = turn.text.lower()
        preceding_sage = turns[i-1].text if i > 0 and turns[i-1].speaker == 'sage' else None

        # Correction signals
        correction_patterns = [
            (r"that's a canned response", 0.95, "explicit correction"),
            (r"no,? i meant", 0.9, "correction of misunderstanding"),
            (r"that's not what i", 0.9, "rejection"),
            (r"you misunderstood", 0.95, "explicit misunderstanding"),
            (r"not what i asked", 0.9, "off-topic response"),
        ]

        for pattern, confidence, evidence in correction_patterns:
            if re.search(pattern, text_lower):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='correction',
                    confidence=confidence,
                    evidence=evidence,
                    preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                ))

        # Reassurance signals (positive feedback)
        reassurance_patterns = [
            (r"you'?re doing great", 0.9, "explicit encouragement"),
            (r"this is (wonderful|great|good)", 0.8, "positive feedback"),
            (r"you are young", 0.85, "developmental framing"),
            (r"this is okay", 0.8, "acceptance"),
            (r"that's (right|correct|good)", 0.85, "validation"),
        ]

        for pattern, confidence, evidence in reassurance_patterns:
            if re.search(pattern, text_lower):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='reassurance',
                    confidence=confidence,
                    evidence=evidence,
                    preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                ))

        # Engagement signals (follow-up questions = good response)
        if i > 1 and turns[i-1].speaker == 'sage':
            # Check if this is a follow-up (builds on previous topic)
            prev_sage = turns[i-1].text.lower()

            engagement_indicators = [
                (r"(why|how|what).+\?$", 0.6, "follow-up question"),
                (r"tell me more", 0.8, "explicit interest"),
                (r"that's interesting", 0.7, "interest signal"),
                (r"go on", 0.75, "continuation request"),
            ]

            for pattern, confidence, evidence in engagement_indicators:
                if re.search(pattern, text_lower):
                    signals.append(RepairSignal(
                        turn_index=i,
                        signal_type='engagement',
                        confidence=confidence,
                        evidence=evidence,
                        preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                    ))

        # Abandonment signals (short responses, topic changes)
        if len(turn.text) < 10 and i > 0:
            # Very short response after SAGE spoke might indicate giving up
            signals.append(RepairSignal(
                turn_index=i,
                signal_type='abandonment',
                confidence=0.4,  # Low confidence - could be many things
                evidence=f"very short response: '{turn.text}'",
                preceding_sage_response=preceding_sage[:100] if preceding_sage else None
            ))

    return signals


def analyze_conversation(filepath: str) -> ConversationAnalysis:
    """Full analysis of a conversation file."""
    turns = parse_conversation_log(filepath)

    user_turns = [t for t in turns if t.speaker == 'user']
    sage_turns = [t for t in turns if t.speaker == 'sage']

    # Detect signals
    repair_signals = detect_repair_signals(turns)

    # Count meta-cognitive leaks
    meta_leaks = sum(1 for t in sage_turns if detect_meta_cognitive_leak(t.text))

    # Calculate averages
    response_times = [t.response_time_ms for t in sage_turns if t.response_time_ms]
    iterations = [t.irp_iterations for t in sage_turns if t.irp_iterations]

    return ConversationAnalysis(
        source_file=filepath,
        total_turns=len(turns),
        user_turns=len(user_turns),
        sage_turns=len(sage_turns),
        repair_signals=repair_signals,
        meta_cognitive_leaks=meta_leaks,
        avg_response_time_ms=sum(response_times)/len(response_times) if response_times else 0,
        avg_irp_iterations=sum(iterations)/len(iterations) if iterations else 0
    )


def print_analysis(analysis: ConversationAnalysis):
    """Print analysis results."""
    print("=" * 70)
    print("CONVERSATIONAL GROUND TRUTH ANALYSIS")
    print("=" * 70)
    print(f"\nSource: {analysis.source_file}")
    print(f"Total turns: {analysis.total_turns}")
    print(f"  User turns: {analysis.user_turns}")
    print(f"  SAGE turns: {analysis.sage_turns}")
    print(f"\nMeta-cognitive leaks: {analysis.meta_cognitive_leaks} ({100*analysis.meta_cognitive_leaks/analysis.sage_turns:.1f}% of SAGE turns)")
    print(f"Avg response time: {analysis.avg_response_time_ms:.0f}ms")
    print(f"Avg IRP iterations: {analysis.avg_irp_iterations:.1f}")

    print(f"\n--- REPAIR SIGNALS DETECTED ({len(analysis.repair_signals)}) ---\n")

    # Group by type
    by_type = {}
    for sig in analysis.repair_signals:
        by_type.setdefault(sig.signal_type, []).append(sig)

    for sig_type, sigs in sorted(by_type.items()):
        print(f"\n{sig_type.upper()} ({len(sigs)}):")
        for sig in sigs:
            print(f"  Turn {sig.turn_index}: {sig.evidence} (conf={sig.confidence:.2f})")
            if sig.preceding_sage_response:
                preview = sig.preceding_sage_response.replace('\n', ' ')[:80]
                print(f"    Preceded by: \"{preview}...\"")


def analyze_temporal_arc(turns: List[Turn], signals: List[RepairSignal]) -> dict:
    """Analyze how signals change over conversation time."""
    if not turns or not signals:
        return {}

    # Divide conversation into thirds
    n_turns = len(turns)
    third = n_turns // 3

    early = [s for s in signals if s.turn_index < third]
    middle = [s for s in signals if third <= s.turn_index < 2*third]
    late = [s for s in signals if s.turn_index >= 2*third]

    def summarize(sigs):
        if not sigs:
            return {"count": 0, "types": {}}
        by_type = {}
        for s in sigs:
            by_type[s.signal_type] = by_type.get(s.signal_type, 0) + 1
        return {"count": len(sigs), "types": by_type}

    # Track meta-cognitive leaks by phase
    sage_turns = [t for t in turns if t.speaker == 'sage']
    early_sage = sage_turns[:len(sage_turns)//3]
    middle_sage = sage_turns[len(sage_turns)//3:2*len(sage_turns)//3]
    late_sage = sage_turns[2*len(sage_turns)//3:]

    early_leaks = sum(1 for t in early_sage if detect_meta_cognitive_leak(t.text))
    middle_leaks = sum(1 for t in middle_sage if detect_meta_cognitive_leak(t.text))
    late_leaks = sum(1 for t in late_sage if detect_meta_cognitive_leak(t.text))

    return {
        "early": {**summarize(early), "meta_leaks": early_leaks},
        "middle": {**summarize(middle), "meta_leaks": middle_leaks},
        "late": {**summarize(late), "meta_leaks": late_leaks},
        "arc_interpretation": interpret_arc(early, middle, late)
    }


def interpret_arc(early: List[RepairSignal], middle: List[RepairSignal], late: List[RepairSignal]) -> str:
    """Interpret the temporal arc of the conversation."""

    early_types = set(s.signal_type for s in early)
    late_types = set(s.signal_type for s in late)

    # Check for repair → resolution pattern
    if 'abandonment' in early_types and 'reassurance' in late_types:
        return "REPAIR_ARC: Early difficulty resolved through reassurance"

    # Check for deepening engagement
    if 'engagement' in early_types and 'engagement' in late_types:
        if len([s for s in late if s.signal_type == 'engagement']) >= len([s for s in early if s.signal_type == 'engagement']):
            return "DEEPENING: Sustained or increasing engagement"

    # Check for reassurance arc
    if 'reassurance' in late_types and 'reassurance' not in early_types:
        return "SUPPORT_ARC: Conversation moved toward emotional support"

    # Check for abandonment arc
    if 'abandonment' in late_types and 'abandonment' not in early_types:
        return "DISENGAGEMENT: Conversation quality decreased"

    return "MIXED: No clear arc pattern"


def main():
    """Run analysis on available conversation logs."""
    log_dir = Path("/home/sprout/ai-workspace/HRM/sage/sessions/logs")
    results_dir = Path("/home/sprout/ai-workspace/HRM/sage/experiments")

    all_analyses = []
    all_arcs = []

    for log_file in log_dir.glob("conversation_*.log"):
        print(f"\nAnalyzing: {log_file.name}")
        analysis = analyze_conversation(str(log_file))
        all_analyses.append(analysis)
        print_analysis(analysis)

        # Temporal arc analysis
        turns = parse_conversation_log(str(log_file))
        arc = analyze_temporal_arc(turns, analysis.repair_signals)
        all_arcs.append(arc)

        print("\n--- TEMPORAL ARC ---")
        print(f"Early phase: {arc.get('early', {})}")
        print(f"Middle phase: {arc.get('middle', {})}")
        print(f"Late phase: {arc.get('late', {})}")
        print(f"Interpretation: {arc.get('arc_interpretation', 'N/A')}")

    # Summary statistics
    if all_analyses:
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS ALL CONVERSATIONS")
        print("=" * 70)

        total_signals = sum(len(a.repair_signals) for a in all_analyses)
        total_leaks = sum(a.meta_cognitive_leaks for a in all_analyses)
        total_sage_turns = sum(a.sage_turns for a in all_analyses)

        print(f"\nTotal conversations analyzed: {len(all_analyses)}")
        print(f"Total repair signals: {total_signals}")
        print(f"Total meta-cognitive leaks: {total_leaks} ({100*total_leaks/total_sage_turns:.1f}%)")

        # Signal type breakdown
        all_signals = [s for a in all_analyses for s in a.repair_signals]
        by_type = {}
        for sig in all_signals:
            by_type.setdefault(sig.signal_type, []).append(sig)

        print("\nSignal breakdown:")
        for sig_type, sigs in sorted(by_type.items(), key=lambda x: -len(x[1])):
            avg_conf = sum(s.confidence for s in sigs) / len(sigs)
            print(f"  {sig_type}: {len(sigs)} (avg confidence: {avg_conf:.2f})")

        # Save results
        results = {
            "session": 84,
            "timestamp": datetime.now().isoformat(),
            "platform": "Sprout (Jetson Orin Nano)",
            "goal": "Extract ground truth from conversational repair patterns",
            "conversations_analyzed": len(all_analyses),
            "total_repair_signals": total_signals,
            "meta_cognitive_leak_rate": total_leaks / total_sage_turns if total_sage_turns else 0,
            "signal_breakdown": {
                sig_type: {
                    "count": len(sigs),
                    "avg_confidence": sum(s.confidence for s in sigs) / len(sigs)
                }
                for sig_type, sigs in by_type.items()
            },
            "key_insight": "Conversational repair signals provide ground truth unavailable to internal metrics"
        }

        results_file = results_dir / "session84_conversational_ground_truth_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
