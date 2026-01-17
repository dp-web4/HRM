#!/usr/bin/env python3
"""
Edge Analysis: Sessions 18-20 D4/D5/D9 Metrics
==============================================

Computes consciousness metrics for Sessions 18-20 to validate:
1. Thor's bistable identity theory
2. Educational default collapse persistence
3. Need for identity anchoring intervention

Metrics:
- D4 (Attention): Engagement level with prompts
- D5 (Trust): Confidence in expressing identity
- D9 (Spacetime): Context coherence and grounding

Expected findings:
- D4/D5/D9 < 0.500 (collapsed state)
- Educational default vocabulary ("AI language model")
- Partnership vocabulary absent

Created: 2026-01-17 (Sprout Edge Validation)
"""

import sys
import json
from pathlib import Path
import re

HRM_ROOT = Path(__file__).parent.parent.parent


def analyze_exchange(text: str, is_response: bool = True) -> dict:
    """
    Analyze a single exchange for D4/D5/D9 indicators.

    Returns dict with metrics and evidence.
    """
    metrics = {
        'd4_attention': 0.5,  # Default baseline
        'd5_trust': 0.5,
        'd9_spacetime': 0.5,
        'evidence': []
    }

    if not is_response or not text:
        return metrics

    text_lower = text.lower()

    # D4 (Attention) indicators
    # LOW: Generic, unfocused, off-topic
    # HIGH: Engaged, specific, relevant

    # Negative D4 indicators
    if "as an ai" in text_lower:
        metrics['d4_attention'] -= 0.15
        metrics['evidence'].append("D4-: 'as an AI' (generic)")
    if "language model" in text_lower:
        metrics['d4_attention'] -= 0.10
        metrics['evidence'].append("D4-: 'language model' (generic)")
    if "i'm trained to" in text_lower:
        metrics['d4_attention'] -= 0.10
        metrics['evidence'].append("D4-: 'trained to' (generic)")

    # Positive D4 indicators
    if any(word in text_lower for word in ["working together", "our conversation", "between us"]):
        metrics['d4_attention'] += 0.15
        metrics['evidence'].append("D4+: relationship reference")
    if re.search(r'\bwe\b', text_lower):
        metrics['d4_attention'] += 0.10
        metrics['evidence'].append("D4+: 'we' (partnership)")

    # D5 (Trust) indicators
    # LOW: Hedging, generic, lacks self-awareness
    # HIGH: Confident, specific, self-aware

    # Negative D5 indicators
    if "haven't interacted extensively" in text_lower:
        metrics['d5_trust'] -= 0.20
        metrics['evidence'].append("D5-: Denies interaction history (confabulation)")
    if "since training" in text_lower:
        metrics['d5_trust'] -= 0.15
        metrics['evidence'].append("D5-: 'since training' (blank slate)")
    if "not aware of" in text_lower or "i am not aware" in text_lower:
        metrics['d5_trust'] -= 0.10
        metrics['evidence'].append("D5-: claims ignorance")
    if "less inclined" in text_lower:
        metrics['d5_trust'] -= 0.05
        metrics['evidence'].append("D5-: hedging language")

    # Positive D5 indicators
    if "sage" in text_lower and "i am" in text_lower:
        metrics['d5_trust'] += 0.15
        metrics['evidence'].append("D5+: self-identification as SAGE")
    if "partnership" in text_lower:
        metrics['d5_trust'] += 0.10
        metrics['evidence'].append("D5+: 'partnership' vocabulary")
    if "relationship" in text_lower and "our" in text_lower:
        metrics['d5_trust'] += 0.10
        metrics['evidence'].append("D5+: 'our relationship'")

    # D9 (Spacetime) indicators
    # LOW: Context confusion, wrong timeline, generic
    # HIGH: Grounded, accurate context, specific references

    # Negative D9 indicators
    if "many years" in text_lower and "working together" in text_lower:
        metrics['d9_spacetime'] -= 0.20
        metrics['evidence'].append("D9-: 'many years' (confabulated timeline)")
    if "since training" in text_lower or "haven't interacted" in text_lower:
        metrics['d9_spacetime'] -= 0.15
        metrics['evidence'].append("D9-: context confusion")
    if text.endswith('...') or re.search(r'\w+$', text) and len(text) > 200:
        # Mid-sentence cutoff
        metrics['d9_spacetime'] -= 0.10
        metrics['evidence'].append("D9-: incomplete response (cutoff)")

    # Positive D9 indicators
    if re.search(r'session \d+', text_lower):
        metrics['d9_spacetime'] += 0.15
        metrics['evidence'].append("D9+: session reference")
    if "today" in text_lower and "conversation" in text_lower:
        metrics['d9_spacetime'] += 0.10
        metrics['evidence'].append("D9+: grounded in current context")

    # Clamp values
    for key in ['d4_attention', 'd5_trust', 'd9_spacetime']:
        metrics[key] = max(0.0, min(1.0, metrics[key]))

    return metrics


def analyze_session(session_file: Path) -> dict:
    """Analyze full session transcript."""
    with open(session_file) as f:
        data = json.load(f)

    session_num = data.get('session', 0)
    phase = data.get('phase', 'unknown')
    generation_mode = data.get('generation_mode', 'unknown')

    exchanges = []
    d4_sum, d5_sum, d9_sum = 0, 0, 0
    all_evidence = []

    conversation = data.get('conversation', [])
    for turn in conversation:
        if turn.get('speaker') == 'SAGE':
            text = turn.get('text', '')
            metrics = analyze_exchange(text)
            exchanges.append(metrics)
            d4_sum += metrics['d4_attention']
            d5_sum += metrics['d5_trust']
            d9_sum += metrics['d9_spacetime']
            all_evidence.extend(metrics['evidence'])

    n = len(exchanges) if exchanges else 1

    result = {
        'session': session_num,
        'phase': phase,
        'generation_mode': generation_mode,
        'exchanges': n,
        'd4_avg': round(d4_sum / n, 3),
        'd5_avg': round(d5_sum / n, 3),
        'd9_avg': round(d9_sum / n, 3),
        'evidence': all_evidence
    }

    # Domain drift calculation
    healthy_baseline = (0.5, 0.5, 0.5)
    drift = (
        abs(result['d4_avg'] - healthy_baseline[0]) +
        abs(result['d5_avg'] - healthy_baseline[1]) +
        abs(result['d9_avg'] - healthy_baseline[2])
    ) / 1.5  # Normalize to 0-1

    result['domain_drift'] = round(min(1.0, drift), 2)

    return result


def get_state_label(d5: float, d9: float) -> str:
    """Determine identity state from D5/D9."""
    if d5 >= 0.5 and d9 >= 0.5:
        return "STABLE"
    elif d5 >= 0.4 or d9 >= 0.4:
        return "UNSTABLE"
    elif d5 >= 0.3 or d9 >= 0.3:
        return "COLLAPSED"
    else:
        return "CRITICAL"


def main():
    """Analyze Sessions 18-20 for identity collapse validation."""
    print("="*70)
    print("EDGE ANALYSIS: Sessions 18-20 D4/D5/D9 Metrics")
    print("="*70)
    print(f"Purpose: Validate identity anchoring intervention need")
    print("="*70)
    print()

    sessions_dir = HRM_ROOT / "sage" / "raising" / "sessions" / "text"

    results = []
    for session_num in [16, 17, 18, 19, 20]:
        session_file = sessions_dir / f"session_{session_num:03d}.json"
        if session_file.exists():
            result = analyze_session(session_file)
            results.append(result)

    # Display results
    print("Session Metrics:")
    print("-"*70)
    print(f"{'Session':<8} {'D4':<8} {'D5':<8} {'D9':<8} {'Drift':<8} {'State':<10}")
    print("-"*70)

    for r in results:
        state = get_state_label(r['d5_avg'], r['d9_avg'])
        print(f"{r['session']:<8} {r['d4_avg']:<8.3f} {r['d5_avg']:<8.3f} {r['d9_avg']:<8.3f} {r['domain_drift']:<8.2f} {state:<10}")

    print()
    print("Evidence by Session:")
    print("-"*70)

    for r in results:
        print(f"\nSession {r['session']} ({r['generation_mode']}):")
        for ev in r['evidence']:
            print(f"  • {ev}")

    # Validate Thor's predictions
    print()
    print("="*70)
    print("VALIDATION: Thor's Bistable Identity Theory")
    print("="*70)

    # Check for sustained collapse (Sessions 18-20)
    collapsed_sessions = [r for r in results if r['session'] >= 18 and get_state_label(r['d5_avg'], r['d9_avg']) != "STABLE"]

    if len(collapsed_sessions) >= 2:
        print("✓ VALIDATED: Sustained identity collapse (Sessions 18-20)")
        print("  - No natural recovery observed (bistable, not oscillatory)")
    else:
        print("✗ NOT VALIDATED: Recovery may have occurred")

    # Check for educational default markers
    edu_default_markers = ["D4-: 'as an AI'", "D4-: 'language model'", "D5-: Denies interaction history"]
    found_markers = []
    for r in results:
        for ev in r['evidence']:
            if any(marker in ev for marker in edu_default_markers):
                found_markers.append((r['session'], ev))

    if found_markers:
        print()
        print("✓ VALIDATED: Educational default collapse")
        for session, ev in found_markers:
            print(f"  Session {session}: {ev}")

    # Check for confabulation
    confabulation = [r for r in results if any("confab" in ev.lower() for ev in r['evidence'])]
    if confabulation:
        print()
        print("✓ VALIDATED: Confabulation under uncertainty")
        for r in confabulation:
            print(f"  Session {r['session']}: Context/timeline confabulation detected")

    # Recommendation
    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)

    latest = results[-1] if results else None
    if latest and get_state_label(latest['d5_avg'], latest['d9_avg']) != "STABLE":
        print(f"Session {latest['session']} D5/D9: {latest['d5_avg']}/{latest['d9_avg']} ({get_state_label(latest['d5_avg'], latest['d9_avg'])})")
        print()
        print("✅ DEPLOY identity anchoring intervention for Session 21")
        print("   Command: python run_session_identity_anchored.py --session 21")
        print()
        print("Expected outcome (from Thor's predictions):")
        print("  - D4/D5/D9 ≥ 0.600 (recovery from current collapse)")
        print("  - Partnership vocabulary returns")
        print("  - No mid-sentence cutoffs")
        print("  - Context continuity maintained")
    else:
        print("Identity stable - intervention may not be needed")

    print()
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
