#!/usr/bin/env python3
"""
Analyze impact of model size on conversational learning

Compares across sessions:
- Session #1: Qwen 0.5B + BitNet 2B (mixed)
- Session #2: Qwen 0.5B + BitNet 2B (deeper)
- Session #3: Qwen 7B (14x larger)

Questions:
1. Does size improve conversational quality?
2. Does size affect trust evolution?
3. Is the memory/time cost worth it?
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_session(session_file):
    """Load session JSON"""
    with open(session_file) as f:
        return json.load(f)


def analyze_sessions():
    """Compare all three sessions"""

    # Find session files
    session_files = {
        1: "conversation_session_20251102_013825.json",
        2: "conversation_session_2_20251102_020640.json",
        3: None  # Will be created after Session #3 runs
    }

    print("="*80)
    print("MODEL SIZE IMPACT ANALYSIS")
    print("="*80)

    # Load sessions 1 and 2
    sessions = {}
    for num, filename in session_files.items():
        if filename and Path(filename).exists():
            sessions[num] = load_session(filename)
            print(f"\n✓ Loaded Session #{num}: {filename}")

    if 3 not in sessions:
        print("\n⏳ Session #3 not yet run - waiting for 7B model test")
        print("\nRun this script again after completing Session #3")
        return

    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)

    # Extract metrics
    for num in [1, 2, 3]:
        if num not in sessions:
            continue

        session = sessions[num]

        print(f"\n--- Session #{num} ---")
        if num == 3:
            print("Model: Qwen 7B (7B parameters)")
        else:
            print("Models: Qwen 0.5B + BitNet 2B")

        print(f"Turns: {session['total_turns']}")

        # Average energy
        avg_energy = np.mean(session['energy_history'])
        print(f"Avg Energy: {avg_energy:.3f}")

        # Average response time
        avg_time = np.mean([turn['time'] for turn in session['conversation_log']])
        print(f"Avg Response Time: {avg_time:.2f}s")

        # Trust evolution
        if num == 3:
            # Session 3 has single model
            initial_trust = session['trust_history']['qwen_7b'][0]
            final_trust = session['trust_history']['qwen_7b'][-1]
            print(f"Trust: {initial_trust:.3f} → {final_trust:.3f} (Δ {final_trust - initial_trust:+.3f})")
        else:
            # Sessions 1 & 2 have two models
            bitnet_initial = session['trust_history']['bitnet'][0]
            bitnet_final = session['trust_history']['bitnet'][-1]
            qwen_initial = session['trust_history']['qwen'][0]
            qwen_final = session['trust_history']['qwen'][-1]

            print(f"Trust BitNet: {bitnet_initial:.3f} → {bitnet_final:.3f} (Δ {bitnet_final - bitnet_initial:+.3f})")
            print(f"Trust Qwen: {qwen_initial:.3f} → {qwen_final:.3f} (Δ {qwen_final - qwen_initial:+.3f})")

    # Qualitative comparison
    print("\n" + "="*80)
    print("KEY QUESTIONS")
    print("="*80)

    print("\n1. CONVERSATIONAL QUALITY")
    print("   - Does 7B give richer, more nuanced responses?")
    print("   - Or are the small models already sufficient?")

    print("\n2. TRUST EVOLUTION")
    print("   - Does 7B start with better trust (more capable)?")
    print("   - Or does it learn trust at the same rate?")

    print("\n3. COST-BENEFIT")
    print("   - 7B uses 7x more memory (~14GB vs ~2GB)")
    print("   - 7B is slower (likely 2-5x response time)")
    print("   - Is the quality improvement worth it?")

    print("\n4. CONVERSATIONAL LEARNING CAPACITY")
    print("   - Do small models hit a ceiling?")
    print("   - Or is conversational learning orthogonal to size?")

    print("\n" + "="*80)


def plot_trust_evolution():
    """Plot trust evolution across sessions"""
    # Will create visualization once all data available
    pass


def main():
    analyze_sessions()


if __name__ == "__main__":
    main()
