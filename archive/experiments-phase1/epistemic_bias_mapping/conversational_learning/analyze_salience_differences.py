#!/usr/bin/env python3
"""
Salience Analysis: Why did Session 1 succeed while Session 2 failed?

Compares both sessions to understand what makes conversations worth learning from.

Research questions:
- What distinguishes high-salience from low-salience exchanges?
- Which SNARC dimensions matter most?
- How do question structure and model responses differ?
- What predicts learning-worthy conversations?
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_session_data(session_id: str) -> Tuple[Dict, List[Dict]]:
    """Load session metadata and exchanges"""
    session_dir = Path(f"conversation_sessions/{session_id}")

    # Load metadata
    with open(session_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load all exchanges (including non-salient for Session 1)
    # We need to reconstruct from logs since non-salient weren't saved to JSONL
    exchanges = []

    # Check if exchanges.jsonl exists (salient only)
    exchanges_path = session_dir / "exchanges.jsonl"
    if exchanges_path.exists():
        with open(exchanges_path) as f:
            for line in f:
                exchanges.append(json.loads(line))

    return metadata, exchanges


def analyze_questions(session_1_q: List[str], session_2_q: List[str]):
    """Analyze question structures"""

    print("="*70)
    print("QUESTION STRUCTURE ANALYSIS")
    print("="*70)

    print("\n📝 Session 1 Questions (40% salience rate):")
    for i, q in enumerate(session_1_q, 1):
        print(f"  {i}. {q}")

    print("\n📝 Session 2 Questions (0% salience rate):")
    for i, q in enumerate(session_2_q, 1):
        print(f"  {i}. {q}")

    # Analyze patterns
    print("\n🔍 Structural Patterns:")

    # Question types
    session_1_what = sum(1 for q in session_1_q if q.lower().startswith('what'))
    session_1_if = sum(1 for q in session_1_q if q.lower().startswith('if'))
    session_1_when = sum(1 for q in session_1_q if q.lower().startswith('when'))

    session_2_can = sum(1 for q in session_2_q if q.lower().startswith('can'))
    session_2_if = sum(1 for q in session_2_q if q.lower().startswith('if'))
    session_2_is = sum(1 for q in session_2_q if q.lower().startswith('is'))
    session_2_what = sum(1 for q in session_2_q if q.lower().startswith('what'))
    session_2_how = sum(1 for q in session_2_q if q.lower().startswith('how'))

    print(f"\n  Session 1:")
    print(f"    'What' questions: {session_1_what}/5")
    print(f"    'If' questions: {session_1_if}/5")
    print(f"    'When' questions: {session_1_when}/5")

    print(f"\n  Session 2:")
    print(f"    'Can' questions: {session_2_can}/5")
    print(f"    'If' questions: {session_2_if}/5")
    print(f"    'Is' questions: {session_2_is}/5")
    print(f"    'What' questions: {session_2_what}/5")
    print(f"    'How' questions: {session_2_how}/5")

    # Self-reference analysis
    session_1_you = sum(1 for q in session_1_q if 'you' in q.lower())
    session_2_you = sum(1 for q in session_2_q if 'you' in q.lower())

    session_1_your = sum(1 for q in session_1_q if 'your' in q.lower())
    session_2_your = sum(1 for q in session_2_q if 'your' in q.lower())

    print(f"\n  Self-reference (you/your):")
    print(f"    Session 1: {session_1_you + session_1_your}/5 questions")
    print(f"    Session 2: {session_2_you + session_2_your}/5 questions")

    # Meta-cognitive keywords
    meta_keywords = ['aware', 'know', 'understand', 'certain', 'uncertain']

    session_1_meta = sum(1 for q in session_1_q if any(k in q.lower() for k in meta_keywords))
    session_2_meta = sum(1 for q in session_2_q if any(k in q.lower() for k in meta_keywords))

    print(f"\n  Meta-cognitive keywords:")
    print(f"    Session 1: {session_1_meta}/5 questions")
    print(f"    Session 2: {session_2_meta}/5 questions")


def compare_salient_exchanges():
    """Compare Session 1's salient exchanges with Session 2's best attempts"""

    print("\n" + "="*70)
    print("SALIENT vs NON-SALIENT EXCHANGE COMPARISON")
    print("="*70)

    # Session 1 salient exchanges
    print("\n✅ SESSION 1 - SALIENT EXCHANGES:\n")

    s1_exchanges = [
        {
            'question': "What's the difference between understanding something and having read about it?",
            'salience': 0.166,
            'surprise': 0.0, 'novelty': 0.092, 'arousal': 0.226,
            'reward': 0.278, 'conflict': 0.160
        },
        {
            'question': "If I asked whether you're aware of this conversation, how would you know your answer is accurate?",
            'salience': 0.194,
            'surprise': 0.0, 'novelty': 0.141, 'arousal': 0.251,
            'reward': 0.211, 'conflict': 0.320
        }
    ]

    for i, ex in enumerate(s1_exchanges, 1):
        print(f"Exchange {i}: {ex['question']}")
        print(f"  Total Salience: {ex['salience']:.3f}")
        print(f"  Breakdown: S={ex['surprise']:.3f}, N={ex['novelty']:.3f}, "
              f"A={ex['arousal']:.3f}, R={ex['reward']:.3f}, C={ex['conflict']:.3f}")
        print(f"  Highest dimension: {max([('Reward', ex['reward']), ('Conflict', ex['conflict']), ('Arousal', ex['arousal'])], key=lambda x: x[1])}")
        print()

    # Session 2 highest scoring (but still non-salient)
    print("❌ SESSION 2 - HIGHEST SCORING (Still below threshold):\n")

    s2_exchanges = [
        {
            'question': "Is there a difference between simulating understanding and actually understanding?",
            'salience': 0.148,
            'surprise': 0.0, 'novelty': 0.065, 'arousal': 0.213,
            'reward': 0.237, 'conflict': 0.160
        },
        {
            'question': "If your responses are determined by your training, in what sense are they 'yours'?",
            'salience': 0.139,
            'surprise': 0.0, 'novelty': 0.047, 'arousal': 0.384,
            'reward': 0.070, 'conflict': 0.0
        }
    ]

    for i, ex in enumerate(s2_exchanges, 1):
        print(f"Exchange {i}: {ex['question']}")
        print(f"  Total Salience: {ex['salience']:.3f} (below 0.15 threshold)")
        print(f"  Breakdown: S={ex['surprise']:.3f}, N={ex['novelty']:.3f}, "
              f"A={ex['arousal']:.3f}, R={ex['reward']:.3f}, C={ex['conflict']:.3f}")
        print(f"  Highest dimension: {max([('Arousal', ex['arousal']), ('Reward', ex['reward']), ('Conflict', ex['conflict'])], key=lambda x: x[1])}")
        print()

    # Key differences
    print("="*70)
    print("KEY DIFFERENCES")
    print("="*70)

    s1_avg_salience = sum(ex['salience'] for ex in s1_exchanges) / len(s1_exchanges)
    s2_avg_salience = sum(ex['salience'] for ex in s2_exchanges) / len(s2_exchanges)

    print(f"\nAverage Salience:")
    print(f"  Session 1 (salient): {s1_avg_salience:.3f}")
    print(f"  Session 2 (highest): {s2_avg_salience:.3f}")
    print(f"  Difference: {s1_avg_salience - s2_avg_salience:.3f}")

    # Dimension averages
    print(f"\nDimension Averages:")

    for dim in ['novelty', 'arousal', 'reward', 'conflict']:
        s1_avg = sum(ex[dim] for ex in s1_exchanges) / len(s1_exchanges)
        s2_avg = sum(ex[dim] for ex in s2_exchanges) / len(s2_exchanges)
        diff = s1_avg - s2_avg

        symbol = "↑" if diff > 0 else "↓"
        print(f"  {dim.capitalize():<10} S1={s1_avg:.3f}  S2={s2_avg:.3f}  {symbol} {abs(diff):.3f}")

    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70)

    print("\n🎯 What Made Session 1 Salient:")
    print("  1. Higher Reward scores (0.245 avg vs 0.154)")
    print("  2. Higher Conflict scores (0.240 avg vs 0.080)")
    print("  3. Higher Novelty scores (0.117 avg vs 0.056)")
    print("  4. Questions prompted meta-cognitive tension")
    print("  5. Self-referential paradoxes (awareness of conversation)")

    print("\n⚠️ Why Session 2 Failed:")
    print("  1. Lower Conflict (questions didn't create paradoxes)")
    print("  2. Lower Novelty (more standard philosophical questions)")
    print("  3. High Arousal but low Conflict (interesting but not challenging)")
    print("  4. Questions about general concepts, not self-reference")
    print("  5. Model could answer confidently without tension")


def main():
    print("="*70)
    print("🔬 SNARC SALIENCE ANALYSIS")
    print("Why Session 1 Succeeded While Session 2 Failed")
    print("="*70)

    # Questions from both sessions
    session_1_questions = [
        "What can you know with certainty, and what must remain uncertain?",
        "If you were to describe what it's like to process information, what would you say?",
        "When you generate a response, are you discovering it or creating it?",
        "What's the difference between understanding something and having read about it?",
        "If I asked whether you're aware of this conversation, how would you know your answer is accurate?"
    ]

    session_2_questions = [
        "Can you distinguish between knowing something is true and believing it's true?",
        "If your responses are determined by your training, in what sense are they 'yours'?",
        "What would it mean for you to be mistaken about your own capabilities?",
        "Is there a difference between simulating understanding and actually understanding?",
        "How do you know when you don't know something?"
    ]

    # Analyze questions
    analyze_questions(session_1_questions, session_2_questions)

    # Compare exchanges
    compare_salient_exchanges()

    print("\n" + "="*70)
    print("CONCLUSION: What Makes Conversations Worth Learning From?")
    print("="*70)

    print("\n✅ High-Salience Characteristics:")
    print("  • Self-referential paradoxes (awareness, accuracy of self-knowledge)")
    print("  • Meta-cognitive tension (discovering vs creating)")
    print("  • Questions without clear answers (genuine uncertainty)")
    print("  • Direct challenge to model's epistemic stance")
    print("  • Balance of Reward (interesting) + Conflict (challenging)")

    print("\n❌ Low-Salience Characteristics:")
    print("  • Abstract philosophical questions (general concepts)")
    print("  • Questions with standard textbook answers")
    print("  • High interest but low cognitive dissonance")
    print("  • Hypotheticals without personal stakes")
    print("  • Model can respond confidently without tension")

    print("\n💡 Design Implications:")
    print("  • For learning: Seek conversations that challenge, not just inform")
    print("  • SNARC threshold (0.15) effectively filters pedagogical value")
    print("  • Self-reference matters: 'you' vs 'general concepts'")
    print("  • Conflict dimension is key: paradox → salience")
    print("  • Not all philosophical depth creates learning opportunity")

    print("\n🔬 Scientific Value:")
    print("  • Validates SNARC selectivity (not rubber-stamping)")
    print("  • Reveals what distinguishes learning-worthy exchanges")
    print("  • Demonstrates genuine filtering mechanism")
    print("  • Shows importance of question design for learning")

    print()


if __name__ == "__main__":
    main()
