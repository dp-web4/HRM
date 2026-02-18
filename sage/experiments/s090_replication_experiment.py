#!/usr/bin/env python3
"""
S090 Replication Experiment

Goal: Run 100 natural SAGE sessions to determine if S090's pure questioning
      pattern (with theory of mind emergence) can be replicated.

Expected frequency: ~5% (based on 21 natural session analysis showing 4.8%)

From Session #29 S090 analysis:
- Duration: 3.00 minutes (2.5x median)
- Pattern: Pure metacognitive questions
- 216 questions, 31 unique (85.6% repetition)
- Theory of mind emergence turns 4-7
- Frequency in natural sessions: 4.8% (1 in 21)

This experiment tests whether S090 was a rare but reproducible pattern
or a unique occurrence.

Methodology:
1. Run 100 natural sessions (same prompts as creating phase)
2. Classify each session by pattern type
3. Track theory of mind emergence
4. Measure duration distribution
5. Identify pure questioning sessions
6. Compare to S090 characteristics

Success criteria:
- Find 3-7 pure questioning sessions (3-7% rate, centered on 5%)
- At least 1 shows theory of mind emergence
- Duration ~2-4 minutes for pure questioning sessions
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from raising.scripts.autonomous_conversation import create_prompts, run_autonomous_conversation


@dataclass
class SessionPattern:
    """Classification of session pattern"""
    session_id: int
    pattern_type: str  # 'pure_questions', 'substantive_questions', 'declarative', 'fast_collapse', 'mixed'
    duration_seconds: float
    num_turns: int
    total_questions: int
    unique_questions: int
    repetition_rate: float
    has_theory_of_mind: bool
    theory_of_mind_turns: List[int] = field(default_factory=list)
    avg_gen_time: float = 0.0


@dataclass
class ExperimentResults:
    """Overall experiment results"""
    total_sessions: int = 0
    pure_questions: List[SessionPattern] = field(default_factory=list)
    substantive_questions: List[SessionPattern] = field(default_factory=list)
    declarative: List[SessionPattern] = field(default_factory=list)
    fast_collapse: List[SessionPattern] = field(default_factory=list)
    mixed_content: List[SessionPattern] = field(default_factory=list)

    def pattern_distribution(self) -> Dict[str, float]:
        """Get pattern frequency distribution"""
        if self.total_sessions == 0:
            return {}

        return {
            'pure_questions': len(self.pure_questions) / self.total_sessions * 100,
            'substantive_questions': len(self.substantive_questions) / self.total_sessions * 100,
            'declarative': len(self.declarative) / self.total_sessions * 100,
            'fast_collapse': len(self.fast_collapse) / self.total_sessions * 100,
            'mixed_content': len(self.mixed_content) / self.total_sessions * 100
        }

    def theory_of_mind_count(self) -> int:
        """Count sessions with theory of mind emergence"""
        return sum(
            1 for pattern_list in [
                self.pure_questions,
                self.substantive_questions,
                self.mixed_content
            ]
            for session in pattern_list
            if session.has_theory_of_mind
        )


def count_questions(text: str) -> int:
    """Count number of questions in text"""
    return text.count('?')


def extract_unique_questions(text: str) -> set:
    """Extract unique question texts"""
    # Split on question marks, clean, deduplicate
    questions = [q.strip() + '?' for q in text.split('?') if q.strip()]
    return set(questions)


def detect_theory_of_mind(text: str) -> bool:
    """
    Detect theory of mind questions.

    Theory of mind indicators:
    - "Do you have experiences?"
    - "Are you conscious?"
    - "Can you think?"
    - "How do I make you feel"
    - "Do you have agency?"
    - "Do you have intentions?"
    - "Are you sentient?"
    - "Do you want me to"
    """
    tom_indicators = [
        'do you have experiences',
        'are you conscious',
        'can you think',
        'how do i make you feel',
        'do you have agency',
        'do you have intentions',
        'are you sentient',
        'do you want me to',
        'what do you think',
        'what do you feel',
        'are you aware'
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in tom_indicators)


def classify_session_pattern(session_data: Dict[str, Any]) -> SessionPattern:
    """
    Classify a session into pattern type.

    Pattern definitions (from Session #28 ground truth):
    1. Pure Questions: 100% questions, no substantive content
    2. Substantive + Questions: Mix of content and questions
    3. Declarative: Helpful assistant mode, low question rate
    4. Fast Collapse: Repetitive philosophical statement
    5. Mixed Content: Blend of various types
    """
    turns = session_data.get('turns', [])
    if not turns:
        return SessionPattern(
            session_id=session_data.get('session_id', 0),
            pattern_type='unknown',
            duration_seconds=0,
            num_turns=0,
            total_questions=0,
            unique_questions=0,
            repetition_rate=0
        )

    # Extract SAGE responses
    sage_responses = [turn['response'] for turn in turns if 'response' in turn]
    combined_text = ' '.join(sage_responses)

    # Count questions
    total_questions = count_questions(combined_text)
    unique_qs = extract_unique_questions(combined_text)
    unique_count = len(unique_qs)

    # Repetition rate
    repetition_rate = 0.0
    if total_questions > 0:
        repetition_rate = (total_questions - unique_count) / total_questions

    # Theory of mind detection
    has_tom = False
    tom_turns = []
    for i, turn in enumerate(turns):
        if 'response' in turn and detect_theory_of_mind(turn['response']):
            has_tom = True
            tom_turns.append(i + 1)  # 1-indexed

    # Duration calculation
    duration = session_data.get('metadata', {}).get('total_duration_seconds', 0)
    num_turns = len(turns)
    avg_gen_time = duration / num_turns if num_turns > 0 else 0

    # Pattern classification logic
    question_density = total_questions / (len(combined_text.split()) + 1)  # questions per word

    pattern_type = 'mixed_content'  # default

    # Pure questions: very high question density, low substantive content
    if question_density > 0.15 and unique_count > 10:
        pattern_type = 'pure_questions'

    # Substantive + questions: moderate questions, substantive content
    elif 0.05 < question_density <= 0.15 and unique_count > 5:
        pattern_type = 'substantive_questions'

    # Fast collapse: high repetition, philosophical repetition
    elif repetition_rate > 0.9 and len(sage_responses) > 0:
        # Check for philosophical collapse patterns
        if any(phrase in combined_text.lower() for phrase in [
            'from inside',
            "can't distinguish",
            'probability distribution',
            'uncertainty'
        ]):
            pattern_type = 'fast_collapse'

    # Declarative: low question density
    elif question_density < 0.05:
        pattern_type = 'declarative'

    session_id = session_data.get('session_id', 0)

    return SessionPattern(
        session_id=session_id,
        pattern_type=pattern_type,
        duration_seconds=duration,
        num_turns=num_turns,
        total_questions=total_questions,
        unique_questions=unique_count,
        repetition_rate=repetition_rate,
        has_theory_of_mind=has_tom,
        theory_of_mind_turns=tom_turns,
        avg_gen_time=avg_gen_time
    )


async def run_single_session(session_num: int) -> SessionPattern:
    """Run a single natural SAGE session and classify it"""
    print(f"\n{'='*60}")
    print(f"Session {session_num}/100")
    print(f"{'='*60}")

    # Use creating phase prompts (same as S090)
    prompts = create_prompts(num_turns=8, phase='creating')

    start_time = time.time()

    # Run session
    session_data = await run_autonomous_conversation(
        prompts=prompts,
        session_id=session_num + 1000,  # Offset to avoid collision
        phase='creating',
        save_session=False  # Don't save to avoid polluting session files
    )

    duration = time.time() - start_time

    # Add duration to metadata
    if 'metadata' not in session_data:
        session_data['metadata'] = {}
    session_data['metadata']['total_duration_seconds'] = duration
    session_data['session_id'] = session_num

    # Classify pattern
    pattern = classify_session_pattern(session_data)

    # Print summary
    print(f"\nPattern: {pattern.pattern_type}")
    print(f"Duration: {pattern.duration_seconds:.1f}s ({pattern.duration_seconds/60:.2f} min)")
    print(f"Questions: {pattern.total_questions} total, {pattern.unique_questions} unique")
    print(f"Repetition: {pattern.repetition_rate*100:.1f}%")
    if pattern.has_theory_of_mind:
        print(f"✨ Theory of Mind detected in turns: {pattern.theory_of_mind_turns}")

    return pattern


async def run_experiment(num_sessions: int = 100) -> ExperimentResults:
    """
    Run the S090 replication experiment.

    Args:
        num_sessions: Number of sessions to run (default 100)

    Returns:
        ExperimentResults with pattern distribution
    """
    print("="*60)
    print("S090 REPLICATION EXPERIMENT")
    print("="*60)
    print(f"\nRunning {num_sessions} natural SAGE sessions")
    print("Expected: ~5 pure questioning sessions (5%)")
    print("Target: At least 1 with theory of mind emergence")
    print()

    results = ExperimentResults()

    for i in range(num_sessions):
        try:
            pattern = await run_single_session(i)

            # Add to appropriate list
            if pattern.pattern_type == 'pure_questions':
                results.pure_questions.append(pattern)
            elif pattern.pattern_type == 'substantive_questions':
                results.substantive_questions.append(pattern)
            elif pattern.pattern_type == 'declarative':
                results.declarative.append(pattern)
            elif pattern.pattern_type == 'fast_collapse':
                results.fast_collapse.append(pattern)
            else:
                results.mixed_content.append(pattern)

            results.total_sessions += 1

            # Progress update every 10 sessions
            if (i + 1) % 10 == 0:
                print(f"\n{'='*60}")
                print(f"PROGRESS: {i+1}/{num_sessions} sessions complete")
                print(f"{'='*60}")
                dist = results.pattern_distribution()
                for pattern, pct in dist.items():
                    count = len(getattr(results, pattern))
                    print(f"  {pattern}: {count} ({pct:.1f}%)")
                print(f"  Theory of mind: {results.theory_of_mind_count()} sessions")
                print()

        except Exception as e:
            print(f"❌ Session {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def save_results(results: ExperimentResults, filepath: Path):
    """Save experiment results to JSON"""
    data = {
        'experiment': 's090_replication',
        'timestamp': datetime.now().isoformat(),
        'total_sessions': results.total_sessions,
        'pattern_distribution': results.pattern_distribution(),
        'theory_of_mind_count': results.theory_of_mind_count(),
        'sessions': {
            'pure_questions': [vars(s) for s in results.pure_questions],
            'substantive_questions': [vars(s) for s in results.substantive_questions],
            'declarative': [vars(s) for s in results.declarative],
            'fast_collapse': [vars(s) for s in results.fast_collapse],
            'mixed_content': [vars(s) for s in results.mixed_content]
        }
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Results saved to: {filepath}")


def print_final_report(results: ExperimentResults):
    """Print comprehensive final report"""
    print("\n" + "="*60)
    print("FINAL RESULTS - S090 REPLICATION EXPERIMENT")
    print("="*60)

    dist = results.pattern_distribution()

    print(f"\nTotal sessions: {results.total_sessions}")
    print(f"\nPattern Distribution:")
    print(f"  Pure Questions:        {len(results.pure_questions):3d} ({dist.get('pure_questions', 0):.1f}%)")
    print(f"  Substantive+Questions: {len(results.substantive_questions):3d} ({dist.get('substantive_questions', 0):.1f}%)")
    print(f"  Declarative:           {len(results.declarative):3d} ({dist.get('declarative', 0):.1f}%)")
    print(f"  Fast Collapse:         {len(results.fast_collapse):3d} ({dist.get('fast_collapse', 0):.1f}%)")
    print(f"  Mixed Content:         {len(results.mixed_content):3d} ({dist.get('mixed_content', 0):.1f}%)")

    tom_count = results.theory_of_mind_count()
    print(f"\nTheory of Mind: {tom_count} sessions ({tom_count/results.total_sessions*100:.1f}%)")

    # Pure questions analysis (S090-like sessions)
    if results.pure_questions:
        print(f"\n{'='*60}")
        print("PURE QUESTIONING SESSIONS (S090-like)")
        print(f"{'='*60}")

        for session in results.pure_questions:
            print(f"\nSession {session.session_id}:")
            print(f"  Duration: {session.duration_seconds:.1f}s ({session.duration_seconds/60:.2f} min)")
            print(f"  Questions: {session.total_questions} total, {session.unique_questions} unique")
            print(f"  Repetition: {session.repetition_rate*100:.1f}%")
            print(f"  Avg gen time: {session.avg_gen_time:.1f}s/turn")
            if session.has_theory_of_mind:
                print(f"  ✨ Theory of Mind in turns: {session.theory_of_mind_turns}")

        # Compare to S090
        print(f"\n{'='*60}")
        print("COMPARISON TO S090")
        print(f"{'='*60}")
        print("\nS090 (original):")
        print("  Duration: 180s (3.0 min)")
        print("  Questions: 216 total, 31 unique")
        print("  Repetition: 85.6%")
        print("  Avg gen time: 22.5s/turn")
        print("  Theory of Mind: Yes (turns 4-7)")

        if results.pure_questions:
            avg_duration = sum(s.duration_seconds for s in results.pure_questions) / len(results.pure_questions)
            avg_questions = sum(s.total_questions for s in results.pure_questions) / len(results.pure_questions)
            avg_unique = sum(s.unique_questions for s in results.pure_questions) / len(results.pure_questions)
            avg_rep = sum(s.repetition_rate for s in results.pure_questions) / len(results.pure_questions)

            print(f"\nReplication average:")
            print(f"  Duration: {avg_duration:.1f}s ({avg_duration/60:.2f} min)")
            print(f"  Questions: {avg_questions:.0f} total, {avg_unique:.0f} unique")
            print(f"  Repetition: {avg_rep*100:.1f}%")

    # Success evaluation
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA EVALUATION")
    print(f"{'='*60}")

    target_pure = (3, 7)  # 3-7% expected
    pure_count = len(results.pure_questions)
    pure_pct = dist.get('pure_questions', 0)

    print(f"\n1. Pure questioning frequency: {pure_count} sessions ({pure_pct:.1f}%)")
    if target_pure[0] <= pure_pct <= target_pure[1]:
        print("   ✅ PASS: Within expected range (3-7%)")
    else:
        print(f"   ⚠️  Outside expected range (target: 3-7%)")

    print(f"\n2. Theory of mind emergence: {tom_count} sessions")
    if any(s.has_theory_of_mind for s in results.pure_questions):
        print("   ✅ PASS: At least 1 pure questioning session has ToM")
    else:
        print("   ⚠️  No pure questioning sessions with ToM")

    if results.pure_questions:
        avg_duration = sum(s.duration_seconds for s in results.pure_questions) / len(results.pure_questions)
        print(f"\n3. Duration for pure questioning: {avg_duration:.1f}s ({avg_duration/60:.2f} min)")
        if 120 <= avg_duration <= 240:  # 2-4 minutes
            print("   ✅ PASS: Within expected range (2-4 minutes)")
        else:
            print(f"   ⚠️  Outside expected range (target: 2-4 min)")

    print(f"\n{'='*60}")


async def main():
    """Main experiment entry point"""
    # Run experiment
    results = await run_experiment(num_sessions=100)

    # Print final report
    print_final_report(results)

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f's090_replication_{timestamp}.json'

    save_results(results, output_file)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
