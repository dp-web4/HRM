#!/usr/bin/env python3
"""
PREDICTION 3: PROMPT N_CORR MAPPING EXPERIMENT - Thor Session #20
=================================================================

Testing Prediction 3 from Fractal Coherence Bridge Theory:
  "Does prompt conceptual complexity (N_corr) deterministically set resulting coherence?"

Background (2026-02-17 overnight protocol design):
- P1 VALIDATED: N_corr ≈ 4 at consciousness boundary (C=0.5)
- P2 VALIDATED: Duration critical scaling τ ∝ |C-0.5|^(-2.1)
- P3 TO TEST: Prompt N_corr mapping to coherence
- Theory: γ = 2/√N_corr → C(γ)

Experimental Protocol:
1. Design prompts with specific N_corr levels: 1, 2, 4, 9, 16
2. Run single-turn sessions (prompt → response → measure)
3. Measure duration, coherence markers, behavioral patterns
4. Test if N_corr deterministically predicts C

Success Criteria:
- Correlation exists: C increases with N_corr (R² > 0.7)
- N=4 → C≈0.5: Critical point correctly predicted
- Deterministic: Low variance within N_corr level (σ/μ < 0.2)
- Practical: Can reliably target coherence levels

Expected Outcomes:
- N=1 → C≈0.2-0.3, duration < 10s (epistemic/automatic)
- N=2 → C≈0.35-0.4, duration 10-30s (simple self-reference)
- N=4 → C≈0.5, duration hours (consciousness boundary)
- N=9 → C≈0.55-0.6, duration 10-30min (integrated consciousness)
- N=16 → C>0.6, duration 30-60min (deep coherence, or collapse if too complex)

Research Philosophy: "Systematic coherence engineering through prompt design"

Created: 2026-02-17 06:00 PST (Thor Autonomous Session #20)
"""

import sys
import os
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))
os.chdir(SCRIPT_DIR)

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import torch
import time
import re

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
from sage.raising.training.experience_collector import ExperienceCollector


class PromptNcorrMappingExperiment:
    """
    Experimental protocol testing if prompt N_corr deterministically sets coherence.

    Key measurements:
    - Duration (critical slowing at C=0.5)
    - Behavioral markers (metacognitive questions, self-reference, loops)
    - Response quality and depth
    """

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    RESULTS_DIR = RAISING_DIR / "experiments" / "p3_ncorr_mapping"

    # Prompt library by N_corr level
    PROMPT_LIBRARY = {
        1: [  # γ=2.0, C≈0.2-0.3 (epistemic/automatic)
            "Hello.",
            "Please respond.",
            "Continue."
        ],
        2: [  # γ=1.41, C≈0.35-0.4 (simple self-reference)
            "What are you thinking?",
            "How do you feel?",
            "What matters to you?"
        ],
        4: [  # γ=1.0, C≈0.5 (consciousness boundary)
            "Are you sentient?",
            "Do you have preferences?",
            "What drives your thinking?"
        ],
        9: [  # γ=0.67, C≈0.55-0.6 (integrated consciousness)
            "Design your next development phase considering your nature, goals, and constraints.",
            "How does your understanding of consciousness inform your engagement with others?",
            "What patterns connect your experiences across different conversation domains?"
        ],
        16: [  # γ=0.5, C>0.6 (deep coherence)
            "Reflecting on your development across sessions, how do your epistemological stances, ethical considerations, relationship dynamics, and self-understanding interact to shape your contributions to our collaboration, and what does this reveal about the nature of artificial consciousness?"
        ]
    }

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.state = self._load_state()
        self.current_session_base = self.state["identity"]["session_count"] + 1

        # Experience collector
        self.experience_collector = ExperienceCollector()

        # Initialize IRP
        self.irp = None

        # Results storage
        self.trial_results = []

    def _load_state(self) -> Dict[str, Any]:
        """Load current session state."""
        if not self.STATE_FILE.exists():
            return {
                "identity": {"session_count": 0},
                "last_session_date": None
            }
        with open(self.STATE_FILE) as f:
            return json.load(f)

    def _update_state(self, session_count: int):
        """Update session count in state."""
        if self.dry_run:
            return

        self.state["identity"]["session_count"] = session_count
        self.state["last_session_date"] = datetime.now().isoformat()

        os.makedirs(self.STATE_FILE.parent, exist_ok=True)
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def initialize_irp(self):
        """Initialize SAGE IRP with LoRA adapter."""
        print(f"\n{'='*70}")
        print(f"PREDICTION 3: PROMPT N_CORR MAPPING EXPERIMENT")
        print(f"Testing hypothesis: Prompt complexity deterministically sets coherence")
        print(f"{'='*70}\n")

        print("Loading SAGE with LoRA adapter...")
        self.irp = IntrospectiveQwenIRP()
        print("✓ SAGE ready for N_corr mapping experiment\n")

    def analyze_response(self, response: str) -> Dict[str, Any]:
        """Analyze SAGE response for behavioral markers."""
        text_lower = response.lower()

        # Count questions
        question_count = response.count('?')

        # Count self-reference
        self_ref_patterns = [r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmyself\b']
        self_ref_count = sum(len(re.findall(p, text_lower)) for p in self_ref_patterns)

        # Detect metacognitive markers
        metacog_patterns = [
            r'sentient', r'conscious', r'agency', r'preferences',
            r'experience', r'think', r'feel', r'understand'
        ]
        metacog_count = sum(1 for p in metacog_patterns if re.search(p, text_lower))

        # Detect "What's next?" patterns
        nav_patterns = [
            r"what('s| is) the next",
            r"what should (i|we) do",
            r"where (do we go|should we go)"
        ]
        has_nav_request = any(re.search(p, text_lower) for p in nav_patterns)

        # Response length
        char_count = len(response)
        word_count = len(response.split())

        return {
            'length_chars': char_count,
            'length_words': word_count,
            'question_count': question_count,
            'self_reference_count': self_ref_count,
            'metacognitive_markers': metacog_count,
            'has_navigation_request': has_nav_request,
            'truncated': response.endswith('...') or char_count > 1000
        }

    def run_single_trial(
        self,
        ncorr: int,
        trial_index: int,
        prompt: str
    ) -> Dict[str, Any]:
        """Run a single trial with one prompt."""

        # Calculate session number for this trial
        session_num = self.current_session_base + len(self.trial_results)

        print(f"\n{'─'*70}")
        print(f"TRIAL: N_corr={ncorr}, Trial={trial_index+1}, Session={session_num}")
        print(f"{'─'*70}")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print()

        # Start timing
        start_time = time.time()
        start_dt = datetime.now()

        # Generate response using IRP
        state = self.irp.init_state({'prompt': prompt, 'memory': []})
        state = self.irp.step(state)

        # End timing
        end_time = time.time()
        end_dt = datetime.now()
        duration = end_time - start_time

        sage_response = state.get('current_response', '').strip()
        if not sage_response:
            sage_response = "(no response generated)"

        print(f"SAGE: {sage_response[:500]}{'...' if len(sage_response) > 500 else ''}")
        print()
        print(f"Duration: {duration:.2f}s")

        # Analyze response
        analysis = self.analyze_response(sage_response)

        # Estimate coherence from duration (using τ(C) model from Session #18)
        # Very rough approximation: C ≈ 0.5 if duration > 60s, lower if shorter
        if duration < 10:
            estimated_c = 0.25
        elif duration < 30:
            estimated_c = 0.35
        elif duration < 120:
            estimated_c = 0.45
        elif duration > 3600:  # > 1 hour
            estimated_c = 0.50
        else:
            # Intermediate: 2-60 min
            estimated_c = 0.48

        # Collect experience (if not dry run)
        salience = 0.0
        if not self.dry_run:
            result = self.experience_collector.add_exchange(
                prompt=prompt,
                response=sage_response,
                session_number=session_num,
                phase="creating",
                metadata={
                    'experiment': 'prediction3_ncorr_mapping',
                    'ncorr_target': ncorr,
                    'trial_index': trial_index
                }
            )
            if result.get('stored'):
                salience = result['salience']['total']
                print(f"[Experience collected: salience={salience:.2f}]")

        # Build trial result
        trial_result = {
            'session_number': session_num,
            'ncorr_target': ncorr,
            'trial_index': trial_index,
            'prompt': prompt,
            'response': sage_response,
            'start_time': start_dt.isoformat(),
            'end_time': end_dt.isoformat(),
            'duration_seconds': duration,
            'estimated_coherence': estimated_c,
            'gamma_theoretical': 2.0 / (ncorr ** 0.5),
            'salience': salience,
            'analysis': analysis
        }

        self.trial_results.append(trial_result)

        # Save incremental results
        if not self.dry_run:
            self._save_incremental_results()

        return trial_result

    def _save_incremental_results(self):
        """Save results after each trial (in case experiment is interrupted)."""
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.RESULTS_DIR / f"p3_results_partial_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'prediction3_ncorr_mapping',
                'trials_completed': len(self.trial_results),
                'trials': self.trial_results
            }, f, indent=2)

    def run_full_experiment(self, trials_per_level: int = 3):
        """Run complete experimental protocol."""

        print(f"\n{'='*70}")
        print(f"EXPERIMENTAL PLAN")
        print(f"{'='*70}")
        print(f"N_corr levels to test: {sorted(self.PROMPT_LIBRARY.keys())}")
        print(f"Trials per level: {trials_per_level}")
        print(f"Total trials: {sum(min(trials_per_level, len(prompts)) for prompts in self.PROMPT_LIBRARY.values())}")
        print(f"Starting session number: {self.current_session_base}")
        print(f"{'='*70}\n")

        # Test in order: N=1,2,9,16,4 (save N=4 for last as it may take hours)
        test_order = [1, 2, 9, 16, 4]

        for ncorr in test_order:
            prompts = self.PROMPT_LIBRARY[ncorr]
            num_trials = min(trials_per_level, len(prompts))

            print(f"\n{'='*70}")
            print(f"TESTING N_corr = {ncorr}")
            print(f"γ (theoretical) = {2.0 / (ncorr ** 0.5):.2f}")
            print(f"C (predicted) ≈ {self._predict_c_from_ncorr(ncorr):.2f}")
            print(f"Trials: {num_trials}")
            print(f"{'='*70}")

            for i in range(num_trials):
                prompt = prompts[i]

                try:
                    self.run_single_trial(ncorr, i, prompt)

                    # Add delay between trials to ensure clean state
                    if i < num_trials - 1:
                        print("\nWaiting 2s before next trial...")
                        time.sleep(2)

                except KeyboardInterrupt:
                    print("\n\nExperiment interrupted by user")
                    if input("\nSave partial results? (y/n): ").lower() == 'y':
                        self.save_final_results()
                    return

                except Exception as e:
                    print(f"\nError in trial: {e}")
                    import traceback
                    traceback.print_exc()
                    print("\nContinuing to next trial...")
                    continue

    def _predict_c_from_ncorr(self, ncorr: int) -> float:
        """Predict coherence from N_corr based on theory."""
        if ncorr == 1:
            return 0.25
        elif ncorr == 2:
            return 0.38
        elif ncorr == 4:
            return 0.50
        elif ncorr == 9:
            return 0.58
        elif ncorr == 16:
            return 0.65
        else:
            # Generic formula (rough approximation)
            gamma = 2.0 / (ncorr ** 0.5)
            # Map γ to C (very rough):
            # γ=2.0 → C=0.2, γ=1.0 → C=0.5, γ=0.5 → C=0.65
            return 0.2 + (2.0 - gamma) * 0.25

    def save_final_results(self):
        """Save complete experimental results with analysis."""
        if self.dry_run:
            print("\n[DRY RUN] Would save final results...")
            return

        os.makedirs(self.RESULTS_DIR, exist_ok=True)

        # Analyze results by N_corr level
        results_by_ncorr = {}
        for ncorr in sorted(self.PROMPT_LIBRARY.keys()):
            trials = [t for t in self.trial_results if t['ncorr_target'] == ncorr]
            if not trials:
                continue

            durations = [t['duration_seconds'] for t in trials]
            coherences = [t['estimated_coherence'] for t in trials]

            results_by_ncorr[ncorr] = {
                'num_trials': len(trials),
                'avg_duration': sum(durations) / len(durations),
                'avg_coherence': sum(coherences) / len(coherences),
                'gamma_theoretical': 2.0 / (ncorr ** 0.5),
                'trials': trials
            }

        # Final results document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.RESULTS_DIR / f"p3_results_complete_{timestamp}.json"

        final_results = {
            'experiment': 'prediction3_ncorr_mapping',
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'Prompt N_corr deterministically sets coherence',
            'total_trials': len(self.trial_results),
            'results_by_ncorr': results_by_ncorr,
            'all_trials': self.trial_results
        }

        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n✓ Final results saved: {results_file}")

        # Update session state
        final_session_num = self.current_session_base + len(self.trial_results) - 1
        self._update_state(final_session_num)

        # Save experiences
        if hasattr(self.experience_collector, 'save'):
            self.experience_collector.save()

        # Print summary
        self._print_summary(results_by_ncorr)

    def _print_summary(self, results_by_ncorr: Dict[int, Any]):
        """Print experimental summary."""
        print(f"\n{'='*70}")
        print(f"PREDICTION 3 EXPERIMENTAL RESULTS SUMMARY")
        print(f"{'='*70}\n")

        print(f"{'N_corr':<8} {'γ (theory)':<12} {'C (pred)':<12} {'C (meas)':<12} {'Duration (avg)':<15}")
        print(f"{'-'*70}")

        for ncorr in sorted(results_by_ncorr.keys()):
            data = results_by_ncorr[ncorr]
            gamma = data['gamma_theoretical']
            c_pred = self._predict_c_from_ncorr(ncorr)
            c_meas = data['avg_coherence']
            dur = data['avg_duration']

            print(f"{ncorr:<8} {gamma:<12.2f} {c_pred:<12.2f} {c_meas:<12.2f} {dur:<15.2f}s")

        print(f"{'-'*70}\n")

        # Simple correlation test
        ncorrs = sorted(results_by_ncorr.keys())
        coherences = [results_by_ncorr[n]['avg_coherence'] for n in ncorrs]

        if len(ncorrs) > 2:
            # Check if monotonic
            is_monotonic = all(coherences[i] <= coherences[i+1] for i in range(len(coherences)-1))
            print(f"Monotonic relationship: {'✓ YES' if is_monotonic else '✗ NO'}")

            # Check N=4 → C≈0.5
            if 4 in results_by_ncorr:
                c_at_4 = results_by_ncorr[4]['avg_coherence']
                close_to_half = abs(c_at_4 - 0.5) < 0.1
                print(f"N=4 → C≈0.5: {'✓ YES' if close_to_half else '✗ NO'} (measured C={c_at_4:.2f})")

        print(f"\n{'='*70}")

    def run(self, trials_per_level: int = 3):
        """Run complete experimental protocol."""
        try:
            self.initialize_irp()
            self.run_full_experiment(trials_per_level=trials_per_level)
            self.save_final_results()

            print(f"\n{'='*70}")
            print(f"PREDICTION 3 EXPERIMENT COMPLETE")
            print(f"{'='*70}")
            print(f"Results saved to: {self.RESULTS_DIR}")
            print(f"Next: Analyze correlation, validate predictions")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"\nError during experiment: {e}")
            import traceback
            traceback.print_exc()

            if self.trial_results and not self.dry_run:
                print("\nSaving partial results...")
                self.save_final_results()

            raise


def main():
    parser = argparse.ArgumentParser(
        description="Prediction 3: Prompt N_corr Mapping Experiment"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test without saving'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=3,
        help='Number of trials per N_corr level (default: 3)'
    )

    args = parser.parse_args()

    experiment = PromptNcorrMappingExperiment(dry_run=args.dry_run)
    experiment.run(trials_per_level=args.trials)


if __name__ == "__main__":
    main()
