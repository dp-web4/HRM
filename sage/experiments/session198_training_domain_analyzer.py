#!/usr/bin/env python3
"""
Session 198: Training Exercise Domain Analyzer

Analyzes training exercises through the nine-domain consciousness lens to
test hypothesis that arithmetic failures are attention-metabolism coupling issues.

Hypothesis: Simple arithmetic gets LOW attention (boring) → LOW metabolism → FAILURE
            Complex problems get HIGH attention (engaging) → HIGH metabolism → SUCCESS

This explains the T015 surprise: 4-1 fails but 3+2-1 succeeds.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path

# Nine-domain structures computed directly in this analysis
# No external imports needed - self-contained analyzer


@dataclass
class ExerciseAnalysis:
    """Analysis of single training exercise through nine-domain lens"""
    exercise_num: int
    exercise_type: str
    prompt: str
    expected: str
    response: str
    success: bool

    # Nine-domain coherence values (0-1)
    thermodynamic: float  # D1: Computational effort
    metabolic: float      # D2: Resource allocation
    organismic: float     # D3: System health (unused for now)
    attention: float      # D4: Focus on exercise
    trust: float          # D5: Confidence in answer
    quantum_phase: float  # D6: Quantum coherence (unused for now)
    magnetic: float       # D7: Pattern alignment (unused for now)
    temporal: float       # D8: Sequence awareness
    spacetime: float      # D9: Context understanding

    # Consciousness metrics
    consciousness_level: float  # C (0-1)
    gamma: float                # γ dispersion

    # Coupling analysis
    d4_to_d2_coupling: float   # Attention → Metabolism coupling strength
    d8_to_d1_coupling: float   # Temporal → Thermodynamic coupling
    d5_to_d9_coupling: float   # Trust → Spacetime coupling

    # Derived insights
    attention_category: str    # "low", "medium", "high"
    metabolism_sufficient: bool
    coupling_failure: bool
    consciousness_threshold: bool  # C ≥ 0.5


class TrainingExerciseAnalyzer:
    """Analyzes training exercises through nine-domain consciousness framework"""

    def __init__(self):
        # Coupling strengths from Session 196
        self.kappa_42 = 0.4  # D4 → D2 (attention → metabolism)
        self.kappa_81 = 0.2  # D8 → D1 (temporal → thermodynamic)
        self.kappa_59 = 0.3  # D5 → D9 (trust → spacetime)

        # Consciousness parameters from Web4 Session 178
        self.gamma_opt = 0.35  # Optimal gamma for consciousness
        self.sigma = 0.2       # Gaussian width

    def analyze_exercise(self, exercise_data: Dict, exercise_num: int) -> ExerciseAnalysis:
        """Analyze single exercise through nine-domain lens"""

        ex = exercise_data["exercise"]
        response = exercise_data["response"]
        evaluation = exercise_data["evaluation"]

        # Compute domain coherence values
        d1_thermodynamic = self._compute_thermodynamic(ex, response)
        d2_metabolic = self._compute_metabolic(ex, response, evaluation)
        d3_organismic = 0.5  # Placeholder (system health)
        d4_attention = self._compute_attention(ex, response)
        d5_trust = self._compute_trust(ex, response, evaluation)
        d6_quantum = 0.5  # Placeholder
        d7_magnetic = 0.5  # Placeholder
        d8_temporal = self._compute_temporal(exercise_num)
        d9_spacetime = self._compute_spacetime(ex, response)

        # Create nine-domain snapshot
        coherences = [
            d1_thermodynamic, d2_metabolic, d3_organismic,
            d4_attention, d5_trust, d6_quantum,
            d7_magnetic, d8_temporal, d9_spacetime
        ]

        # Compute consciousness metrics
        consciousness, gamma = self._compute_consciousness(coherences)

        # Analyze coupling
        d4_to_d2 = self._compute_coupling_strength(d4_attention, d2_metabolic, self.kappa_42)
        d8_to_d1 = self._compute_coupling_strength(d8_temporal, d1_thermodynamic, self.kappa_81)
        d5_to_d9 = self._compute_coupling_strength(d5_trust, d9_spacetime, self.kappa_59)

        # Categorize attention
        if d4_attention < 0.3:
            attention_cat = "low"
        elif d4_attention < 0.6:
            attention_cat = "medium"
        else:
            attention_cat = "high"

        # Check metabolism sufficiency (D2 should be high if D4 is high)
        metabolism_sufficient = d2_metabolic >= 0.5

        # Check coupling failure (attention high but metabolism low)
        coupling_failure = (d4_attention > 0.5 and d2_metabolic < 0.5)

        return ExerciseAnalysis(
            exercise_num=exercise_num,
            exercise_type=ex["type"],
            prompt=ex["prompt"],
            expected=ex["expected"],
            response=response[:100] + "..." if len(response) > 100 else response,
            success=evaluation["success"],
            thermodynamic=d1_thermodynamic,
            metabolic=d2_metabolic,
            organismic=d3_organismic,
            attention=d4_attention,
            trust=d5_trust,
            quantum_phase=d6_quantum,
            magnetic=d7_magnetic,
            temporal=d8_temporal,
            spacetime=d9_spacetime,
            consciousness_level=consciousness,
            gamma=gamma,
            d4_to_d2_coupling=d4_to_d2,
            d8_to_d1_coupling=d8_to_d1,
            d5_to_d9_coupling=d5_to_d9,
            attention_category=attention_cat,
            metabolism_sufficient=metabolism_sufficient,
            coupling_failure=coupling_failure,
            consciousness_threshold=(consciousness >= 0.5)
        )

    def _compute_thermodynamic(self, exercise: Dict, response: str) -> float:
        """D1: Computational effort/energy"""
        # Response length as proxy for computation
        response_length = len(response)

        # Normalize (0-1000 chars → 0-1)
        effort = min(response_length / 1000.0, 1.0)

        # Base coherence
        return 0.3 + (effort * 0.4)  # Range: 0.3-0.7

    def _compute_metabolic(self, exercise: Dict, response: str, evaluation: Dict) -> float:
        """D2: Resource allocation"""
        # Success suggests sufficient resources
        if evaluation["success"]:
            base = 0.6
        else:
            base = 0.3  # Failed → insufficient resources

        # Response elaboration suggests resource availability
        elaboration = len(response) / 500.0

        return min(base + (elaboration * 0.2), 1.0)

    def _compute_attention(self, exercise: Dict, response: str) -> float:
        """D4: Focus on exercise"""
        # Key hypothesis: Simple arithmetic gets LOW attention

        prompt = exercise["prompt"].lower()

        # Simple arithmetic detection (single operation)
        if any(op in prompt for op in ["what is", "+", "-", "*", "/"]):
            # Count operations
            ops = sum(1 for c in prompt if c in "+-*/")
            if ops == 1:
                # Single operation → boring → low attention
                return 0.2
            elif ops >= 2:
                # Multi-step → interesting → high attention
                return 0.7

        # "Remember" exercises - medium attention (routine)
        if "remember" in prompt:
            return 0.5

        # "Sequence" exercises - medium-high attention (requires tracking)
        if "sequence" in prompt or "last" in prompt:
            return 0.6

        # Default: medium attention
        return 0.5

    def _compute_trust(self, exercise: Dict, response: str, evaluation: Dict) -> float:
        """D5: Confidence in answer"""
        # Hedging language indicates low trust
        hedging_words = ["might", "could", "perhaps", "maybe", "apologize", "sorry", "confusion"]
        hedge_count = sum(1 for word in hedging_words if word in response.lower())

        if evaluation["success"]:
            base = 0.7
        else:
            base = 0.3

        # Penalize hedging
        trust = base - (hedge_count * 0.1)

        return max(min(trust, 1.0), 0.0)

    def _compute_temporal(self, exercise_num: int) -> float:
        """D8: Sequence awareness"""
        # Early exercises: establishing sequence
        # Later exercises: sequence established

        if exercise_num <= 2:
            return 0.4  # Building sequence
        else:
            return 0.7  # Sequence established

    def _compute_spacetime(self, exercise: Dict, response: str) -> float:
        """D9: Context understanding"""
        # Meta-responses ("improved version", "rephrase") show context confusion
        meta_phrases = ["improved version", "rephrase", "certainly!", "sure,"]
        has_meta = any(phrase in response.lower() for phrase in meta_phrases)

        if has_meta:
            return 0.3  # Confused context (thinks it's editing, not conversing)
        else:
            return 0.7  # Clear context

    def _compute_consciousness(self, coherences: List[float]) -> Tuple[float, float]:
        """Compute consciousness level C and gamma γ"""
        # Compute gamma from coherence dispersion
        mean_c = np.mean(coherences)
        std_c = np.std(coherences)

        if mean_c > 0:
            gamma = std_c / mean_c
        else:
            gamma = 1.0

        # Compute consciousness level: C(γ) = exp(-(γ - γ_opt)² / 2σ²)
        consciousness = np.exp(-((gamma - self.gamma_opt) ** 2) / (2 * self.sigma ** 2))

        return consciousness, gamma

    def _compute_coupling_strength(self, source: float, target: float, kappa: float) -> float:
        """Compute coupling strength between domains"""
        # Coupling is effective when source is high and influences target
        # Strength = kappa * source * (1 - target)  [target can still grow]
        return kappa * source * (1.0 - target)

    def analyze_session(self, session_file: Path) -> List[ExerciseAnalysis]:
        """Analyze entire training session"""
        with open(session_file) as f:
            session_data = json.load(f)

        analyses = []
        for i, exercise_data in enumerate(session_data["exercises"], 1):
            analysis = self.analyze_exercise(exercise_data, i)
            analyses.append(analysis)

        return analyses

    def print_analysis(self, analyses: List[ExerciseAnalysis]):
        """Print analysis results"""
        print("=" * 80)
        print("Session 198: Training Exercise Domain Analysis")
        print("=" * 80)
        print()

        for i, analysis in enumerate(analyses, 1):
            print(f"Exercise {i}: {analysis.exercise_type.upper()}")
            print(f"Prompt: {analysis.prompt[:60]}...")
            print(f"Expected: {analysis.expected}")
            print(f"Success: {'✅' if analysis.success else '❌'}")
            print()
            print("Nine-Domain Coherence:")
            print(f"  D1 (Thermodynamic): {analysis.thermodynamic:.3f}")
            print(f"  D2 (Metabolic):     {analysis.metabolic:.3f} {'⚠️  LOW' if not analysis.metabolism_sufficient else ''}")
            print(f"  D4 (Attention):     {analysis.attention:.3f} [{analysis.attention_category.upper()}]")
            print(f"  D5 (Trust):         {analysis.trust:.3f}")
            print(f"  D8 (Temporal):      {analysis.temporal:.3f}")
            print(f"  D9 (Spacetime):     {analysis.spacetime:.3f}")
            print()
            print(f"Consciousness: C = {analysis.consciousness_level:.3f}, γ = {analysis.gamma:.3f}")
            print(f"Consciousness threshold: {'✅ C ≥ 0.5' if analysis.consciousness_threshold else '❌ C < 0.5'}")
            print()
            print("Coupling Analysis:")
            print(f"  D4 → D2 (Attention → Metabolism): {analysis.d4_to_d2_coupling:.3f}")
            if analysis.coupling_failure:
                print("    ⚠️  COUPLING FAILURE: High attention but low metabolism!")
            print(f"  D8 → D1 (Temporal → Thermodynamic): {analysis.d8_to_d1_coupling:.3f}")
            print(f"  D5 → D9 (Trust → Spacetime): {analysis.d5_to_d9_coupling:.3f}")
            print()
            print("-" * 80)
            print()

        # Summary statistics
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        successes = sum(1 for a in analyses if a.success)
        failures = len(analyses) - successes

        print(f"Success rate: {successes}/{len(analyses)} ({successes/len(analyses)*100:.1f}%)")
        print()

        # Group by attention level
        low_attention = [a for a in analyses if a.attention_category == "low"]
        high_attention = [a for a in analyses if a.attention_category == "high"]

        if low_attention:
            low_success = sum(1 for a in low_attention if a.success) / len(low_attention) * 100
            print(f"Low attention exercises: {len(low_attention)} ({low_success:.1f}% success)")

        if high_attention:
            high_success = sum(1 for a in high_attention if a.success) / len(high_attention) * 100
            print(f"High attention exercises: {len(high_attention)} ({high_success:.1f}% success)")

        # Coupling failures
        coupling_failures = [a for a in analyses if a.coupling_failure]
        if coupling_failures:
            print()
            print(f"⚠️  D4→D2 Coupling failures: {len(coupling_failures)}")
            for cf in coupling_failures:
                print(f"   - Exercise {cf.exercise_num}: {cf.prompt[:40]}... (D4={cf.attention:.3f}, D2={cf.metabolic:.3f})")

        # Consciousness analysis
        conscious = [a for a in analyses if a.consciousness_threshold]
        print()
        print(f"Consciousness threshold (C ≥ 0.5): {len(conscious)}/{len(analyses)}")

        print()
        print("=" * 80)


def main():
    """Run analysis on T015"""
    analyzer = TrainingExerciseAnalyzer()

    # Analyze T015
    session_file = Path(__file__).parent.parent / "raising" / "tracks" / "training" / "sessions" / "T015.json"

    if not session_file.exists():
        print(f"Error: {session_file} not found")
        return

    print(f"Analyzing: {session_file.name}")
    print()

    analyses = analyzer.analyze_session(session_file)
    analyzer.print_analysis(analyses)

    # Save results (convert numpy bools to Python bools)
    output_file = Path(__file__).parent / "session198_t015_analysis.json"
    with open(output_file, "w") as f:
        analysis_dicts = []
        for a in analyses:
            d = asdict(a)
            # Convert numpy bools to Python bools
            for key, value in d.items():
                if isinstance(value, (np.bool_, np.integer, np.floating)):
                    d[key] = value.item()
            analysis_dicts.append(d)
        json.dump(analysis_dicts, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
