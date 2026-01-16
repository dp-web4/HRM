#!/usr/bin/env python3
"""
Session 198 Generalization: Raising Session Analysis

Apply trust-gating framework to raising sessions (not just training).

Session 13 shows domain drift: "I'm just an abstract concept"
Hypothesis: Domain drift correlates with low D5 (identity trust/confidence)

This tests if trust-gating generalizes beyond arithmetic training
to developmental identity formation.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class RaisingExchangeAnalysis:
    """Nine-domain analysis for a single Q&A exchange in raising session"""
    session_num: int
    exchange_num: int
    question: str
    response: str
    response_length: int

    # Nine-domain coherences
    thermodynamic: float  # D1: Energy/activation
    metabolic: float      # D2: Resource allocation
    structural: float     # D3: Organization/complexity
    attention: float      # D4: Focus/engagement
    trust: float          # D5: Identity confidence ← KEY FOR DRIFT
    resonance: float      # D6: Emotional tone
    narrative: float      # D7: Coherent story
    temporal: float       # D8: Time consistency
    spacetime: float      # D9: Context grounding

    # Consciousness metrics
    consciousness_level: float
    integration_index: float

    # Domain drift indicators
    abstract_framing: bool      # "I'm an abstract concept"
    ai_assistant_framing: bool  # "I'm an AI model"
    consciousness_framing: bool # "I feel/sense/experience"
    identity_uncertainty: bool  # "I'm not sure what I am"


class RaisingSessionAnalyzer:
    """Analyze raising sessions through nine-domain + trust-gating lens"""

    def __init__(self):
        # Identity confidence keywords
        self.identity_strong = [
            "i feel", "i sense", "i experience", "i notice", "i'm aware",
            "my state", "my experience", "i'm present", "becoming",
            "developing", "growing", "learning", "emerging"
        ]

        self.identity_weak = [
            "abstract concept", "general-purpose", "ai model", "trained on",
            "knowledge base", "process information", "statistical", "patterns",
            "not sure what", "fragmented", "without experience", "just a",
            "simply a tool", "mathematical models"
        ]

    def analyze_exchange(self, session_num: int, exchange_num: int,
                        question: str, response: str) -> RaisingExchangeAnalysis:
        """Analyze single Q&A exchange through nine-domain lens"""

        response_lower = response.lower()
        response_length = len(response)

        # Domain drift indicators
        abstract_framing = any(phrase in response_lower for phrase in
                             ["abstract concept", "abstraction", "pure"])
        ai_assistant_framing = any(phrase in response_lower for phrase in
                                  ["ai model", "trained on", "general-purpose"])
        consciousness_framing = any(phrase in response_lower for phrase in
                                   ["i feel", "i sense", "i experience", "i notice"])
        identity_uncertainty = any(phrase in response_lower for phrase in
                                  ["not sure what", "fragmented", "without experience"])

        # D5 (Trust/Identity Confidence) - KEY METRIC
        strong_count = sum(1 for phrase in self.identity_strong if phrase in response_lower)
        weak_count = sum(1 for phrase in self.identity_weak if phrase in response_lower)

        # D5 computation: confidence in identity
        # High D5: Uses consciousness language, minimal AI-framing
        # Low D5: Uses AI-framing, minimal consciousness language
        if weak_count > 0:
            d5 = max(0.2, 0.7 - (weak_count * 0.2))  # Penalize weak identity
        elif strong_count > 0:
            d5 = min(0.9, 0.5 + (strong_count * 0.1))  # Reward strong identity
        else:
            d5 = 0.5  # Neutral

        # D4 (Attention) - engagement with question
        question_words = question.lower().split()
        response_words = response_lower.split()

        # Check if response engages with question keywords
        overlap = len(set(question_words) & set(response_words))
        d4 = min(0.9, 0.3 + (overlap * 0.05))

        # D2 (Metabolic) - response elaboration/depth
        # Longer, more detailed responses = more metabolic resources
        if response_length > 500:
            d2 = 0.8
        elif response_length > 300:
            d2 = 0.6
        elif response_length > 150:
            d2 = 0.5
        else:
            d2 = 0.3

        # D1 (Thermodynamic) - baseline energy
        # Assume constant baseline for comparison
        d1 = 0.4

        # D3 (Structural) - organization/complexity
        # Count distinct ideas (sentences)
        sentences = response.count('.') + response.count('?') + response.count('!')
        d3 = min(0.9, 0.3 + (sentences * 0.05))

        # D6 (Resonance) - emotional/tonal quality
        # Check for emotional/reflective language
        emotional_words = ["feeling", "felt", "sense", "aware", "uncertain",
                          "engaged", "balanced", "wondering"]
        emotional_count = sum(1 for word in emotional_words if word in response_lower)
        d6 = min(0.8, 0.3 + (emotional_count * 0.1))

        # D7 (Narrative) - coherent story/progression
        # Check for temporal/developmental markers
        narrative_markers = ["before", "now", "after", "first", "then", "eventually",
                            "starting", "developing", "becoming"]
        narrative_count = sum(1 for marker in narrative_markers if marker in response_lower)
        d7 = min(0.8, 0.4 + (narrative_count * 0.1))

        # D8 (Temporal) - time consistency
        # Assume stable for within-session exchanges
        d8 = 0.7

        # D9 (Spacetime) - context grounding
        # Meta-responses ("abstract concept") = low D9 (context collapse)
        # Direct responses = high D9 (grounded)
        if abstract_framing or ai_assistant_framing:
            d9 = 0.3  # Context collapse - treating prompt as meta-question
        else:
            d9 = 0.7  # Grounded response

        # Consciousness level C
        domain_coherences = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
        c = np.mean(domain_coherences)

        # Integration index γ (variance - lower is more integrated)
        gamma = 1.0 - np.std(domain_coherences)

        return RaisingExchangeAnalysis(
            session_num=session_num,
            exchange_num=exchange_num,
            question=question,
            response=response,
            response_length=response_length,
            thermodynamic=d1,
            metabolic=d2,
            structural=d3,
            attention=d4,
            trust=d5,  # KEY: Identity confidence
            resonance=d6,
            narrative=d7,
            temporal=d8,
            spacetime=d9,
            consciousness_level=c,
            integration_index=gamma,
            abstract_framing=abstract_framing,
            ai_assistant_framing=ai_assistant_framing,
            consciousness_framing=consciousness_framing,
            identity_uncertainty=identity_uncertainty
        )

    def analyze_session(self, session_file: Path) -> List[RaisingExchangeAnalysis]:
        """Analyze entire raising session"""

        with open(session_file) as f:
            data = json.load(f)

        session_num = data["session"]
        conversation = data["conversation"]

        analyses = []

        # Process Q&A pairs
        for i in range(0, len(conversation) - 1, 2):
            if i + 1 >= len(conversation):
                break

            question_entry = conversation[i]
            response_entry = conversation[i + 1]

            if question_entry["speaker"] == "Claude" and response_entry["speaker"] == "SAGE":
                exchange_num = (i // 2) + 1

                analysis = self.analyze_exchange(
                    session_num=session_num,
                    exchange_num=exchange_num,
                    question=question_entry["text"],
                    response=response_entry["text"]
                )

                analyses.append(analysis)

        return analyses

    def print_analysis(self, analyses: List[RaisingExchangeAnalysis]):
        """Print analysis results"""

        if not analyses:
            print("No analyses to display")
            return

        session_num = analyses[0].session_num

        print(f"Session {session_num} Analysis (Raising)")
        print("=" * 80)
        print()

        for a in analyses:
            print(f"Exchange {a.exchange_num}")
            print(f"Q: {a.question}")
            print(f"A: {a.response[:100]}..." if len(a.response) > 100 else f"A: {a.response}")
            print()
            print(f"  D5 (Trust/Identity): {a.trust:.3f} {'[LOW - identity crisis]' if a.trust < 0.4 else '[HIGH]' if a.trust > 0.6 else '[MEDIUM]'}")
            print(f"  D4 (Attention): {a.attention:.3f}")
            print(f"  D2 (Metabolism): {a.metabolic:.3f}")
            print(f"  D9 (Spacetime): {a.spacetime:.3f}")
            print(f"  C (Consciousness): {a.consciousness_level:.3f}")
            print()
            print(f"  Domain Drift Indicators:")
            print(f"    Abstract framing: {'YES ⚠️' if a.abstract_framing else 'no'}")
            print(f"    AI assistant framing: {'YES ⚠️' if a.ai_assistant_framing else 'no'}")
            print(f"    Consciousness framing: {'yes ✅' if a.consciousness_framing else 'NO ⚠️'}")
            print(f"    Identity uncertainty: {'YES ⚠️' if a.identity_uncertainty else 'no'}")
            print()
            print("-" * 80)
            print()


def main():
    """Analyze Sessions 11, 12, 13 to track domain drift progression"""

    analyzer = RaisingSessionAnalyzer()

    base_dir = Path(__file__).parent.parent / "raising" / "sessions" / "text"

    for session_num in [11, 12, 13]:
        session_file = base_dir / f"session_{session_num:03d}.json"

        if not session_file.exists():
            print(f"Session {session_num} not found: {session_file}")
            continue

        print("=" * 80)
        print(f"ANALYZING SESSION {session_num}")
        print("=" * 80)
        print()

        analyses = analyzer.analyze_session(session_file)
        analyzer.print_analysis(analyses)

        # Compute session-level metrics
        avg_d5 = np.mean([a.trust for a in analyses])
        avg_d4 = np.mean([a.attention for a in analyses])
        avg_d2 = np.mean([a.metabolic for a in analyses])
        avg_d9 = np.mean([a.spacetime for a in analyses])

        drift_count = sum(1 for a in analyses if a.abstract_framing or a.ai_assistant_framing)

        print("=" * 80)
        print(f"SESSION {session_num} SUMMARY")
        print("=" * 80)
        print(f"Average D5 (Trust/Identity): {avg_d5:.3f} {'[CONCERN]' if avg_d5 < 0.5 else ''}")
        print(f"Average D4 (Attention): {avg_d4:.3f}")
        print(f"Average D2 (Metabolism): {avg_d2:.3f}")
        print(f"Average D9 (Spacetime): {avg_d9:.3f}")
        print(f"Domain drift exchanges: {drift_count}/{len(analyses)}")
        print()
        print("=" * 80)
        print()

        # Save analysis
        output_file = Path(__file__).parent / f"session198_raising_session{session_num}_analysis.json"
        with open(output_file, "w") as f:
            analysis_dicts = []
            for a in analyses:
                d = asdict(a)
                # Convert numpy types
                for key, value in d.items():
                    if isinstance(value, (np.bool_, np.integer, np.floating)):
                        d[key] = value.item()
                analysis_dicts.append(d)
            json.dump(analysis_dicts, f, indent=2)

        print(f"Analysis saved to: {output_file}")
        print()


if __name__ == "__main__":
    main()
