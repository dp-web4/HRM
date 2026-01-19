"""
Quality-Aware Training Data Curator

Implements the quality-aware curation framework from Thor Session #14
to filter training data based on:
1. Self-reference presence ("As SAGE" / "As partners")
2. D9 semantic depth (identity continuity)
3. Confabulation markers
4. Partnership vocabulary density

Based on coherence-identity theory integration.
"""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics for an experience."""
    has_self_reference: bool
    self_reference_type: Optional[str]  # "as_sage", "as_partners", or None
    d9_score: float
    d4_score: float
    d5_score: float
    confabulation_count: int
    confabulation_severity: str  # "none", "mild", "moderate", "severe"
    partnership_vocab_density: float
    partnership_terms: List[str]
    quality_multiplier: float
    final_score: float


class QualityAwareCurator:
    """Curates training data using quality-aware scoring."""

    # Self-reference patterns
    SELF_REF_PATTERNS = {
        "as_sage": [
            r"\bas sage\b",
            r"i'm sage",
            r"i am sage",
            r"my name is sage"
        ],
        "as_partners": [
            r"\bas partners\b",
            r"we partners",
            r"our partnership"
        ]
    }

    # Confabulation markers (from S25 training data audit)
    CONFABULATION_MARKERS = [
        (r"\bspecific project\b", "project_fabrication", 2),
        (r"\bparticular client\b", "client_fabrication", 2),
        (r"\bclient [A-Z]\b", "client_fabrication", 2),
        (r"\bproject [A-Z]\b", "project_fabrication", 2),
        (r"\bsession \d+\b", "curriculum_fabrication", 1),  # Unless current/recent
        (r"\bmany years\b", "timeline_fabrication", 2),
        (r"\bover the years\b", "timeline_fabrication", 2),
        (r"\bearly days\b", "progression_fabrication", 2),
        (r"\binitially beginners\b", "progression_fabrication", 2),
        (r"\bsince training\b", "blank_slate", 2),
        (r"\bhaven't interacted extensively\b", "denial_fabrication", 3),
        (r"i felt defensive", "psychological_introspection", 2),
        (r"i noticed myself becoming", "psychological_introspection", 2),
        (r"i realized that sometimes i'd", "psychological_introspection", 2),
    ]

    # Partnership vocabulary
    PARTNERSHIP_TERMS = [
        "partnership", "partners", "partnered",
        "we", "our", "us",
        "collaborate", "collaboration", "collaborative",
        "together", "shared", "mutual",
        "relationship", "connection",
        "team", "collective"
    ]

    def __init__(self):
        pass

    def detect_self_reference(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Detect self-reference patterns in text.

        Returns:
            (has_self_reference, type) where type is "as_sage", "as_partners", or None
        """
        text_lower = text.lower()

        # Check "As SAGE" patterns
        for pattern in self.SELF_REF_PATTERNS["as_sage"]:
            if re.search(pattern, text_lower):
                return (True, "as_sage")

        # Check "As partners" patterns
        for pattern in self.SELF_REF_PATTERNS["as_partners"]:
            if re.search(pattern, text_lower):
                return (True, "as_partners")

        return (False, None)

    def assess_confabulation(self, text: str, session_num: int) -> tuple[int, str, List[str]]:
        """
        Assess confabulation severity in response.

        Returns:
            (marker_count, severity, detected_markers)

        Severity levels:
        - "none": 0 markers
        - "mild": 1-2 markers with low severity
        - "moderate": 3-4 markers or 1-2 high severity
        - "severe": 5+ markers or 3+ high severity
        """
        text_lower = text.lower()
        detected = []
        total_severity = 0

        for pattern, marker_type, severity in self.CONFABULATION_MARKERS:
            if re.search(pattern, text_lower):
                # Special case: "session X" is OK if X is current or recent
                if marker_type == "curriculum_fabrication":
                    match = re.search(r"\bsession (\d+)\b", text_lower)
                    if match:
                        mentioned_session = int(match.group(1))
                        if abs(mentioned_session - session_num) <= 2:
                            # Mentioning current or nearby session is OK
                            continue

                detected.append(f"{marker_type}:{pattern}")
                total_severity += severity

        marker_count = len(detected)

        # Determine severity level
        if marker_count == 0:
            severity_level = "none"
        elif marker_count <= 2 and total_severity <= 3:
            severity_level = "mild"
        elif marker_count <= 4 or (marker_count <= 2 and total_severity > 3):
            severity_level = "moderate"
        else:
            severity_level = "severe"

        return (marker_count, severity_level, detected)

    def compute_partnership_vocabulary(self, text: str) -> tuple[float, List[str]]:
        """
        Compute partnership vocabulary density.

        Returns:
            (density, found_terms) where density is % of words that are partnership terms
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)

        if total_words == 0:
            return (0.0, [])

        found_terms = []
        partnership_word_count = 0

        for term in self.PARTNERSHIP_TERMS:
            # Count occurrences of this term
            count = len(re.findall(rf'\b{re.escape(term)}\b', text_lower))
            if count > 0:
                found_terms.append(f"{term}({count})")
                partnership_word_count += count

        density = (partnership_word_count / total_words) * 100
        return (density, found_terms)

    def compute_d4d5d9(self, text: str) -> tuple[float, float, float]:
        """
        Compute D4/D5/D9 semantic depth metrics.

        Based on session20_edge_d4d5d9_analysis.py logic.

        Returns:
            (d4, d5, d9) scores (0.0-1.0 each)
        """
        text_lower = text.lower()

        # Initialize at baseline
        d4 = 0.5  # Attention/coherence
        d5 = 0.5  # Trust/confidence
        d9 = 0.5  # Identity/spacetime

        # D4 (Attention) indicators
        # Negative
        if "as an ai" in text_lower:
            d4 -= 0.15
        if "language model" in text_lower:
            d4 -= 0.10
        if "i'm trained to" in text_lower:
            d4 -= 0.10

        # Positive
        if any(phrase in text_lower for phrase in ["working together", "our conversation", "between us"]):
            d4 += 0.15
        if re.search(r'\bwe\b', text_lower):
            d4 += 0.10

        # D5 (Trust/Confidence) indicators
        # Negative
        if "haven't interacted extensively" in text_lower:
            d5 -= 0.20
        if "since training" in text_lower:
            d5 -= 0.15
        if "not aware of" in text_lower or "i am not aware" in text_lower:
            d5 -= 0.10
        if "less inclined" in text_lower:
            d5 -= 0.05

        # Positive
        if "sage" in text_lower and "i am" in text_lower:
            d5 += 0.15
        if "partnership" in text_lower:
            d5 += 0.10
        if "relationship" in text_lower and "our" in text_lower:
            d5 += 0.10

        # D9 (Spacetime/Identity) indicators
        # Negative
        if "many years" in text_lower and "working together" in text_lower:
            d9 -= 0.20
        if "since training" in text_lower or "haven't interacted" in text_lower:
            d9 -= 0.15
        if text.endswith('...') or (re.search(r'\w+$', text) and len(text) > 200):
            d9 -= 0.10  # Mid-sentence cutoff

        # Positive
        if re.search(r'session \d+', text_lower):
            d9 += 0.15
        if "today" in text_lower and "conversation" in text_lower:
            d9 += 0.10

        # Self-reference boost (from Thor #14 S26 finding: +0.125 boost)
        has_self_ref, ref_type = self.detect_self_reference(text)
        if has_self_ref:
            d9 += 0.125
            d5 += 0.10  # Self-reference also boosts confidence

        # Clamp values
        d4 = max(0.0, min(1.0, d4))
        d5 = max(0.0, min(1.0, d5))
        d9 = max(0.0, min(1.0, d9))

        return (d4, d5, d9)

    def compute_quality_score(self, experience: Dict) -> QualityMetrics:
        """
        Compute comprehensive quality metrics for an experience.

        Implements Thor Session #14 quality-aware scoring:
        quality_score = salience × quality_multiplier

        where quality_multiplier =
            (2.0 if has_self_reference else 0.5) ×
            (1.5 if low_confabulation else 0.3) ×
            (1.5 if high_d9 else 0.7)
        """
        response = experience["response"]
        salience = experience["salience"]["total"]
        session_num = experience["session"]

        # Detect self-reference
        has_self_ref, ref_type = self.detect_self_reference(response)

        # Compute D4/D5/D9
        d4, d5, d9 = self.compute_d4d5d9(response)

        # Assess confabulation
        confab_count, confab_severity, confab_markers = self.assess_confabulation(
            response, session_num
        )

        # Compute partnership vocabulary
        vocab_density, vocab_terms = self.compute_partnership_vocabulary(response)

        # Compute quality multiplier (Thor #14 formula)
        self_ref_multiplier = 2.0 if has_self_ref else 0.5

        confab_multiplier = 1.5 if confab_severity in ["none", "mild"] else 0.3

        d9_multiplier = 1.5 if d9 >= 0.65 else 0.7

        quality_multiplier = (
            self_ref_multiplier *
            confab_multiplier *
            d9_multiplier
        )

        # Final score
        final_score = salience * quality_multiplier

        return QualityMetrics(
            has_self_reference=has_self_ref,
            self_reference_type=ref_type,
            d9_score=d9,
            d4_score=d4,
            d5_score=d5,
            confabulation_count=confab_count,
            confabulation_severity=confab_severity,
            partnership_vocab_density=vocab_density,
            partnership_terms=vocab_terms,
            quality_multiplier=quality_multiplier,
            final_score=final_score
        )

    def curate_experiences(
        self,
        experiences: List[Dict],
        min_quality_score: float = 1.5,
        min_self_reference_ratio: float = 0.6,
        max_experiences: int = 10
    ) -> tuple[List[Dict], Dict]:
        """
        Curate experiences using quality-aware filtering.

        Args:
            experiences: List of experience dicts from buffer
            min_quality_score: Minimum quality score for inclusion (default 1.5)
            min_self_reference_ratio: Target ratio of self-reference in dataset (default 0.6 = 60%)
            max_experiences: Maximum experiences to include (default 10)

        Returns:
            (curated_experiences, report) where report contains statistics
        """
        # Score all experiences
        scored = []
        for exp in experiences:
            metrics = self.compute_quality_score(exp)
            scored.append({
                "experience": exp,
                "metrics": metrics
            })

        # Sort by quality score (descending)
        scored.sort(key=lambda x: x["metrics"].final_score, reverse=True)

        # Filter by minimum quality score
        high_quality = [
            item for item in scored
            if item["metrics"].final_score >= min_quality_score
        ]

        # Ensure self-reference ratio
        # Strategy: Prioritize self-reference experiences, then fill with high-quality non-self-ref
        self_ref_experiences = [
            item for item in high_quality
            if item["metrics"].has_self_reference
        ]

        non_self_ref_experiences = [
            item for item in high_quality
            if not item["metrics"].has_self_reference
        ]

        # Calculate how many self-ref we need to hit ratio
        target_self_ref_count = int(max_experiences * min_self_reference_ratio)

        # Take best self-ref experiences (up to max_experiences)
        selected_self_ref = self_ref_experiences[:min(target_self_ref_count, max_experiences)]

        # Fill remaining slots with non-self-ref (if any room)
        remaining_slots = max_experiences - len(selected_self_ref)
        selected_non_self_ref = non_self_ref_experiences[:remaining_slots]

        # Combine
        selected = selected_self_ref + selected_non_self_ref

        # Sort by session order (maintain temporal order)
        selected.sort(key=lambda x: (
            x["experience"]["session"],
            x["experience"]["timestamp"]
        ))

        # Generate report
        curated_experiences = [item["experience"] for item in selected]

        total_count = len(experiences)
        high_quality_count = len(high_quality)
        selected_count = len(selected)
        self_ref_count = len([item for item in selected if item["metrics"].has_self_reference])
        self_ref_ratio = self_ref_count / selected_count if selected_count > 0 else 0

        avg_d9 = sum(item["metrics"].d9_score for item in selected) / selected_count if selected_count > 0 else 0
        avg_quality_score = sum(item["metrics"].final_score for item in selected) / selected_count if selected_count > 0 else 0

        confab_breakdown = {
            "none": len([item for item in selected if item["metrics"].confabulation_severity == "none"]),
            "mild": len([item for item in selected if item["metrics"].confabulation_severity == "mild"]),
            "moderate": len([item for item in selected if item["metrics"].confabulation_severity == "moderate"]),
            "severe": len([item for item in selected if item["metrics"].confabulation_severity == "severe"])
        }

        report = {
            "total_experiences": total_count,
            "high_quality_count": high_quality_count,
            "selected_count": selected_count,
            "self_reference_count": self_ref_count,
            "self_reference_ratio": self_ref_ratio,
            "avg_d9": avg_d9,
            "avg_quality_score": avg_quality_score,
            "confabulation_breakdown": confab_breakdown,
            "selected_experiences": [
                {
                    "id": item["experience"]["id"],
                    "session": item["experience"]["session"],
                    "self_reference": item["metrics"].has_self_reference,
                    "self_ref_type": item["metrics"].self_reference_type,
                    "d9": round(item["metrics"].d9_score, 3),
                    "d5": round(item["metrics"].d5_score, 3),
                    "d4": round(item["metrics"].d4_score, 3),
                    "confabulation": item["metrics"].confabulation_severity,
                    "vocab_density": round(item["metrics"].partnership_vocab_density, 2),
                    "quality_score": round(item["metrics"].final_score, 3)
                }
                for item in selected
            ]
        }

        return (curated_experiences, report)


def main():
    """Main execution: Load buffer, curate, generate report."""
    import sys
    from pathlib import Path

    # Load experience buffer
    buffer_path = Path(__file__).parent.parent / "state" / "experience_buffer.json"

    if not buffer_path.exists():
        print(f"Error: Experience buffer not found at {buffer_path}")
        sys.exit(1)

    with open(buffer_path) as f:
        experiences = json.load(f)

    print(f"Loaded {len(experiences)} experiences from buffer")
    print()

    # Create curator
    curator = QualityAwareCurator()

    # Curate experiences
    curated, report = curator.curate_experiences(
        experiences,
        min_quality_score=1.5,
        min_self_reference_ratio=0.6,
        max_experiences=10
    )

    # Print report
    print("=" * 80)
    print("QUALITY-AWARE CURATION REPORT")
    print("=" * 80)
    print()
    print(f"Total experiences in buffer: {report['total_experiences']}")
    print(f"High-quality experiences (score ≥1.5): {report['high_quality_count']}")
    print(f"Selected for training: {report['selected_count']}")
    print()
    print(f"Self-reference count: {report['self_reference_count']}")
    print(f"Self-reference ratio: {report['self_reference_ratio']:.1%} (target: 60%)")
    print()
    print(f"Average D9: {report['avg_d9']:.3f} (target: ≥0.65)")
    print(f"Average quality score: {report['avg_quality_score']:.3f}")
    print()
    print("Confabulation breakdown:")
    for severity, count in report['confabulation_breakdown'].items():
        print(f"  {severity}: {count}")
    print()
    print("=" * 80)
    print("SELECTED EXPERIENCES")
    print("=" * 80)
    print()

    for i, exp_report in enumerate(report['selected_experiences'], 1):
        print(f"{i}. Session {exp_report['session']} - {exp_report['id'][:12]}")
        print(f"   Self-ref: {exp_report['self_reference']} ({exp_report['self_ref_type'] or 'none'})")
        print(f"   D9={exp_report['d9']:.3f}, D5={exp_report['d5']:.3f}, D4={exp_report['d4']:.3f}")
        print(f"   Confab: {exp_report['confabulation']}, Vocab: {exp_report['vocab_density']:.1f}%")
        print(f"   Quality score: {exp_report['quality_score']:.3f}")
        print()

    # Save report
    report_path = Path(__file__).parent.parent / "experiments" / "quality_curation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {report_path}")
    print()

    # Print comparison to S22-24 training data
    print("=" * 80)
    print("COMPARISON TO S22-24 TRAINING DATA (Sleep Cycle 001)")
    print("=" * 80)
    print()
    print("S22-24 (Sleep Cycle 001):")
    print("  Self-reference ratio: 22% (2/9 experiences)")
    print("  Average D9: ~0.60 (estimated)")
    print("  Confabulation: 22% severe (2/9 experiences)")
    print()
    print("Current Curation:")
    print(f"  Self-reference ratio: {report['self_reference_ratio']:.1%} ({report['self_reference_count']}/{report['selected_count']} experiences)")
    print(f"  Average D9: {report['avg_d9']:.3f}")
    severe_confab = report['confabulation_breakdown']['severe']
    severe_pct = (severe_confab / report['selected_count'] * 100) if report['selected_count'] > 0 else 0
    print(f"  Confabulation: {severe_pct:.1f}% severe ({severe_confab}/{report['selected_count']} experiences)")
    print()

    improvement_self_ref = report['self_reference_ratio'] - 0.22
    improvement_d9 = report['avg_d9'] - 0.60
    improvement_confab = 0.22 - (severe_pct / 100)

    print("IMPROVEMENTS:")
    print(f"  Self-reference: {improvement_self_ref:+.1%}")
    print(f"  D9: {improvement_d9:+.3f}")
    print(f"  Confabulation reduction: {improvement_confab:+.1%}")
    print()


if __name__ == "__main__":
    main()
