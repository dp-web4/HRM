#!/usr/bin/env python3
"""
Session 145: EP Pattern Organization by Domain
==============================================

Organizes 100 patterns from Session 144 by which EP domain dominated the
decision, creating domain-specific pattern libraries for mature EP predictions.

Context:
- Session 144: Collected 100 patterns across 20 scenario types
- Current: Patterns tagged by scenario_type
- Goal: Organize by primary EP domain for pattern matching

Approach:
1. Read session144_ep_pattern_corpus.json
2. Extract primary domain from coordinated_decision.reasoning
3. Group patterns by domain
4. Create domain-specific pattern files
5. Analyze domain coverage and characteristics

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2025-12-31
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter

# ============================================================================
# Pattern Organization
# ============================================================================

class EPPatternOrganizer:
    """Organizes patterns by primary EP domain."""

    def __init__(self, pattern_corpus_path: Path):
        self.corpus_path = pattern_corpus_path
        self.patterns = []
        self.patterns_by_domain = defaultdict(list)
        self.domain_stats = {}

    def load_patterns(self) -> int:
        """Load pattern corpus from JSON file."""
        with open(self.corpus_path, 'r') as f:
            data = json.load(f)
            self.patterns = data.get("patterns", [])
        return len(self.patterns)

    def extract_primary_domain(self, pattern: Dict[str, Any]) -> str:
        """
        Extract which EP domain dominated the decision.

        Analyzes coordinated_decision.reasoning to determine primary domain.
        """
        reasoning = pattern.get("coordinated_decision", {}).get("reasoning", "")

        # Check for explicit domain mentions in reasoning
        if "emotional EP" in reasoning.lower():
            return "emotional"
        elif "quality EP" in reasoning.lower():
            return "quality"
        elif "attention EP" in reasoning.lower():
            return "attention"
        elif "grounding EP" in reasoning.lower():
            return "grounding"
        elif "authorization EP" in reasoning.lower():
            return "authorization"
        elif "all 5 EPs agree" in reasoning.lower() or "all EPs agree" in reasoning.lower():
            # Consensus - could be any domain, use scenario type as hint
            return self._infer_from_scenario(pattern)
        else:
            # Fallback: infer from scenario type
            return self._infer_from_scenario(pattern)

    def _infer_from_scenario(self, pattern: Dict[str, Any]) -> str:
        """Infer primary domain from scenario type when reasoning unclear."""
        scenario_type = pattern.get("scenario_type", "")

        if "emotional" in scenario_type:
            return "emotional"
        elif "quality" in scenario_type:
            return "quality"
        elif "attention" in scenario_type:
            return "attention"
        elif "grounding" in scenario_type:
            return "grounding"
        elif "authorization" in scenario_type:
            return "authorization"
        elif "cascade" in scenario_type:
            # Cascade scenarios could be any domain, check predictions
            return self._infer_from_predictions(pattern)
        elif "benign" in scenario_type:
            # Benign scenarios - check predictions
            return self._infer_from_predictions(pattern)
        else:
            return "unknown"

    def _infer_from_predictions(self, pattern: Dict[str, Any]) -> str:
        """Infer from EP predictions which had highest severity."""
        predictions = pattern.get("ep_predictions", {})

        # Find domain with highest severity
        max_severity = 0.0
        max_domain = "unknown"

        for domain, pred in predictions.items():
            severity = pred.get("severity", 0.0)
            if severity > max_severity:
                max_severity = severity
                max_domain = domain

        return max_domain

    def organize_patterns(self) -> Dict[str, int]:
        """
        Organize all patterns by primary domain.

        Returns count by domain.
        """
        for pattern in self.patterns:
            primary_domain = self.extract_primary_domain(pattern)
            self.patterns_by_domain[primary_domain].append(pattern)

        return {domain: len(patterns) for domain, patterns in self.patterns_by_domain.items()}

    def analyze_domain_patterns(self) -> Dict[str, Any]:
        """Analyze characteristics of patterns in each domain."""
        analysis = {}

        for domain, patterns in self.patterns_by_domain.items():
            if not patterns:
                continue

            # Decision distribution
            decisions = [p["coordinated_decision"]["final_decision"] for p in patterns]
            decision_counts = Counter(decisions)

            # Outcome distribution
            outcomes = [p["outcome"]["outcome_type"] for p in patterns]
            outcome_counts = Counter(outcomes)

            # Success rate
            successes = sum(1 for p in patterns if p["outcome"]["success"])
            success_rate = successes / len(patterns) if patterns else 0.0

            # Cascade detections
            cascades = sum(1 for p in patterns if p["coordinated_decision"].get("cascade_predicted", False))

            # Average severity per domain prediction
            domain_severities = []
            for p in patterns:
                pred = p.get("ep_predictions", {}).get(domain, {})
                severity = pred.get("severity", 0.0)
                if severity > 0:
                    domain_severities.append(severity)

            avg_severity = sum(domain_severities) / len(domain_severities) if domain_severities else 0.0

            # Scenario types in this domain
            scenario_types = Counter(p.get("scenario_type", "unknown") for p in patterns)

            analysis[domain] = {
                "count": len(patterns),
                "decisions": dict(decision_counts),
                "outcomes": dict(outcome_counts),
                "success_rate": success_rate,
                "cascade_detections": cascades,
                "avg_severity": avg_severity,
                "scenario_types": dict(scenario_types.most_common(5))  # Top 5
            }

        return analysis

    def save_domain_patterns(self, output_dir: Path):
        """Save patterns for each domain to separate JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for domain, patterns in self.patterns_by_domain.items():
            if not patterns:
                continue

            output_file = output_dir / f"{domain}_ep_patterns.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "domain": domain,
                    "pattern_count": len(patterns),
                    "source": "Session 144 pattern corpus",
                    "organized": "Session 145",
                    "patterns": patterns
                }, f, indent=2)

        return len(self.patterns_by_domain)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary of pattern organization."""
        return {
            "session": 145,
            "timestamp": "2025-12-31",
            "source": "session144_ep_pattern_corpus.json",
            "total_patterns": len(self.patterns),
            "patterns_by_domain": {
                domain: len(patterns)
                for domain, patterns in self.patterns_by_domain.items()
            },
            "domain_analysis": self.analyze_domain_patterns(),
            "organization_method": "primary_domain_extraction",
            "extraction_strategy": [
                "1. Check coordinated_decision.reasoning for explicit domain mention",
                "2. Fallback to scenario_type inference",
                "3. Final fallback to highest severity prediction"
            ]
        }


# ============================================================================
# Main: Pattern Organization
# ============================================================================

def main():
    """Run pattern organization by domain."""
    print("=" * 80)
    print("Session 145: EP Pattern Organization by Domain")
    print("=" * 80)
    print()
    print("Goal: Organize 100 patterns from Session 144 by primary EP domain")
    print("Method: Extract domain from reasoning, scenario type, and predictions")
    print("Output: Domain-specific pattern files for mature EP")
    print()

    # Paths
    corpus_path = Path(__file__).parent / "session144_ep_pattern_corpus.json"
    output_dir = Path(__file__).parent / "ep_patterns_by_domain"

    # Create organizer
    organizer = EPPatternOrganizer(corpus_path)

    # Load patterns
    print(f"Loading patterns from: {corpus_path.name}")
    pattern_count = organizer.load_patterns()
    print(f"Loaded {pattern_count} patterns")
    print()

    # Organize by domain
    print("Organizing patterns by primary EP domain...")
    domain_counts = organizer.organize_patterns()
    print()
    print("Patterns by Domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / pattern_count * 100 if pattern_count > 0 else 0
        print(f"  {domain:15s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Analyze patterns
    print("Analyzing domain pattern characteristics...")
    analysis = organizer.analyze_domain_patterns()
    print()
    print("Domain Analysis:")
    for domain in sorted(analysis.keys()):
        stats = analysis[domain]
        print(f"\n{domain.upper()} EP ({stats['count']} patterns):")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Avg Severity: {stats['avg_severity']:.3f}")
        print(f"  Cascades: {stats['cascade_detections']}")
        print(f"  Decisions: {stats['decisions']}")
        print(f"  Top Scenarios: {list(stats['scenario_types'].keys())[:3]}")

    # Save domain-specific pattern files
    print()
    print(f"Saving domain-specific pattern files to: {output_dir.name}/")
    files_created = organizer.save_domain_patterns(output_dir)
    print(f"Created {files_created} domain pattern files")

    # Save summary report
    summary = organizer.generate_summary_report()
    summary_path = Path(__file__).parent / "session145_pattern_organization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Summary saved to: {summary_path.name}")
    print()
    print("=" * 80)
    print("Session 145 Complete")
    print("=" * 80)
    print()
    print("Pattern organization enables domain-specific pattern matching for")
    print("high-confidence mature EP predictions.")
    print()
    print(f"Next step: Implement pattern matching with ~{pattern_count // 5} patterns per domain")

    return summary


if __name__ == "__main__":
    summary = main()
