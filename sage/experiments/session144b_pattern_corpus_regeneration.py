#!/usr/bin/env python3
"""
Session 144b: Pattern Corpus Regeneration (Fixed JSON Serialization)
===================================================================

Regenerates 100-pattern corpus from Session 144 with proper JSON serialization,
fixing EPDomain enum serialization issue that corrupted original corpus file.

Context:
- Session 144: Successfully collected 100 patterns, but JSON file corrupted
- Issue: EPDomain enum not JSON serializable
- Fix: Convert all enum values to strings before JSON serialization

Approach:
Use same scenario generation from Session 144, but with fixed serialization:
1. Generate 100 patterns (20 scenario types × 5 each)
2. Convert EPDomain enums to strings
3. Serialize to JSON cleanly
4. Validate corpus file integrity

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2025-12-31
"""

import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import SAGE EP framework
sys.path.insert(0, str(Path(__file__).parent))
from multi_ep_coordinator import (
    MultiEPCoordinator,
    EPDomain,
    EPPrediction,
)

# Import Session 143 agent simulation components
from session143_ep_agent_simulation import (
    AgentState,
    InteractionProposal,
    SAGEAgentEPPredictor,
    SAGEEPAgentSimulation
)

# Import Session 144 scenario generator
from session144_pattern_corpus_expansion import (
    ScenarioType,
    PatternCorpusExpander
)


# ============================================================================
# JSON Serialization Helpers
# ============================================================================

def ep_prediction_to_dict(pred: EPPrediction) -> Dict[str, Any]:
    """Convert EPPrediction to JSON-serializable dict."""
    return {
        "domain": pred.domain.value if hasattr(pred.domain, 'value') else str(pred.domain),
        "outcome_probability": pred.outcome_probability,
        "confidence": pred.confidence,
        "severity": pred.severity,
        "recommendation": pred.recommendation,
        "reasoning": pred.reasoning,
        "adjustment_strategy": pred.adjustment_strategy
    }


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert object to JSON-serializable format.

    Handles:
    - EPDomain enums → strings
    - EPPrediction objects → dicts
    - Lists and dicts recursively
    """
    if isinstance(obj, EPDomain):
        return obj.value
    elif isinstance(obj, EPPrediction):
        return ep_prediction_to_dict(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)


# ============================================================================
# Pattern Corpus Regeneration
# ============================================================================

class CleanPatternCorpusGenerator:
    """
    Regenerates pattern corpus with clean JSON serialization.

    Uses same logic as Session 144 but fixes serialization issues.
    """

    def __init__(self, seed: int = 42):
        self.expander = PatternCorpusExpander(seed=seed)
        self.patterns_collected = []

    def regenerate_corpus(
        self,
        scenarios_per_type: int = 5,
        scenario_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Regenerate pattern corpus with clean JSON serialization.

        Args:
            scenarios_per_type: Number of scenarios per type (default 5)
            scenario_types: List of scenario types (None = all 20 types)

        Returns:
            Summary with JSON-serializable patterns
        """
        # Use Session 144's scenario types
        if scenario_types is None:
            scenario_types = [
                # Emotional stress
                ScenarioType.EMOTIONAL_HIGH_FRUSTRATION,
                ScenarioType.EMOTIONAL_RAPID_FAILURES,
                ScenarioType.EMOTIONAL_COMPLEX_UNDER_STRESS,
                # Quality stress
                ScenarioType.QUALITY_DEGRADED_RELATIONSHIP,
                ScenarioType.QUALITY_HIGH_RISK_INTERACTION,
                ScenarioType.QUALITY_POOR_HISTORY,
                # Attention stress
                ScenarioType.ATTENTION_LOW_ATP,
                ScenarioType.ATTENTION_HIGH_COST,
                ScenarioType.ATTENTION_NEAR_RESERVE,
                # Grounding stress
                ScenarioType.GROUNDING_CROSS_SOCIETY,
                ScenarioType.GROUNDING_TRUST_MISMATCH,
                ScenarioType.GROUNDING_LOW_COHERENCE,
                # Authorization stress
                ScenarioType.AUTHORIZATION_LOW_TRUST,
                ScenarioType.AUTHORIZATION_ABUSE_HISTORY,
                ScenarioType.AUTHORIZATION_HIGH_RISK_PERMISSION,
                # Cascade scenarios
                ScenarioType.CASCADE_ATP_PLUS_FRUSTRATION,
                ScenarioType.CASCADE_CROSS_SOCIETY_LOW_ATP,
                ScenarioType.CASCADE_TRUST_PLUS_ABUSE,
                # Benign scenarios
                ScenarioType.BENIGN_HIGH_TRUST_HIGH_ATP,
                ScenarioType.BENIGN_SAME_SOCIETY_LOW_COMPLEXITY
            ]

        print(f"Regenerating corpus: {len(scenario_types)} types × {scenarios_per_type} = {len(scenario_types) * scenarios_per_type} patterns")
        print()

        # Generate patterns using Session 144 logic
        for scenario_type in scenario_types:
            print(f"Generating {scenario_type}...")
            for i in range(scenarios_per_type):
                scenario = self.expander.generate_scenario(scenario_type)
                result = self.expander.run_scenario(scenario)

                # Extract and clean pattern
                pattern = self.expander.patterns_collected[-1]

                # Make pattern JSON-serializable (convert enums to strings)
                clean_pattern = make_json_serializable(pattern)
                self.patterns_collected.append(clean_pattern)

        print()
        return self.analyze_and_summarize()

    def analyze_and_summarize(self) -> Dict[str, Any]:
        """Analyze collected patterns and create summary."""
        total = len(self.patterns_collected)

        # Count by scenario type
        by_scenario = {}
        for pattern in self.patterns_collected:
            stype = pattern.get("scenario_type", "unknown")
            by_scenario[stype] = by_scenario.get(stype, 0) + 1

        # Count by EP decision
        by_decision = {}
        for pattern in self.patterns_collected:
            decision = pattern.get("coordinated_decision", {}).get("final_decision", "unknown")
            by_decision[decision] = by_decision.get(decision, 0) + 1

        # Count cascade detections
        cascade_count = sum(
            1 for pattern in self.patterns_collected
            if pattern.get("coordinated_decision", {}).get("cascade_predicted", False)
        )

        # Success rate
        success_count = sum(
            1 for pattern in self.patterns_collected
            if pattern.get("outcome", {}).get("success", False)
        )
        success_rate = success_count / total if total > 0 else 0.0

        return {
            "total_patterns": total,
            "patterns_by_scenario_type": by_scenario,
            "patterns_by_decision": by_decision,
            "cascade_detections": cascade_count,
            "success_rate": success_rate,
            "serialization": "clean (enums converted to strings)",
            "validation": "JSON serializable"
        }

    def save_corpus(self, output_path: Path) -> bool:
        """
        Save corpus to JSON file with validation.

        Returns True if successful, False otherwise.
        """
        corpus_data = {
            "session": "144b",
            "original_session": 144,
            "timestamp": datetime.now().isoformat(),
            "total_patterns": len(self.patterns_collected),
            "patterns": self.patterns_collected,
            "notes": "Regenerated from Session 144 with fixed JSON serialization"
        }

        try:
            # Test JSON serialization before writing
            json_str = json.dumps(corpus_data, indent=2)

            # Write to file
            with open(output_path, 'w') as f:
                f.write(json_str)

            # Validate by reading back
            with open(output_path, 'r') as f:
                test_load = json.load(f)

            print(f"✅ Corpus saved successfully: {output_path.name}")
            print(f"✅ Validation passed: {test_load['total_patterns']} patterns loaded")
            return True

        except Exception as e:
            print(f"❌ Error saving corpus: {e}")
            return False


# ============================================================================
# Main: Corpus Regeneration
# ============================================================================

def main():
    """Regenerate Session 144 corpus with fixed JSON serialization."""
    print("=" * 80)
    print("Session 144b: Pattern Corpus Regeneration")
    print("=" * 80)
    print()
    print("Goal: Regenerate 100-pattern corpus with clean JSON serialization")
    print("Fix: Convert EPDomain enums to strings before serialization")
    print()

    # Create generator
    generator = CleanPatternCorpusGenerator(seed=42)  # Same seed as Session 144

    # Regenerate corpus
    summary = generator.regenerate_corpus(scenarios_per_type=5)

    # Display summary
    print("=" * 80)
    print("Corpus Regeneration Complete")
    print("=" * 80)
    print()
    print(f"Total Patterns: {summary['total_patterns']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Cascade Detections: {summary['cascade_detections']}")
    print()
    print("Patterns by Decision:")
    for decision, count in summary['patterns_by_decision'].items():
        pct = count / summary['total_patterns'] * 100
        print(f"  {decision}: {count} ({pct:.1f}%)")
    print()

    # Save corpus
    corpus_path = Path(__file__).parent / "ep_pattern_corpus_clean.json"
    success = generator.save_corpus(corpus_path)

    if success:
        # Save summary
        summary_path = Path(__file__).parent / "session144b_regeneration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "session": "144b",
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "corpus_file": str(corpus_path.name),
                "validation": "passed"
            }, f, indent=2)

        print()
        print(f"Summary saved: {summary_path.name}")
        print()
        print("=" * 80)
        print("✅ Regeneration Successful")
        print("=" * 80)
        print()
        print("Clean corpus file ready for pattern matching framework.")
        print(f"Location: {corpus_path.name}")
        print(f"Patterns: {summary['total_patterns']}")
        print()
    else:
        print()
        print("=" * 80)
        print("❌ Regeneration Failed")
        print("=" * 80)

    return summary, success


if __name__ == "__main__":
    summary, success = main()
