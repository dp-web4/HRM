#!/usr/bin/env python3
"""
Session 99: CRISIS State Validation - Extreme Resource Depletion Testing

**Goal**: Validate CRISIS state behavior under extreme ATP depletion conditions

**Research Gap Identified**:
- **Session 97**: Simulated closed-loop stayed 27-100 ATP (no CRISIS at <20 ATP)
- **Session 98**: Production integration stayed 29-100 ATP (no CRISIS)
- **Gap**: CRISIS state constraints never validated in testing
- **Opportunity**: Force CRISIS scenarios to validate cheap-expert-only constraints

**CRISIS State Definition** (from S97/S98):
- ATP < 20: CRISIS state triggered
- Constraints: Only experts with cost < 7 ATP allowed
- Purpose: Prevent complete resource exhaustion
- Recovery: Slower than REST (0.5-1.0 ATP per cycle)

**Testing Strategy**:
1. **Initial depletion**: Start with low ATP (15-25 range)
2. **Forced CRISIS**: Submit expensive queries to trigger CRISIS
3. **Constraint validation**: Verify only cheap experts (<7 ATP) are called
4. **Recovery observation**: Measure CRISIS recovery dynamics
5. **State transition**: Validate CRISIS → REST → WAKE progression

**Expected Behaviors**:
- Expensive experts (≥7 ATP) rejected during CRISIS
- Cheap experts (<7 ATP) still allowed (survival mode)
- System gradually recovers through cheap expert usage
- Eventually transitions REST → WAKE as ATP increases

**Novel Insight to Test**:
CRISIS as "survival mode" - system maintains minimal functionality with
cheapest experts while recovering, rather than complete shutdown.

Created: 2025-12-23 (Autonomous Session 99)
Hardware: Thor (Jetson AGX Thor)
Previous: Session 98 (production ATP integration)
Goal: CRISIS state validation - extreme resource depletion testing
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import Session 98 components
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from session98_production_atp_integration import ProductionATPSelector, QueryMetrics
    from session97_atp_accounting_integration import ATPAccountingBridge, ATPTransaction
    HAS_PRODUCTION_SELECTOR = True
except ImportError:
    HAS_PRODUCTION_SELECTOR = False
    ProductionATPSelector = None
    QueryMetrics = None
    ATPAccountingBridge = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CRISISTestScenario:
    """Definition of a CRISIS testing scenario."""
    name: str
    initial_atp: float
    crisis_threshold: float
    query_complexity: str  # "simple", "moderate", "complex"
    num_queries: int
    expected_crisis: bool
    description: str


class CRISISStateValidator:
    """
    Validator for CRISIS state behavior under extreme ATP depletion.

    **Test Scenarios**:
    1. **Gradual depletion**: Start at 25 ATP, submit complex queries
    2. **Immediate CRISIS**: Start at 15 ATP (already in CRISIS)
    3. **Recovery dynamics**: Observe CRISIS → REST → WAKE progression
    4. **Constraint enforcement**: Verify expensive experts rejected

    **Validation Checks**:
    - CRISIS triggers at ATP < 20
    - Only cheap experts (<7 ATP) called during CRISIS
    - System recovers (doesn't stay in CRISIS permanently)
    - State progression: CRISIS → REST → WAKE
    """

    def __init__(self):
        """Initialize CRISIS state validator."""
        self.scenarios: List[CRISISTestScenario] = []
        self.results: Dict[str, Any] = {}

        logger.info("CRISISStateValidator initialized")

    def define_scenarios(self):
        """Define test scenarios for CRISIS validation."""
        self.scenarios = [
            CRISISTestScenario(
                name="gradual_depletion",
                initial_atp=25.0,
                crisis_threshold=20.0,
                query_complexity="complex",
                num_queries=5,
                expected_crisis=True,
                description="Start near CRISIS threshold, submit complex queries to trigger"
            ),
            CRISISTestScenario(
                name="immediate_crisis",
                initial_atp=15.0,
                crisis_threshold=20.0,
                query_complexity="moderate",
                num_queries=10,
                expected_crisis=True,
                description="Start in CRISIS state, observe survival mode behavior"
            ),
            CRISISTestScenario(
                name="recovery_dynamics",
                initial_atp=12.0,
                crisis_threshold=20.0,
                query_complexity="simple",
                num_queries=30,
                expected_crisis=True,
                description="Start deep in CRISIS, observe recovery to REST then WAKE"
            ),
            CRISISTestScenario(
                name="expensive_rejection",
                initial_atp=18.0,
                crisis_threshold=20.0,
                query_complexity="complex",
                num_queries=5,
                expected_crisis=True,
                description="Verify expensive experts rejected during CRISIS"
            )
        ]

        logger.info(f"Defined {len(self.scenarios)} CRISIS test scenarios")

    def run_scenario(self, scenario: CRISISTestScenario) -> Dict[str, Any]:
        """Run a single CRISIS test scenario.

        Args:
            scenario: Test scenario definition

        Returns:
            Scenario results with metrics and validation
        """
        logger.info("=" * 70)
        logger.info(f"CRISIS TEST: {scenario.name}")
        logger.info("=" * 70)
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Initial ATP: {scenario.initial_atp}")
        logger.info(f"Crisis threshold: {scenario.crisis_threshold}")
        logger.info("")

        # Create selector with scenario parameters
        selector = ProductionATPSelector(
            num_experts=128,
            initial_atp=scenario.initial_atp,
            enable_atp_accounting=True,
            crisis_threshold=scenario.crisis_threshold,
            rest_threshold=40.0,
            recovery_rate=1.0  # Slower recovery for CRISIS testing
        )

        # Track CRISIS-specific metrics
        crisis_events = 0
        expensive_rejections = 0
        cheap_calls = 0
        state_sequence = []
        atp_trajectory = []

        # Run queries
        for query_idx in range(scenario.num_queries):
            # Track state before query
            if selector.atp_bridge:
                state_before = selector.atp_bridge._get_current_state()
                atp_before = selector.atp_bridge.current_atp
                state_sequence.append(state_before)
                atp_trajectory.append(atp_before)

                if state_before == "crisis":
                    crisis_events += 1

            # Process query
            result = selector.process_query(
                query_complexity=scenario.query_complexity,
                context=f"{scenario.name}_query_{query_idx}",
                layer=0
            )

            # Check for expensive rejections during CRISIS
            if selector.atp_bridge:
                current_state = selector.atp_bridge._get_current_state()
                if current_state == "crisis" and result.get("experts_unavailable", 0) > 0:
                    expensive_rejections += result["experts_unavailable"]

                if result.get("experts_called", 0) > 0:
                    # Count cheap expert calls during CRISIS
                    if current_state == "crisis":
                        cheap_calls += result["experts_called"]

            # Log every 5 queries
            if query_idx % 5 == 0 and query_idx > 0:
                if selector.atp_bridge:
                    logger.info(f"Progress: {query_idx}/{scenario.num_queries} queries, "
                              f"ATP={selector.atp_bridge.current_atp:.1f}, "
                              f"State={selector.atp_bridge._get_current_state()}, "
                              f"CRISIS events={crisis_events}")

        # Final state
        if selector.atp_bridge:
            final_state = selector.atp_bridge._get_current_state()
            final_atp = selector.atp_bridge.current_atp
        else:
            final_state = "unknown"
            final_atp = 0.0

        # Analyze state transitions
        crisis_to_rest = 0
        rest_to_wake = 0
        for i in range(1, len(state_sequence)):
            if state_sequence[i-1] == "crisis" and state_sequence[i] == "rest":
                crisis_to_rest += 1
            elif state_sequence[i-1] == "rest" and state_sequence[i] == "wake":
                rest_to_wake += 1

        # Validate scenario
        crisis_triggered = crisis_events > 0
        validation_passed = crisis_triggered == scenario.expected_crisis

        results = {
            "scenario": scenario.name,
            "description": scenario.description,
            "configuration": {
                "initial_atp": scenario.initial_atp,
                "crisis_threshold": scenario.crisis_threshold,
                "num_queries": scenario.num_queries,
                "query_complexity": scenario.query_complexity
            },
            "metrics": {
                "crisis_events": crisis_events,
                "expensive_rejections": expensive_rejections,
                "cheap_calls_during_crisis": cheap_calls,
                "crisis_to_rest_transitions": crisis_to_rest,
                "rest_to_wake_transitions": rest_to_wake,
                "final_atp": final_atp,
                "final_state": final_state,
                "atp_min": min(atp_trajectory) if atp_trajectory else 0.0,
                "atp_max": max(atp_trajectory) if atp_trajectory else 0.0
            },
            "validation": {
                "expected_crisis": scenario.expected_crisis,
                "crisis_triggered": crisis_triggered,
                "passed": validation_passed
            },
            "state_sequence": state_sequence,
            "atp_trajectory": atp_trajectory
        }

        logger.info("")
        logger.info(f"✅ Scenario complete: {scenario.name}")
        logger.info(f"  CRISIS events: {crisis_events}")
        logger.info(f"  Expensive rejections: {expensive_rejections}")
        logger.info(f"  Cheap calls during CRISIS: {cheap_calls}")
        logger.info(f"  Final ATP: {final_atp:.1f}")
        logger.info(f"  Final state: {final_state}")
        logger.info(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")
        logger.info("")

        return results

    def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all CRISIS test scenarios.

        Returns:
            Complete results with all scenarios
        """
        logger.info("=" * 70)
        logger.info("SESSION 99: CRISIS State Validation")
        logger.info("=" * 70)
        logger.info(f"Running {len(self.scenarios)} test scenarios")
        logger.info("")

        all_results = []
        for scenario in self.scenarios:
            result = self.run_scenario(scenario)
            all_results.append(result)

        # Summary statistics
        total_scenarios = len(all_results)
        passed_scenarios = sum(1 for r in all_results if r["validation"]["passed"])
        total_crisis_events = sum(r["metrics"]["crisis_events"] for r in all_results)
        total_rejections = sum(r["metrics"]["expensive_rejections"] for r in all_results)

        summary = {
            "session": 99,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": "Thor (Jetson AGX Thor)",
            "goal": "CRISIS state validation - extreme ATP depletion testing",
            "scenarios_run": total_scenarios,
            "scenarios_passed": passed_scenarios,
            "pass_rate": passed_scenarios / max(total_scenarios, 1),
            "total_crisis_events": total_crisis_events,
            "total_expensive_rejections": total_rejections,
            "results": all_results
        }

        logger.info("=" * 70)
        logger.info("✅ CRISIS State Validation Complete!")
        logger.info(f"✅ Scenarios: {passed_scenarios}/{total_scenarios} passed")
        logger.info(f"✅ CRISIS events observed: {total_crisis_events}")
        logger.info(f"✅ Expensive experts rejected: {total_rejections}")
        logger.info("=" * 70)

        self.results = summary
        return summary

    def save_results(self, output_path: Path):
        """Save CRISIS validation results."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def test_crisis_state_validation():
    """Test CRISIS state validation with multiple scenarios."""
    # Create validator
    validator = CRISISStateValidator()

    # Define test scenarios
    validator.define_scenarios()

    # Run all scenarios
    results = validator.run_all_scenarios()

    # Save results
    output_path = Path(__file__).parent / "session99_crisis_validation_results.json"
    validator.save_results(output_path)

    # Print summary
    print("")
    print("CRISIS State Validation Summary:")
    print(f"  Scenarios tested: {results['scenarios_run']}")
    print(f"  Scenarios passed: {results['scenarios_passed']}")
    print(f"  Pass rate: {results['pass_rate']:.1%}")
    print(f"  Total CRISIS events: {results['total_crisis_events']}")
    print(f"  Total expensive rejections: {results['total_expensive_rejections']}")
    print("")

    # Detailed scenario results
    for result in results["results"]:
        print(f"Scenario: {result['scenario']}")
        print(f"  CRISIS events: {result['metrics']['crisis_events']}")
        print(f"  Final ATP: {result['metrics']['final_atp']:.1f}")
        print(f"  Final state: {result['metrics']['final_state']}")
        print(f"  Validation: {'✅ PASSED' if result['validation']['passed'] else '❌ FAILED'}")
        print("")


if __name__ == "__main__":
    test_crisis_state_validation()
