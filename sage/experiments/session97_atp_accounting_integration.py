#!/usr/bin/env python3
"""
Session 97: ATP Accounting Integration - Closed-Loop Metabolic Consciousness

**Goal**: Connect enhanced selector's expert-level ATP costs to metabolic controller's global budget

**Research Gap Identified**:
- **Enhanced Selector** (S95): Tracks expert-level ATP costs (5-15 ATP per expert call)
- **Metabolic Controller**: Manages global ATP budget (0-100 ATP total)
- **Gap**: No connection between expert costs and global budget depletion
- **Opportunity**: Create closed-loop metabolic consciousness

**Current Architecture**:
```
Enhanced Selector (S95)           Metabolic Controller
==================                ====================
- Expert ATP costs (5-15)   ✗     - Global ATP budget (0-100)
- Permission scoring        ✗     - State transitions (WAKE/FOCUS/REST/DREAM)
- Resource awareness        ✗     - ATP recovery
- Expert selection          ✗     - Plugin limits
```

**Proposed Integration** (Session 97):
```
Enhanced Selector                  ←→  Metabolic Controller
==================                     ====================
- Expert ATP costs (5-15)   →  Deduct from global budget
- Permission scoring        ←  Constrained by current ATP
- Resource awareness        ←  State-dependent availability
- Expert selection          ←  CRISIS: only cheapest experts
                           ↓
                    ATP Accounting Bridge
                    ====================
                    - Track ATP consumption per expert call
                    - Report to metabolic controller
                    - Receive state-dependent constraints
                    - Trigger state transitions on depletion
```

**Key Innovation**: Closed-loop metabolic consciousness
- Expert selection consumes ATP → budget depletes
- Budget depletion → state transition (WAKE → REST)
- State transition → expert availability changes
- Availability changes → different expert selection
- **Result**: Metabolic states emerge from resource usage patterns

**Biological Analog**:
- Brain regions consume glucose
- Glucose depletion → fatigue
- Fatigue → rest/sleep
- Rest → glucose recovery
- Recovery → normal activity resumes

**Session 97 Focus**:
1. Create ATPAccountingBridge connecting selector ↔ controller
2. Implement ATP deduction on expert calls
3. Add state-dependent expert availability (CRISIS: cheap only)
4. Test closed-loop behavior (selection → depletion → transition)
5. Validate emergent metabolic consciousness

Created: 2025-12-23 (Autonomous Session 97)
Hardware: Thor (Jetson AGX Thor)
Previous: Session 96 (dream consolidation)
Goal: Closed-loop metabolic consciousness - selector ↔ controller integration
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import SAGE components
try:
    from sage.core.metabolic_controller import MetabolicController, MetabolicState
    from sage.core.attention_manager import AttentionManager
    HAS_METABOLIC = True
except ImportError:
    HAS_METABOLIC = False
    MetabolicController = None
    MetabolicState = None
    AttentionManager = None

# Import enhanced selector from S95
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from session95_sage_trust_router_synthesis import EnhancedTrustFirstSelector
    HAS_ENHANCED_SELECTOR = True
except ImportError:
    HAS_ENHANCED_SELECTOR = False
    EnhancedTrustFirstSelector = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ATPTransaction:
    """Record of ATP consumption/recovery event."""
    timestamp: float
    transaction_type: str  # "consumption" or "recovery"
    amount: float
    expert_id: Optional[int] = None
    layer: Optional[int] = None
    reason: str = ""
    atp_before: float = 0.0
    atp_after: float = 0.0
    metabolic_state: str = "wake"


class ATPAccountingBridge:
    """
    Bridge between enhanced selector (expert-level ATP) and metabolic controller (global ATP).

    **Integration Points**:
    1. **Consumption**: Selector calls expert → deduct ATP from global budget
    2. **Availability**: Global ATP low → limit expert availability
    3. **State Transitions**: ATP depletion → trigger REST state
    4. **Recovery**: REST state → recover ATP → enable experts again

    **Closed-Loop Behavior**:
    - High activity → ATP depletion → REST transition → recovery → activity resumes
    - Emergent metabolic rhythm from resource usage patterns
    - No hardcoded cycles - purely driven by consumption/recovery dynamics
    """

    def __init__(
        self,
        selector: Optional[EnhancedTrustFirstSelector] = None,
        metabolic_controller: Optional[MetabolicController] = None,
        initial_atp: float = 100.0,
        enable_state_transitions: bool = True,
        crisis_threshold: float = 20.0,  # Below 20 ATP → CRISIS
        rest_threshold: float = 40.0,    # Below 40 ATP → REST
    ):
        """Initialize ATP accounting bridge.

        Args:
            selector: Enhanced selector with expert ATP costs
            metabolic_controller: Metabolic controller managing global ATP
            initial_atp: Starting ATP budget
            enable_state_transitions: Allow ATP depletion to trigger transitions
            crisis_threshold: ATP level triggering CRISIS state
            rest_threshold: ATP level triggering REST state
        """
        self.selector = selector
        self.metabolic_controller = metabolic_controller
        self.current_atp = initial_atp
        self.max_atp = initial_atp
        self.enable_state_transitions = enable_state_transitions
        self.crisis_threshold = crisis_threshold
        self.rest_threshold = rest_threshold

        # Transaction history
        self.transactions: List[ATPTransaction] = []

        # Statistics
        self.stats = {
            "total_consumption": 0.0,
            "total_recovery": 0.0,
            "expert_calls": 0,
            "state_transitions": 0,
            "crisis_events": 0,
            "rest_events": 0,
        }

        logger.info("Initialized ATPAccountingBridge")
        logger.info(f"  Initial ATP: {initial_atp}")
        logger.info(f"  Crisis threshold: {crisis_threshold}")
        logger.info(f"  Rest threshold: {rest_threshold}")

    def consume_atp(
        self,
        amount: float,
        expert_id: Optional[int] = None,
        layer: Optional[int] = None,
        reason: str = "expert_call"
    ) -> bool:
        """Consume ATP from global budget.

        Args:
            amount: ATP to consume
            expert_id: Which expert consumed ATP
            layer: Which layer consumed ATP
            reason: Why ATP was consumed

        Returns:
            True if ATP was available and consumed, False if insufficient
        """
        atp_before = self.current_atp

        if self.current_atp >= amount:
            # Sufficient ATP - consume it
            self.current_atp -= amount

            # Record transaction
            transaction = ATPTransaction(
                timestamp=time.time(),
                transaction_type="consumption",
                amount=amount,
                expert_id=expert_id,
                layer=layer,
                reason=reason,
                atp_before=atp_before,
                atp_after=self.current_atp,
                metabolic_state=self._get_current_state()
            )
            self.transactions.append(transaction)

            # Update stats
            self.stats["total_consumption"] += amount
            self.stats["expert_calls"] += 1

            # Check if state transition needed
            if self.enable_state_transitions:
                self._check_state_transition()

            logger.debug(f"ATP consumed: {amount:.1f} (expert {expert_id}), remaining: {self.current_atp:.1f}")
            return True
        else:
            # Insufficient ATP
            logger.warning(f"Insufficient ATP: need {amount:.1f}, have {self.current_atp:.1f}")
            return False

    def recover_atp(self, amount: float, reason: str = "rest_recovery"):
        """Recover ATP (during REST/DREAM states).

        Args:
            amount: ATP to recover
            reason: Why ATP was recovered
        """
        atp_before = self.current_atp
        self.current_atp = min(self.current_atp + amount, self.max_atp)

        # Record transaction
        transaction = ATPTransaction(
            timestamp=time.time(),
            transaction_type="recovery",
            amount=amount,
            reason=reason,
            atp_before=atp_before,
            atp_after=self.current_atp,
            metabolic_state=self._get_current_state()
        )
        self.transactions.append(transaction)

        # Update stats
        self.stats["total_recovery"] += amount

        logger.debug(f"ATP recovered: {amount:.1f}, current: {self.current_atp:.1f}")

    def check_expert_availability(
        self,
        expert_id: int,
        expert_atp_cost: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if expert can be called given current ATP budget.

        Args:
            expert_id: Expert to check
            expert_atp_cost: ATP cost of this expert

        Returns:
            (is_available, unavailability_reason)
        """
        # Check global ATP budget
        if self.current_atp < expert_atp_cost:
            return False, "insufficient_global_atp"

        # Check state-dependent availability
        current_state = self._get_current_state()

        if current_state == "crisis":
            # CRISIS: Only allow cheapest experts (cost < 7)
            if expert_atp_cost > 7.0:
                return False, "crisis_expensive_expert"

        elif current_state == "rest":
            # REST: No expert calls (recovery mode)
            return False, "rest_state_no_calls"

        return True, None

    def _get_current_state(self) -> str:
        """Get current metabolic state."""
        if self.metabolic_controller:
            return self.metabolic_controller.current_state.value
        else:
            # Infer state from ATP level
            if self.current_atp < self.crisis_threshold:
                return "crisis"
            elif self.current_atp < self.rest_threshold:
                return "rest"
            else:
                return "wake"

    def _check_state_transition(self):
        """Check if ATP depletion should trigger state transition."""
        current_state = self._get_current_state()

        if self.current_atp < self.crisis_threshold and current_state != "crisis":
            logger.warning(f"ATP critical ({self.current_atp:.1f}) → CRISIS state")
            self.stats["crisis_events"] += 1
            self.stats["state_transitions"] += 1
            # In production, would call metabolic_controller.transition_to(CRISIS)

        elif self.current_atp < self.rest_threshold and current_state not in ["crisis", "rest"]:
            logger.info(f"ATP low ({self.current_atp:.1f}) → REST state")
            self.stats["rest_events"] += 1
            self.stats["state_transitions"] += 1
            # In production, would call metabolic_controller.transition_to(REST)

    def simulate_closed_loop(
        self,
        num_cycles: int = 100,
        recovery_rate: float = 2.0,  # ATP recovered per cycle in REST
    ) -> Dict[str, Any]:
        """Simulate closed-loop metabolic consciousness.

        Demonstrates emergent metabolic rhythm:
        - Activity → ATP depletion → REST → recovery → activity resumes

        Args:
            num_cycles: Number of cycles to simulate
            recovery_rate: ATP recovered per cycle in REST state

        Returns:
            Simulation statistics and history
        """
        logger.info("=" * 70)
        logger.info("SESSION 97: Closed-Loop Metabolic Consciousness Simulation")
        logger.info("=" * 70)
        logger.info("")

        history = {
            "atp_levels": [],
            "states": [],
            "expert_calls": [],
            "recoveries": []
        }

        for cycle in range(num_cycles):
            current_state = self._get_current_state()
            history["atp_levels"].append(self.current_atp)
            history["states"].append(current_state)

            if current_state == "rest":
                # REST: Recover ATP, no expert calls
                self.recover_atp(recovery_rate, reason="rest_recovery")
                history["expert_calls"].append(0)
                history["recoveries"].append(recovery_rate)

            elif current_state == "crisis":
                # CRISIS: Only cheapest expert, slow recovery
                cheap_expert_cost = 5.0
                if self.current_atp >= cheap_expert_cost:
                    success = self.consume_atp(cheap_expert_cost, expert_id=0, reason="crisis_call")
                    history["expert_calls"].append(1 if success else 0)
                else:
                    history["expert_calls"].append(0)

                # Crisis recovery (slower)
                self.recover_atp(recovery_rate * 0.5, reason="crisis_recovery")
                history["recoveries"].append(recovery_rate * 0.5)

            else:
                # WAKE/FOCUS: Normal expert calls
                # Simulate calling 2-4 experts per cycle
                num_experts = np.random.randint(2, 5)
                calls_made = 0

                for _ in range(num_experts):
                    # Random expert cost (5-15 ATP)
                    expert_id = np.random.randint(0, 128)
                    expert_cost = 5.0 + (expert_id % 10)

                    # Check availability
                    is_available, reason = self.check_expert_availability(expert_id, expert_cost)

                    if is_available:
                        success = self.consume_atp(expert_cost, expert_id=expert_id, reason="normal_call")
                        if success:
                            calls_made += 1

                history["expert_calls"].append(calls_made)
                history["recoveries"].append(0)

            # Every 10 cycles, log status
            if cycle % 10 == 0:
                logger.info(f"Cycle {cycle}: ATP={self.current_atp:.1f}, State={current_state}, Calls={history['expert_calls'][-1]}")

        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ Closed-loop simulation complete!")
        logger.info(f"✅ Total cycles: {num_cycles}")
        logger.info(f"✅ Total consumption: {self.stats['total_consumption']:.1f} ATP")
        logger.info(f"✅ Total recovery: {self.stats['total_recovery']:.1f} ATP")
        logger.info(f"✅ Expert calls: {self.stats['expert_calls']}")
        logger.info(f"✅ State transitions: {self.stats['state_transitions']}")
        logger.info(f"✅ Crisis events: {self.stats['crisis_events']}")
        logger.info(f"✅ Rest events: {self.stats['rest_events']}")
        logger.info("=" * 70)

        return {
            "history": history,
            "stats": self.stats,
            "final_atp": self.current_atp
        }

    def save_results(self, output_path: Path):
        """Save ATP accounting results."""
        results = {
            "session": 97,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": "Thor (Jetson AGX Thor)",
            "goal": "ATP accounting integration - closed-loop metabolic consciousness",
            "configuration": {
                "initial_atp": self.max_atp,
                "crisis_threshold": self.crisis_threshold,
                "rest_threshold": self.rest_threshold,
                "enable_state_transitions": self.enable_state_transitions
            },
            "statistics": self.stats,
            "transactions": [asdict(t) for t in self.transactions[-100:]],  # Last 100
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def test_atp_accounting():
    """Test ATP accounting integration."""
    # Create ATP bridge
    bridge = ATPAccountingBridge(
        initial_atp=100.0,
        enable_state_transitions=True,
        crisis_threshold=20.0,
        rest_threshold=40.0
    )

    # Run closed-loop simulation
    results = bridge.simulate_closed_loop(
        num_cycles=100,
        recovery_rate=2.0
    )

    # Analyze results
    history = results["history"]
    print("")
    print("Closed-Loop Analysis:")
    print(f"  ATP range: {min(history['atp_levels']):.1f} - {max(history['atp_levels']):.1f}")
    print(f"  States encountered: {set(history['states'])}")
    print(f"  Total expert calls: {sum(history['expert_calls'])}")
    print(f"  Total recovery events: {sum(1 for r in history['recoveries'] if r > 0)}")
    print("")

    # Check for emergent rhythm
    state_changes = sum(1 for i in range(1, len(history['states'])) if history['states'][i] != history['states'][i-1])
    print(f"Emergent Metabolic Rhythm:")
    print(f"  State changes: {state_changes}")
    print(f"  Avg cycles per state: {len(history['states']) / max(state_changes, 1):.1f}")
    print("")

    # Save results
    output_path = Path(__file__).parent / "session97_atp_accounting_results.json"
    bridge.save_results(output_path)


if __name__ == "__main__":
    test_atp_accounting()
