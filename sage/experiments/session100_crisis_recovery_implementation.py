#!/usr/bin/env python3
"""
Session 100: CRISIS Recovery Implementation - Basal ATP Metabolism

**Goal**: Implement basal ATP recovery during CRISIS state to prevent permanent depletion

**Research Gap Identified** (Session 99):
- **Problem**: System can reach ATP=0 and stay stuck permanently in CRISIS
- **Root Cause**: No ATP recovery mechanism during CRISIS state
- **Evidence**: 3/4 scenarios in S99 ended at ATP=0 with no recovery
- **Impact**: Production systems could get permanently stuck

**Biological Insight**:
Even organisms in crisis maintain **basal metabolic rate** - minimal energy
generation to sustain core functions and enable recovery. Current SAGE lacks this.

**Solution Design**:
```python
# Before (Session 97-99)
CRISIS state (ATP < 20):
  - ✅ Only cheap experts allowed (cost < 7 ATP)
  - ❌ No ATP recovery
  - ❌ Can reach ATP=0 and stay stuck

# After (Session 100)
CRISIS state (ATP < 20):
  - ✅ Only cheap experts allowed (cost < 7 ATP)
  - ✅ Basal ATP recovery (0.5-1.0 per cycle)
  - ✅ Gradual recovery: ATP=0 → ATP=20 → REST → WAKE
```

**Recovery Parameters**:
- **Basal recovery rate**: 0.5 ATP per cycle during CRISIS
- **Recovery condition**: Triggered when in CRISIS with no active processing
- **Recovery path**: CRISIS (ATP=0) → CRISIS (ATP>20) → REST → WAKE

**Biological Analog**:
- Brain in crisis: Still maintains minimal metabolic function
- Heart in failure: Still generates minimal ATP to sustain cells
- Starvation: Body breaks down reserves for minimal energy
- **Never** true zero - basal metabolism prevents complete shutdown

**Testing Strategy**:
1. Implement basal recovery in ATPAccountingBridge
2. Test isolated recovery: Start ATP=0, observe recovery to REST
3. Re-run Session 99 scenarios with recovery enabled
4. Validate: All scenarios can recover from any ATP level
5. Measure recovery time from ATP=0 to WAKE

Created: 2025-12-23 (Autonomous Session 100)
Hardware: Thor (Jetson AGX Thor)
Previous: Session 99 (CRISIS validation - gap discovered)
Goal: Complete biological metabolic model with basal ATP recovery
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import Session 97 components (with modifications)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from session97_atp_accounting_integration import ATPTransaction
    HAS_ATP_TRANSACTION = True
except ImportError:
    HAS_ATP_TRANSACTION = False
    ATPTransaction = None

# Import Session 98/99 components
try:
    from session98_production_atp_integration import ProductionATPSelector
    from session99_crisis_state_validation import CRISISStateValidator, CRISISTestScenario
    HAS_PRODUCTION_COMPONENTS = True
except ImportError:
    HAS_PRODUCTION_COMPONENTS = False
    ProductionATPSelector = None
    CRISISStateValidator = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ATPAccountingBridgeWithBasalRecovery:
    """
    Enhanced ATP accounting bridge with basal metabolic recovery during CRISIS.

    **Key Addition**: Basal ATP Recovery
    - Even in CRISIS, system maintains minimal metabolic function
    - Slow but steady recovery (0.5 ATP per cycle)
    - Prevents permanent depletion at ATP=0
    - Enables recovery path: CRISIS → REST → WAKE

    **Recovery Mechanism**:
    - Triggered during CRISIS when no queries being processed
    - Slower than REST recovery (0.5 vs 2.0 ATP/cycle)
    - Continues until ATP > 20 (CRISIS → REST transition)
    - Biological analog: Basal metabolic rate in extreme stress

    **Design Philosophy**:
    "No reachable state should be a trap" - system must be able to
    recover from any state it can reach through normal operation.
    """

    def __init__(
        self,
        initial_atp: float = 100.0,
        max_atp: Optional[float] = None,  # Allow separate max from initial
        crisis_threshold: float = 20.0,
        rest_threshold: float = 40.0,
        enable_state_transitions: bool = True,
        # New parameters for basal recovery
        enable_basal_recovery: bool = True,
        basal_recovery_rate: float = 0.5,  # Slower than REST (2.0)
    ):
        """Initialize ATP bridge with basal recovery support.

        Args:
            initial_atp: Starting ATP budget
            max_atp: Maximum ATP capacity (defaults to 100.0 if not specified)
            crisis_threshold: ATP level triggering CRISIS state
            rest_threshold: ATP level triggering REST state
            enable_state_transitions: Allow state transitions
            enable_basal_recovery: Enable basal ATP recovery during CRISIS
            basal_recovery_rate: ATP recovered per cycle during CRISIS (default: 0.5)
        """
        self.current_atp = initial_atp
        self.max_atp = max_atp if max_atp is not None else 100.0
        self.crisis_threshold = crisis_threshold
        self.rest_threshold = rest_threshold
        self.enable_state_transitions = enable_state_transitions

        # Basal recovery parameters
        self.enable_basal_recovery = enable_basal_recovery
        self.basal_recovery_rate = basal_recovery_rate

        # Transaction history
        self.transactions: List[ATPTransaction] = []

        # Statistics
        self.stats = {
            "total_consumption": 0.0,
            "total_recovery": 0.0,
            "basal_recovery": 0.0,  # Track basal recovery separately
            "expert_calls": 0,
            "state_transitions": 0,
            "crisis_events": 0,
            "rest_events": 0,
            "basal_recovery_cycles": 0,  # Number of cycles with basal recovery
        }

        logger.info("ATPAccountingBridge with Basal Recovery initialized")
        logger.info(f"  Initial ATP: {initial_atp}")
        logger.info(f"  Crisis threshold: {crisis_threshold}")
        logger.info(f"  Rest threshold: {rest_threshold}")
        logger.info(f"  Basal recovery: {'ENABLED' if enable_basal_recovery else 'DISABLED'}")
        logger.info(f"  Basal recovery rate: {basal_recovery_rate} ATP/cycle")

    def consume_atp(
        self,
        amount: float,
        expert_id: Optional[int] = None,
        layer: Optional[int] = None,
        reason: str = "expert_call"
    ) -> bool:
        """Consume ATP from global budget (same as Session 97)."""
        atp_before = self.current_atp

        if self.current_atp >= amount:
            self.current_atp -= amount

            # Record transaction
            if HAS_ATP_TRANSACTION:
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

            # Check state transition
            if self.enable_state_transitions:
                self._check_state_transition()

            logger.debug(f"ATP consumed: {amount:.1f} (expert {expert_id}), remaining: {self.current_atp:.1f}")
            return True
        else:
            logger.warning(f"Insufficient ATP: need {amount:.1f}, have {self.current_atp:.1f}")
            return False

    def recover_atp(self, amount: float, reason: str = "rest_recovery"):
        """Recover ATP (during REST/DREAM states or basal recovery)."""
        atp_before = self.current_atp
        self.current_atp = min(self.current_atp + amount, self.max_atp)

        # Record transaction
        if HAS_ATP_TRANSACTION:
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
        if reason == "basal_recovery":
            self.stats["basal_recovery"] += amount
            self.stats["basal_recovery_cycles"] += 1

        logger.debug(f"ATP recovered: {amount:.1f} ({reason}), current: {self.current_atp:.1f}")

    def apply_basal_recovery(self):
        """Apply basal ATP recovery during CRISIS state.

        This is the key innovation - even in CRISIS, system maintains
        minimal metabolic function that slowly recovers ATP.

        **When Applied**:
        - System is in CRISIS state (ATP < 20)
        - Basal recovery is enabled
        - No active query processing (idle cycle)

        **Recovery Rate**:
        - Slower than REST (0.5 vs 2.0 ATP/cycle)
        - Reflects minimal metabolic function vs normal recovery
        - Sufficient to eventually reach REST threshold (20 ATP)

        **Biological Justification**:
        Even organisms under extreme stress maintain basal metabolic rate.
        This prevents complete shutdown and enables recovery.
        """
        current_state = self._get_current_state()

        if current_state == "crisis" and self.enable_basal_recovery:
            # Apply basal recovery
            self.recover_atp(self.basal_recovery_rate, reason="basal_recovery")
            logger.debug(f"Basal recovery applied: +{self.basal_recovery_rate} ATP (now {self.current_atp:.1f})")

            # Check if recovered enough to transition to REST
            if self.current_atp >= self.crisis_threshold:
                logger.info(f"Basal recovery successful: ATP {self.current_atp:.1f} ≥ {self.crisis_threshold} (CRISIS → REST)")
                self.stats["state_transitions"] += 1

    def check_expert_availability(
        self,
        expert_id: int,
        expert_atp_cost: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if expert can be called given current ATP budget (same as Session 97)."""
        if self.current_atp < expert_atp_cost:
            return False, "insufficient_global_atp"

        current_state = self._get_current_state()

        if current_state == "crisis":
            if expert_atp_cost > 7.0:
                return False, "crisis_expensive_expert"
        elif current_state == "rest":
            return False, "rest_state_no_calls"

        return True, None

    def _get_current_state(self) -> str:
        """Get current metabolic state (same as Session 97)."""
        if self.current_atp < self.crisis_threshold:
            return "crisis"
        elif self.current_atp < self.rest_threshold:
            return "rest"
        else:
            return "wake"

    def _check_state_transition(self):
        """Check if ATP changes trigger state transitions (same as Session 97)."""
        current_state = self._get_current_state()

        if self.current_atp < self.crisis_threshold and current_state != "crisis":
            logger.warning(f"ATP critical ({self.current_atp:.1f}) → CRISIS state")
            self.stats["crisis_events"] += 1
            self.stats["state_transitions"] += 1

        elif self.current_atp < self.rest_threshold and current_state not in ["crisis", "rest"]:
            logger.info(f"ATP low ({self.current_atp:.1f}) → REST state")
            self.stats["rest_events"] += 1
            self.stats["state_transitions"] += 1


def test_basal_recovery():
    """Test basal ATP recovery in isolation."""
    logger.info("=" * 70)
    logger.info("SESSION 100: Basal ATP Recovery Test")
    logger.info("=" * 70)
    logger.info("")

    # Create bridge with basal recovery enabled
    bridge = ATPAccountingBridgeWithBasalRecovery(
        initial_atp=0.0,  # Start at zero - worst case
        crisis_threshold=20.0,
        rest_threshold=40.0,
        enable_basal_recovery=True,
        basal_recovery_rate=0.5
    )

    logger.info(f"Starting ATP: {bridge.current_atp}")
    logger.info(f"Starting state: {bridge._get_current_state()}")
    logger.info("")

    # Apply basal recovery for 50 cycles
    logger.info("Applying basal recovery for 50 cycles...")
    for cycle in range(50):
        bridge.apply_basal_recovery()

        # Log every 10 cycles
        if cycle % 10 == 0 and cycle > 0:
            logger.info(f"Cycle {cycle}: ATP={bridge.current_atp:.1f}, State={bridge._get_current_state()}")

    logger.info("")
    logger.info(f"Final ATP: {bridge.current_atp}")
    logger.info(f"Final state: {bridge._get_current_state()}")
    logger.info(f"Basal recovery applied: {bridge.stats['basal_recovery']:.1f} ATP over {bridge.stats['basal_recovery_cycles']} cycles")
    logger.info("")

    # Verify recovery path
    assert bridge.current_atp > 0, "Should have recovered from ATP=0"
    assert bridge.current_atp >= bridge.crisis_threshold, f"Should have reached REST (ATP={bridge.current_atp} < {bridge.crisis_threshold})"

    logger.info("✅ Basal recovery test PASSED")
    logger.info("✅ System can recover from ATP=0")
    logger.info(f"✅ Recovery time: {bridge.stats['basal_recovery_cycles']} cycles to reach REST")
    logger.info("")

    return bridge.stats


def test_crisis_recovery_with_basal():
    """Test CRISIS scenarios with basal recovery enabled."""
    logger.info("=" * 70)
    logger.info("SESSION 100: CRISIS Recovery Validation")
    logger.info("=" * 70)
    logger.info("Re-running Session 99 scenarios with basal recovery enabled")
    logger.info("")

    # Note: This would re-run Session 99 scenarios with modified bridge
    # For now, just log the intention
    logger.info("⚠️  Full re-validation requires integration with ProductionATPSelector")
    logger.info("⚠️  Would modify ProductionATPSelector to use ATPAccountingBridgeWithBasalRecovery")
    logger.info("⚠️  Current implementation validates basal recovery mechanism in isolation")
    logger.info("")

    return {"note": "Full integration test requires ProductionATPSelector modification"}


if __name__ == "__main__":
    # Test basal recovery in isolation
    stats = test_basal_recovery()

    # Note about full integration
    test_crisis_recovery_with_basal()

    # Save results
    results = {
        "session": 100,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Thor (Jetson AGX Thor)",
        "goal": "CRISIS recovery implementation - basal ATP metabolism",
        "basal_recovery_test": stats,
        "validation": {
            "recovery_from_zero": True,
            "reached_rest_threshold": True,
            "mechanism": "basal_recovery",
            "rate": 0.5
        }
    }

    output_path = Path(__file__).parent / "session100_crisis_recovery_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
