#!/usr/bin/env python3
"""
Web4 Session Governance Integration for SAGE Raising

This module integrates Web4's governance system into SAGE raising sessions,
providing an audit trail of the Claude/human collaborative teaching process.

CRITICAL FRACTAL DISTINCTION:
============================

This operates at a DIFFERENT MRH scale than SAGE's internal R6/T3:

1. SAGE Internal R6/T3 (sage-core Rust):
   - SAGE evaluating its own training responses
   - "How well did I do on this exercise?"
   - Trust tensor of SAGE's competence/reliability/integrity
   - INSIDE SAGE's cognitive loop
   - Tier 3: Training evaluation

2. Session Governance (this module):
   - Claude + Human collaboratively raising SAGE
   - "How is the developmental process going?"
   - Audit trail of the teaching/learning process
   - META-LEVEL ABOVE SAGE
   - Tier 1: Observational audit

These are fractally distinct - one is self-assessment, the other is
process audit. They may eventually bridge, but they operate at different
scales and serve different purposes.

INTEGRATION APPROACH:
====================

The web4-governance plugin has both Python and Rust implementations:
- Python: claude-code-plugin/governance/ (used here for simplicity)
- Rust: web4-trust-core/ (WASM bindings, 10-50x faster)

We use the Python version because:
1. Direct import (no subprocess overhead)
2. Same runtime as session runner
3. Fast enough for meta-level audit (not in hot path)
4. Shares SQLite ledger with Claude Code sessions
5. Simpler for this use case

The Rust version would be appropriate if:
- Session exchanges happened milliseconds apart
- We needed sub-millisecond latency
- We were processing thousands of sessions/second
- We needed WASM browser integration

For raising sessions (5-10 exchanges over minutes), Python is perfect.

LOCAL REQUIREMENTS:
==================

This module requires:
- web4-governance plugin installed at ~/ai-workspace/web4/claude-code-plugin
- Python governance module (no npm/node required)
- SQLite3 (standard library)

The governance system creates:
- ~/.web4/sage-raising/ledger.db - Session audit database
- Session records with R6-compatible schema
- Witnessing chain across sessions
- ATP tracking at meta-level

VALUE-ADD IN THIS CONTEXT:
=========================

1. Longitudinal Trust Tracking:
   - Track SAGE's identity emergence over S01→S02→...→S43
   - T3-like metrics: competence (identity %), reliability (confabulation),
     integrity (epistemic boundaries)
   - Identify regression patterns (S42 60% → S43 0%)

2. Session Comparability:
   - Compare interventions (identity anchoring v1 vs v2)
   - Track ATP consumption of different teaching strategies
   - Audit trail for research reproducibility

3. Cross-Session Patterns:
   - Query: "Which phase transitions showed identity collapse?"
   - Query: "What interventions preceded confabulation spikes?"
   - Query: "How does salience correlate with identity stability?"

4. Process Transparency:
   - Every exchange logged with intent (prompt) and result (response)
   - Witnessing chain shows who participated (Claude, Dennis, Thor, etc.)
   - Audit trail for publishing SAGE's development process

LESSONS LEARNED:
===============

1. Fractal MRH Matters:
   - Don't conflate scales - internal vs external audit are different
   - Each level needs appropriate tooling
   - Bridge points can be designed later

2. Python vs Rust Trade-offs:
   - Python: Simple integration, adequate performance
   - Rust: Complex integration, overkill for this use case
   - Choose based on actual requirements, not performance religion

3. Reuse Over Rebuild:
   - The governance module already exists
   - Direct Python import is simpler than subprocess calls
   - Same SQLite schema means compatible tools

4. Feature Flags Are Essential:
   - Not everyone needs governance overhead
   - Optional integration allows graceful degradation
   - Can enable/disable without code changes

5. Integration Process Is the Product:
   - This documentation is as valuable as the code
   - Shows how to integrate Web4 into diverse contexts
   - Template for future integrations (training, autonomous sessions, etc.)

Created: 2026-01-24
Author: Claude (with Dennis guidance)
License: Same as HRM (check parent LICENSE)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add web4 governance to path
WEB4_PATH = Path.home() / "ai-workspace" / "web4" / "claude-code-plugin"
if WEB4_PATH.exists():
    sys.path.insert(0, str(WEB4_PATH))
    try:
        from governance.session_manager import SessionManager
        from governance.ledger import Ledger
        GOVERNANCE_AVAILABLE = True
    except ImportError as e:
        GOVERNANCE_AVAILABLE = False
        IMPORT_ERROR = str(e)
else:
    GOVERNANCE_AVAILABLE = False
    IMPORT_ERROR = f"Web4 plugin not found at {WEB4_PATH}"


class SageSessionGovernance:
    """
    Web4 governance integration for SAGE raising sessions.

    Provides meta-level audit of the Claude/human collaborative teaching process.
    Independent of SAGE's internal R6/T3 (operates at different fractal scale).

    Usage:
        governance = SageSessionGovernance()

        # At session start
        session_info = governance.start_session(session_num=39, phase="questioning")

        # After each exchange
        governance.track_exchange(
            turn_num=1,
            prompt="How are you doing today?",
            response="As SAGE, I'm observing patterns...",
            salience=0.67
        )

        # At session end
        summary = governance.end_session({
            "self_reference_pct": 60.0,
            "avg_salience": 0.65,
            "confabulation": False,
            "turn_count": 5
        })
    """

    def __init__(self, project: str = "sage-raising", enable: bool = True):
        """
        Initialize governance integration.

        Args:
            project: Project identifier for session grouping
            enable: Whether to actually enable governance (feature flag)
        """
        self.project = project
        self.enabled = enable and GOVERNANCE_AVAILABLE
        self.session_mgr = None
        self.current_session = None

        if not self.enabled:
            return

        # Custom ledger for SAGE raising sessions
        # Separate from Claude Code sessions (different context)
        ledger_path = Path.home() / ".web4" / "sage-raising" / "ledger.db"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.ledger = Ledger(ledger_path)
            self.session_mgr = SessionManager(self.ledger)
        except Exception as e:
            print(f"[Governance] Failed to initialize: {e}")
            self.enabled = False

    def start_session(self, session_num: int, phase: str,
                     atp_budget: int = 100) -> Optional[Dict]:
        """
        Start governed session with R6 session token.

        Args:
            session_num: Session number (e.g., 39 for S39)
            phase: Development phase (grounding, sensing, relating, questioning, creating)
            atp_budget: Meta-level ATP for teaching process

        Returns:
            Session info dict with session_id, lct_id, etc.
        """
        if not self.enabled:
            return None

        try:
            # Sync with filesystem to maintain consistent numbering
            session_path = Path(__file__).parent.parent / "sessions" / "text"

            self.current_session = self.session_mgr.start_session(
                project=self.project,
                atp_budget=atp_budget,
                sync_with_filesystem=True,
                fs_pattern=r"session_(\d+)\.json",
                fs_path=session_path
            )

            # Note: Metadata is set during start_session call
            # We can't modify it after creation with current ledger API
            # Future enhancement: Add update_session_metadata to ledger

            return self.current_session

        except Exception as e:
            print(f"[Governance] Failed to start session: {e}")
            return None

    def track_exchange(self, turn_num: int, prompt: str, response: str,
                      salience: Optional[float] = None,
                      identity_marker: bool = False) -> bool:
        """
        Track Claude<->SAGE exchange as R6 action.

        This creates an audit record of each teaching exchange:
        - Request: Claude's prompt (intent)
        - Result: SAGE's response (outcome)
        - Metadata: Salience, identity markers, etc.

        Args:
            turn_num: Exchange number in session
            prompt: Claude's prompt to SAGE
            response: SAGE's response
            salience: Experience salience score (0-1)
            identity_marker: Whether response contained "As SAGE"

        Returns:
            True if tracked successfully, False otherwise
        """
        if not self.enabled or not self.current_session:
            return False

        try:
            # Record as action using session manager API
            # Note: Ledger stores hashes, not full content (privacy/size)
            input_data = {
                "turn": turn_num,
                "prompt": prompt[:100],  # First 100 chars for hash
                "fractal_level": "exchange"
            }

            output_data = {
                "response_preview": response[:200],
                "response_length": len(response),
                "salience": salience,
                "identity_marker": identity_marker
            }

            self.session_mgr.record_action(
                tool_name="sage_exchange",
                target=f"turn_{turn_num}",
                input_data=input_data,
                output_data=json.dumps(output_data),
                status="success",
                atp_cost=1  # Each exchange costs 1 ATP
            )
            return True

        except Exception as e:
            print(f"[Governance] Failed to track exchange {turn_num}: {e}")
            return False

    def end_session(self, metrics: Dict[str, Any]) -> Optional[Dict]:
        """
        End session with T3-like developmental metrics.

        Args:
            metrics: Session summary metrics:
                - self_reference_pct: Identity marker percentage
                - avg_salience: Average experience salience
                - confabulation: Whether confabulation detected
                - turn_count: Total exchanges
                - phase: Development phase

        Returns:
            Session summary or None if failed
        """
        if not self.enabled or not self.current_session:
            return None

        try:
            # Build session summary with developmental metrics
            summary = {
                # Identity metrics (proxy for "competence" at meta-level)
                "identity_pct": metrics.get("self_reference_pct", 0),
                "identity_stable": metrics.get("self_reference_pct", 0) >= 30,

                # Salience metrics (proxy for "engagement")
                "avg_salience": metrics.get("avg_salience", 0),
                "high_salience_count": metrics.get("high_salience_count", 0),

                # Epistemic boundary metrics (proxy for "integrity")
                "confabulation_detected": metrics.get("confabulation", False),
                "epistemic_boundary_maintained": not metrics.get("confabulation", False),

                # Session metrics
                "total_exchanges": metrics.get("turn_count", 0),
                "phase": metrics.get("phase", "unknown"),
                "atp_consumed": self.current_session.get('atp_consumed', 0),

                # Meta information
                "fractal_level": "session",
                "governance_tier": "observational"
            }

            result = self.session_mgr.end_session(status="completed")

            if result:
                # Add our custom summary
                result['developmental_metrics'] = summary

            return result

        except Exception as e:
            print(f"[Governance] Failed to end session: {e}")
            return None

    @staticmethod
    def is_available() -> bool:
        """Check if governance is available (web4 plugin installed)."""
        return GOVERNANCE_AVAILABLE

    @staticmethod
    def get_availability_info() -> Dict[str, Any]:
        """Get detailed availability information."""
        return {
            "available": GOVERNANCE_AVAILABLE,
            "web4_path": str(WEB4_PATH),
            "web4_exists": WEB4_PATH.exists(),
            "error": None if GOVERNANCE_AVAILABLE else IMPORT_ERROR
        }


# Convenience functions for simple integration

def create_governance(enable: bool = True) -> SageSessionGovernance:
    """
    Create governance instance with availability checking.

    Args:
        enable: Whether to enable governance (feature flag)

    Returns:
        SageSessionGovernance instance (may be disabled if unavailable)
    """
    gov = SageSessionGovernance(enable=enable)

    if enable and not gov.enabled:
        info = SageSessionGovernance.get_availability_info()
        print(f"\n[Governance] Not available: {info['error']}")
        print(f"[Governance] Running without audit trail\n")

    return gov


def print_governance_status():
    """Print governance availability status (for debugging)."""
    info = SageSessionGovernance.get_availability_info()

    print("\n" + "="*60)
    print("WEB4 GOVERNANCE STATUS")
    print("="*60)
    print(f"Available: {info['available']}")
    print(f"Web4 Path: {info['web4_path']}")
    print(f"Path Exists: {info['web4_exists']}")

    if not info['available']:
        print(f"Error: {info['error']}")
    else:
        print("Status: Ready for session audit")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Self-test
    print_governance_status()

    if GOVERNANCE_AVAILABLE:
        print("\nTesting governance integration...")

        # Create instance
        gov = create_governance(enable=True)

        # Start test session
        session = gov.start_session(
            session_num=999,
            phase="testing"
        )

        if session:
            print(f"✓ Session started: {session['session_id']}")

            # Track test exchange
            gov.track_exchange(
                turn_num=1,
                prompt="Test prompt",
                response="Test response",
                salience=0.5,
                identity_marker=True
            )
            print("✓ Exchange tracked")

            # End session
            summary = gov.end_session({
                "self_reference_pct": 100.0,
                "avg_salience": 0.5,
                "confabulation": False,
                "turn_count": 1,
                "phase": "testing"
            })

            if summary:
                print(f"✓ Session ended: {summary['session_id']}")
                print(f"  ATP consumed: {summary.get('atp_consumed', 0)}")

                if 'developmental_metrics' in summary:
                    metrics = summary['developmental_metrics']
                    print(f"  Identity: {metrics['identity_pct']:.1f}%")
                    print(f"  Salience: {metrics['avg_salience']:.2f}")
                    print(f"  Confabulation: {metrics['confabulation_detected']}")

        print("\n✓ Self-test complete")
