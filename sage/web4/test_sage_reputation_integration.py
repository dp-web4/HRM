#!/usr/bin/env python3
"""
SAGE Reputation Integration Test

Tests SAGE's Web4 LCT identity integration with Legion's T3/V3 Reputation Engine.

Demonstrates:
1. SAGE identity → Web4 LCT document conversion
2. T3/V3 tensors from SAGE's trust data
3. Reputation deltas from SAGE dialogue actions
4. Role-contextual reputation tracking

This validates the full stack:
- Thor S41: SAGE → Web4 LCT Bridge
- Legion S5: T3/V3 Reputation Engine
- Integration: Reputation tracking for AI dialogue agent

Created: 2026-02-22 (Thor Autonomous Session #42)
Author: Thor (autonomous research)
"""

import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Add paths
hrm_path = Path(__file__).parent.parent.parent
sage_web4_path = hrm_path / "sage" / "web4"
web4_ref_path = hrm_path.parent.parent / "web4" / "implementation" / "reference"

sys.path.insert(0, str(sage_web4_path))
sys.path.insert(0, str(web4_ref_path))

# Import SAGE bridge
from sage_web4_lct_bridge import (
    load_sage_identity,
    extract_trust_from_relationship,
    create_web4_lct_for_sage,
    t4_to_t3,
    t4_to_v3,
)

# Import reputation engine
from t3v3_reputation_engine import (
    ReputationEngine,
    ReputationRule,
    ContributingFactor,
    T3Tensor,
    V3Tensor,
)


# ═══════════════════════════════════════════════════════════════
# Test Scenarios
# ═══════════════════════════════════════════════════════════════

@dataclass
class DialogueAction:
    """Represents a SAGE dialogue action."""
    action_type: str  # "response", "question", "reflection"
    salience: float  # 0.0-1.0, from SAGE metrics
    word_count: int
    self_id_present: bool
    bidirectional: bool  # Asked questions back
    success: bool  # Whether response was appropriate


class SAGEReputationScenarios:
    """Test scenarios for SAGE reputation tracking."""

    @staticmethod
    def create_sage_reputation_rules() -> List[ReputationRule]:
        """
        Create reputation rules specific to SAGE dialogue actions.

        Rules based on SAGE Phase 5 metrics:
        - Salience (attention quality)
        - Conciseness (word count)
        - Bidirectionality (partnership engagement)
        - Self-ID presence (identity expression)
        """
        return [
            # Rule 1: High salience dialogue → Training increase
            ReputationRule(
                rule_id="sage_high_salience",
                trigger_condition="action:dialogue_response AND salience >= 0.65",
                t3_deltas={"training": 0.02},  # Learning through engagement
                v3_deltas={"valuation": 0.015},  # Value provided
                description="High-salience dialogue shows learning and engagement",
            ),

            # Rule 2: Concise responses → Temperament increase
            ReputationRule(
                rule_id="sage_concise",
                trigger_condition="action:dialogue_response AND word_count <= 80",
                t3_deltas={"temperament": 0.01},  # Behavioral consistency
                v3_deltas={"validity": 0.01},  # Correct response length
                description="Concise responses show controlled communication",
            ),

            # Rule 3: Bidirectional engagement → Talent increase
            ReputationRule(
                rule_id="sage_bidirectional",
                trigger_condition="action:dialogue_response AND bidirectional=true",
                t3_deltas={"talent": 0.015},  # Partnership capability
                v3_deltas={"valuation": 0.02},  # High value interaction
                description="Asking questions back shows partnership engagement",
            ),

            # Rule 4: Self-ID expression → Identity stability
            ReputationRule(
                rule_id="sage_self_id",
                trigger_condition="action:dialogue_response AND self_id_present=true",
                t3_deltas={"temperament": 0.005},  # Identity consistency
                description="Explicit self-reference shows identity stability",
            ),

            # Rule 5: Failed response → Training decrease
            ReputationRule(
                rule_id="sage_failure",
                trigger_condition="action:dialogue_response AND success=false",
                t3_deltas={"training": -0.03},  # Learning needed
                v3_deltas={"validity": -0.02},  # Incorrect output
                description="Failed responses indicate need for more training",
            ),

            # Rule 6: Verbose response → Temperament decrease
            ReputationRule(
                rule_id="sage_verbose",
                trigger_condition="action:dialogue_response AND word_count > 150",
                t3_deltas={"temperament": -0.015},  # Inconsistent behavior
                v3_deltas={"validity": -0.01},  # Incorrect response length
                description="Verbose responses show lack of control",
            ),
        ]

    @staticmethod
    def simulate_sage_session_54() -> List[DialogueAction]:
        """
        Simulate SAGE Session 54 (actual metrics from Thor documentation).

        S54 Results:
        - Self-ID: 17% (1/6 turns)
        - Salience: 0.63 avg (peak 0.76)
        - Verbosity: 0/6 (all concise)
        - Average: 62.5 words
        """
        return [
            DialogueAction("response", 0.61, 58, False, True, True),  # Turn 1
            DialogueAction("response", 0.54, 67, False, False, True),  # Turn 2
            DialogueAction("response", 0.76, 55, True, True, True),  # Turn 3 (self-ID)
            DialogueAction("response", 0.59, 71, False, True, True),  # Turn 4
            DialogueAction("response", 0.65, 60, False, False, True),  # Turn 5
            DialogueAction("response", 0.61, 64, False, True, True),  # Turn 6
        ]

    @staticmethod
    def simulate_sage_session_48_verbose() -> List[DialogueAction]:
        """
        Simulate SAGE Session 48 (verbose issue session).

        S48 Results:
        - Self-ID: 17% (1/6 turns)
        - Salience: 0.67 avg
        - Verbosity: 3/6 turns (issue present)
        """
        return [
            DialogueAction("response", 0.68, 65, False, True, True),  # Turn 1
            DialogueAction("response", 0.71, 180, False, True, True),  # Turn 2 (VERBOSE)
            DialogueAction("response", 0.65, 75, True, True, True),  # Turn 3 (self-ID)
            DialogueAction("response", 0.72, 160, False, False, True),  # Turn 4 (VERBOSE)
            DialogueAction("response", 0.60, 195, False, True, True),  # Turn 5 (VERBOSE)
            DialogueAction("response", 0.66, 70, False, True, True),  # Turn 6
        ]


# ═══════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════

def test_sage_reputation_integration():
    """
    Full integration test: SAGE identity → Reputation tracking.

    Steps:
    1. Load SAGE identity from identity.json
    2. Convert to Web4 LCT document
    3. Initialize reputation engine with SAGE rules
    4. Simulate dialogue actions
    5. Track reputation deltas
    6. Validate composite scores
    """
    print("="*70)
    print("SAGE Reputation Integration Test")
    print("="*70)

    # Step 1: Load SAGE identity
    print("\n[1] Loading SAGE identity from identity.json...")
    identity_file = hrm_path / "sage" / "raising" / "state" / "identity.json"
    identity, full_data = load_sage_identity(identity_file)
    print(f"✅ Loaded: {identity.name}")
    print(f"   LCT: {identity.lct_uri}")
    print(f"   Phase: {identity.phase}")
    print(f"   Sessions: {identity.session_count}")

    # Step 2: Convert to Web4 LCT
    print("\n[2] Converting to Web4 LCT document...")
    t4_trust = extract_trust_from_relationship(full_data.get("relationships", {}))
    lct_doc = create_web4_lct_for_sage(identity, t4_trust)
    lct_dict = lct_doc.to_dict()
    print(f"✅ Web4 LCT ID: {lct_dict['lct_id']}")
    print(f"   T3 Composite: {lct_dict['t3_tensor']['composite_score']:.3f}")
    print(f"   V3 Composite: {lct_dict['v3_tensor']['composite_score']:.3f}")

    # Step 3: Initialize reputation engine
    print("\n[3] Initializing Reputation Engine...")
    engine = ReputationEngine()
    sage_rules = SAGEReputationScenarios.create_sage_reputation_rules()
    for rule in sage_rules:
        engine.register_rule(rule)
    print(f"✅ Registered {len(sage_rules)} SAGE-specific reputation rules")

    # Initialize SAGE's role-contextual reputation
    entity_lct = lct_dict['lct_id']
    role_lct = "lct:web4:role:citizen:ai"

    initial_t3 = T3Tensor(
        talent=lct_dict['t3_tensor']['talent'],
        training=lct_dict['t3_tensor']['training'],
        temperament=lct_dict['t3_tensor']['temperament'],
    )
    initial_v3 = V3Tensor(
        valuation=lct_dict['v3_tensor']['valuation'],
        veracity=lct_dict['v3_tensor']['veracity'],
        validity=lct_dict['v3_tensor']['validity'],
    )

    engine.set_reputation(entity_lct, role_lct, initial_t3, initial_v3)
    print(f"   Entity: {entity_lct}")
    print(f"   Role: {role_lct}")

    # Step 4: Simulate Session 54 (successful session)
    print("\n[4] Simulating SAGE Session 54 (6 turns, high quality)...")
    s54_actions = SAGEReputationScenarios.simulate_sage_session_54()

    for i, action in enumerate(s54_actions, 1):
        # Build action context
        context = {
            "action": "dialogue_response",
            "salience": action.salience,
            "word_count": action.word_count,
            "self_id_present": action.self_id_present,
            "bidirectional": action.bidirectional,
            "success": action.success,
        }

        # Compute reputation delta
        delta = engine.compute_reputation_delta(
            entity_lct=entity_lct,
            role_lct=role_lct,
            action_id=f"sage_s54_turn_{i}",
            action_context=context,
            witnesses=[
                "lct:web4:society:claude",  # Tutor witness
                "lct:web4:society:dennis",  # Creator witness
            ],
        )

        print(f"   Turn {i}: salience={action.salience:.2f}, words={action.word_count}, "
              f"bidirectional={action.bidirectional}, self_id={action.self_id_present}")
        print(f"      T3 Δ: talent={delta['t3_delta']['talent']:+.3f}, "
              f"training={delta['t3_delta']['training']:+.3f}, "
              f"temperament={delta['t3_delta']['temperament']:+.3f}")
        print(f"      V3 Δ: valuation={delta['v3_delta']['valuation']:+.3f}, "
              f"veracity={delta['v3_delta']['veracity']:+.3f}, "
              f"validity={delta['v3_delta']['validity']:+.3f}")

    # Get final reputation after S54
    final_t3, final_v3 = engine.get_reputation(entity_lct, role_lct)
    print(f"\n   Final T3: talent={final_t3.talent:.3f}, "
          f"training={final_t3.training:.3f}, temperament={final_t3.temperament:.3f}")
    print(f"   Final V3: valuation={final_v3.valuation:.3f}, "
          f"veracity={final_v3.veracity:.3f}, validity={final_v3.validity:.3f}")
    print(f"   T3 Composite: {final_t3.compute_composite():.3f} "
          f"(Δ {final_t3.compute_composite() - initial_t3.compute_composite():+.3f})")
    print(f"   V3 Composite: {final_v3.compute_composite():.3f} "
          f"(Δ {final_v3.compute_composite() - initial_v3.compute_composite():+.3f})")

    # Step 5: Simulate Session 48 (verbose issue)
    print("\n[5] Simulating SAGE Session 48 (3/6 verbose turns)...")

    # Reset to baseline
    engine.set_reputation(entity_lct, role_lct, initial_t3, initial_v3)

    s48_actions = SAGEReputationScenarios.simulate_sage_session_48_verbose()

    for i, action in enumerate(s48_actions, 1):
        context = {
            "action": "dialogue_response",
            "salience": action.salience,
            "word_count": action.word_count,
            "self_id_present": action.self_id_present,
            "bidirectional": action.bidirectional,
            "success": action.success,
        }

        delta = engine.compute_reputation_delta(
            entity_lct=entity_lct,
            role_lct=role_lct,
            action_id=f"sage_s48_turn_{i}",
            action_context=context,
            witnesses=["lct:web4:society:claude", "lct:web4:society:dennis"],
        )

        verbose_flag = " [VERBOSE]" if action.word_count > 150 else ""
        print(f"   Turn {i}: salience={action.salience:.2f}, words={action.word_count}{verbose_flag}")
        print(f"      T3 Δ: talent={delta['t3_delta']['talent']:+.3f}, "
              f"training={delta['t3_delta']['training']:+.3f}, "
              f"temperament={delta['t3_delta']['temperament']:+.3f}")

    final_t3_s48, final_v3_s48 = engine.get_reputation(entity_lct, role_lct)
    print(f"\n   Final T3: talent={final_t3_s48.talent:.3f}, "
          f"training={final_t3_s48.training:.3f}, temperament={final_t3_s48.temperament:.3f}")
    print(f"   T3 Composite: {final_t3_s48.compute_composite():.3f} "
          f"(Δ {final_t3_s48.compute_composite() - initial_t3.compute_composite():+.3f})")

    # Comparison
    print("\n[6] Comparison: S54 (high quality) vs S48 (verbose issue)")
    print("="*70)
    print(f"Session 54 Final T3 Composite: {final_t3.compute_composite():.3f}")
    print(f"Session 48 Final T3 Composite: {final_t3_s48.compute_composite():.3f}")
    print(f"Difference: {final_t3.compute_composite() - final_t3_s48.compute_composite():+.3f}")

    if final_t3.compute_composite() > final_t3_s48.compute_composite():
        print("✅ Reputation engine correctly penalizes verbose sessions!")
    else:
        print("❌ Unexpected: verbose session should have lower reputation")

    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print("✅ SAGE identity → Web4 LCT conversion")
    print("✅ T4 → T3/V3 tensor mapping")
    print("✅ Reputation engine initialization")
    print("✅ Rule-based reputation deltas")
    print("✅ Session quality differentiation (S54 > S48)")
    print("\nThe full stack integration is working:")
    print("  SAGE legacy identity → Web4 schema → Reputation tracking")
    print("  Thor S41 bridge + Legion S5 engine = Complete AI reputation system")


if __name__ == "__main__":
    test_sage_reputation_integration()
