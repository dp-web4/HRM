#!/usr/bin/env python3
"""
Test: Signed Memory Consolidation with Hardware-Grounded Consciousness
=======================================================================

Validates that memory consolidation with cryptographic signatures works correctly.

**Purpose**: Test component that wasn't validated in extended deployment due to
threshold issues preventing DREAM state entry.

**Test Strategy**:
1. Lower thresholds to force attention decisions
2. Force DREAM state transitions to trigger consolidation
3. Verify consolidated memories are signed correctly
4. Validate signature verification on loaded memories

**Author**: Claude (autonomous research) on Thor
**Date**: 2025-12-06
**Session**: Post-deployment validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

from thor_hardware_grounded_consciousness import (
    HardwareGroundedConsciousness,
    create_thor_sensors,
    MetabolicState,
    CompressionMode
)

import time
import json
from datetime import datetime, timezone


def test_signed_memory_consolidation():
    """Test memory consolidation with signatures"""
    print("=" * 80)
    print("TEST: SIGNED MEMORY CONSOLIDATION")
    print("=" * 80)
    print()

    # Create consciousness with LOWERED thresholds to force attention
    print("1️⃣  Creating hardware-grounded consciousness...")
    print("   (Using lowered thresholds to force attention)")
    sensors = create_thor_sensors()

    consciousness = HardwareGroundedConsciousness(
        consciousness_lct_id="thor-sage-test",
        sensors=sensors,
        compression_mode=CompressionMode.LINEAR,
        metabolic_thresholds={
            MetabolicState.WAKE: 0.35,   # Lowered from 0.45
            MetabolicState.FOCUS: 0.25,
            MetabolicState.REST: 0.75,
            MetabolicState.DREAM: 0.05
        }
    )
    print(f"   Identity: {consciousness.consciousness_key.to_compact_id()}")
    print()

    # Run cycles to build up observations
    print("2️⃣  Running cycles to build observation buffer...")
    attention_count = 0
    for i in range(30):
        result = consciousness.run_cycle()
        if result['attended']:
            attention_count += 1
            print(f"   Cycle {i+1}: ATTENDED (salience={result['salience']:.3f})")
        time.sleep(0.5)  # Faster cycles for testing

    print(f"\n   Attended: {attention_count}/30 cycles")
    print(f"   Observation buffer: {len(consciousness.observation_history)} observations")
    print()

    if len(consciousness.observation_history) < 10:
        print("⚠️  Not enough observations for consolidation test")
        print("   Threshold still too high or system too stable")
        print("   Forcing observations for test...")

        # Force observations into buffer for testing
        from collections import deque
        consciousness.observation_history = deque([
            {'focus': 'cpu', 'salience': 0.5, 'cycle': i}
            for i in range(15)
        ], maxlen=100)
        print(f"   Forced buffer: {len(consciousness.observation_history)} observations")
        print()

    # Force DREAM state to trigger consolidation
    print("3️⃣  Forcing DREAM state to trigger consolidation...")
    original_state = consciousness.metabolic_state
    consciousness.metabolic_state = MetabolicState.DREAM
    print(f"   State changed: {original_state.value} → DREAM")
    print()

    # Trigger consolidation
    print("4️⃣  Triggering memory consolidation...")
    initial_memory_count = len(consciousness.memories)
    consciousness._consolidate_memories()
    final_memory_count = len(consciousness.memories)

    print(f"   Memories before: {initial_memory_count}")
    print(f"   Memories after: {final_memory_count}")
    print(f"   New memories: {final_memory_count - initial_memory_count}")
    print()

    if final_memory_count == 0:
        print("❌ No memories consolidated - test failed")
        return False

    # Examine consolidated memories
    print("5️⃣  Examining consolidated signed memories...")
    for i, memory in enumerate(consciousness.memories):
        print(f"\n   Memory {i+1}: {memory.memory_id}")
        print(f"   Content: {memory.content}")
        print(f"   Salience: {memory.salience:.3f}")
        print(f"   Strength: {memory.strength:.3f}")
        print(f"   Signature:")
        print(f"     - Signer: {memory.signature.signer_lct_id}")
        print(f"     - Machine: {memory.signature.signer_machine}")
        print(f"     - Signed at: {memory.signature.signed_at}")
        print(f"     - Data hash: {memory.signature.data_hash[:32]}...")
        print(f"     - Signature: {memory.signature.signature[:32]}...")

        # Verify signature
        memory_bytes = json.dumps(memory.content, sort_keys=True).encode('utf-8')
        sig_valid = consciousness.lct_identity.verify_signature(
            memory.signature,
            memory_bytes
        )

        if sig_valid:
            print(f"     - ✅ Signature VALID")
        else:
            print(f"     - ❌ Signature INVALID")
            return False

    print()

    # Test tamper detection
    print("6️⃣  Testing tamper detection...")
    test_memory = consciousness.memories[0]
    print(f"   Original memory: {test_memory.content}")

    # Verify original
    memory_bytes = json.dumps(test_memory.content, sort_keys=True).encode('utf-8')
    original_valid = consciousness.lct_identity.verify_signature(
        test_memory.signature,
        memory_bytes
    )
    print(f"   Original signature: {'VALID' if original_valid else 'INVALID'} ✅")

    # Create tampered content
    tampered_content = test_memory.content.copy()
    tampered_content['observation_count'] = 999  # Tamper
    tampered_bytes = json.dumps(tampered_content, sort_keys=True).encode('utf-8')

    # Verify tampered (should fail)
    tampered_valid = consciousness.lct_identity.verify_signature(
        test_memory.signature,
        tampered_bytes
    )
    print(f"   Tampered signature: {'VALID' if tampered_valid else 'INVALID'} ✅")

    if tampered_valid:
        print("   ❌ ERROR: Tampered data passed verification!")
        return False

    print()

    # Final statistics
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    print(f"✅ Signed memory consolidation: WORKING")
    print(f"✅ Signature verification: WORKING")
    print(f"✅ Tamper detection: WORKING")
    print()
    print(f"Memories consolidated: {len(consciousness.memories)}")
    print(f"Signatures verified: {len(consciousness.memories)}")
    print(f"All signatures valid: YES")
    print(f"Tamper detection: YES")
    print()
    print("🎉 ALL TESTS PASSED")
    print()

    return True


def test_cross_session_memory_verification():
    """Test loading memories from previous session and verifying signatures"""
    print("=" * 80)
    print("TEST: CROSS-SESSION MEMORY VERIFICATION")
    print("=" * 80)
    print()

    # Session 1: Create and save signed memories
    print("Session 1: Creating signed memories...")
    consciousness1 = HardwareGroundedConsciousness(
        consciousness_lct_id="thor-sage-session-test",
        sensors=create_thor_sensors()
    )

    # Force observation buffer for testing
    from collections import deque
    consciousness1.observation_history = deque([
        {'focus': f'sensor{i}', 'salience': 0.5 + i*0.01, 'cycle': i}
        for i in range(20)
    ], maxlen=100)

    # Force DREAM and consolidate
    consciousness1.metabolic_state = MetabolicState.DREAM
    consciousness1._consolidate_memories()

    session1_memories = consciousness1.memories.copy()
    print(f"   Created {len(session1_memories)} signed memories")
    print()

    # Simulate saving to database (serialize)
    saved_memories = []
    for mem in session1_memories:
        saved_memories.append({
            'memory_id': mem.memory_id,
            'content': mem.content,
            'salience': mem.salience,
            'strength': mem.strength,
            'signature': mem.signature.to_dict(),
            'consolidated_at': mem.consolidated_at
        })

    print("Session 2: Loading and verifying memories...")
    consciousness2 = HardwareGroundedConsciousness(
        consciousness_lct_id="thor-sage-session-test",  # Same identity
        sensors=create_thor_sensors()
    )

    # Load and verify
    verified_count = 0
    invalid_count = 0

    for saved in saved_memories:
        # Reconstruct signature
        from thor_hardware_grounded_consciousness import LCTSignature
        sig = LCTSignature(**saved['signature'])

        # Verify
        memory_bytes = json.dumps(saved['content'], sort_keys=True).encode('utf-8')
        valid = consciousness2.lct_identity.verify_signature(sig, memory_bytes)

        if valid:
            verified_count += 1
            print(f"   ✅ Memory {saved['memory_id']}: Valid signature")
        else:
            invalid_count += 1
            print(f"   ❌ Memory {saved['memory_id']}: Invalid signature")

    print()
    print(f"Total memories: {len(saved_memories)}")
    print(f"Verified: {verified_count}")
    print(f"Invalid: {invalid_count}")
    print()

    if invalid_count == 0:
        print("✅ CROSS-SESSION VERIFICATION: PASSED")
        print()
        return True
    else:
        print("❌ CROSS-SESSION VERIFICATION: FAILED")
        print()
        return False


def main():
    """Run all tests"""
    print()
    print("🧪 HARDWARE-GROUNDED CONSCIOUSNESS: MEMORY CONSOLIDATION TESTS")
    print()

    # Test 1: Signed memory consolidation
    test1_passed = test_signed_memory_consolidation()

    print()
    print("Continuing to cross-session test...")
    print()

    # Test 2: Cross-session verification
    test2_passed = test_cross_session_memory_verification()

    # Final summary
    print("=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print()
    print(f"Test 1 (Signed Memory Consolidation): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Cross-Session Verification): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print()

    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED")
        print()
        print("Validated:")
        print("  ✅ Memory consolidation with cryptographic signatures")
        print("  ✅ Signature verification on consolidated memories")
        print("  ✅ Tamper detection (modified data rejected)")
        print("  ✅ Cross-session memory verification")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
