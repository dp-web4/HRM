#!/usr/bin/env python3
"""
Test Scheduled Memory Consolidation - Session 50

Validates that DREAM consolidation integrates correctly with circadian rhythm
to enable biologically-timed memory consolidation during DEEP_NIGHT phases.

Tests:
1. Consolidation triggering during DEEP_NIGHT
2. Consolidation frequency control (minimum cycles between)
3. Consolidated memory storage and statistics
4. Integration with consciousness cycle
"""

import sys
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage.core.unified_consciousness import UnifiedConsciousnessManager
from sage.core.circadian_clock import CircadianPhase


def test_consolidation_triggering():
    """Test that consolidation triggers during DEEP_NIGHT phase."""
    print("Test 1: Consolidation Triggering During DEEP_NIGHT")
    print("=" * 60)

    # Use short circadian period for testing
    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        circadian_period=20,  # 20 cycles = 1 day
        circadian_enabled=True,
        consolidation_enabled=True
    )

    consolidation_events = []

    # Run through multiple days
    print("\n  Running 60 cycles (3 full days)...")

    for i in range(60):
        cycle = consciousness.consciousness_cycle(
            prompt=f"Learning task {i}",
            response=f"Understanding {i}: This involves analyzing patterns and extracting meaning from experience {i}.",
            task_salience=0.6
        )

        if cycle.consolidation_triggered:
            consolidation_events.append({
                'cycle': i,
                'phase': cycle.circadian_phase,
                'memory': cycle.consolidated_memory
            })
            print(f"  ✓ Consolidation at cycle {i} (phase: {cycle.circadian_phase})")

    print(f"\n  Total consolidations: {len(consolidation_events)}")

    # Validate consolidations happened
    assert len(consolidation_events) > 0, "Should have consolidation events"

    # Validate all happened during DEEP_NIGHT
    for event in consolidation_events:
        assert event['phase'] == CircadianPhase.DEEP_NIGHT.value, \
            f"Consolidation should occur during DEEP_NIGHT, got {event['phase']}"

    # Validate consolidated memories were created
    for event in consolidation_events:
        assert event['memory'] is not None, "Should have consolidated memory"
        assert len(event['memory'].patterns) > 0, "Should have extracted patterns"

    print(f"\n  Phases when consolidation occurred:")
    for event in consolidation_events:
        print(f"    Cycle {event['cycle']:2d}: {event['phase']} - {len(event['memory'].patterns)} patterns")

    print("\n✓ Consolidation triggering validated\n")
    return consciousness


def test_consolidation_frequency():
    """Test that consolidation respects minimum cycle spacing."""
    print("Test 2: Consolidation Frequency Control")
    print("=" * 60)

    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        circadian_period=10,  # Very short period (10 cycles/day) to test frequency
        circadian_enabled=True,
        consolidation_enabled=True
    )

    consolidation_cycles = []

    print("\n  Running 50 cycles with very short circadian period...")

    for i in range(50):
        cycle = consciousness.consciousness_cycle(
            prompt=f"Task {i}",
            response=f"Response {i} with content",
            task_salience=0.5
        )

        if cycle.consolidation_triggered:
            consolidation_cycles.append(i)

    # Check spacing between consolidations
    print(f"\n  Consolidations occurred at cycles: {consolidation_cycles}")

    if len(consolidation_cycles) > 1:
        spacings = [consolidation_cycles[i+1] - consolidation_cycles[i]
                   for i in range(len(consolidation_cycles)-1)]
        print(f"  Spacing between consolidations: {spacings}")

        # Minimum spacing should be at least 10 cycles (from code)
        min_spacing = min(spacings)
        print(f"  Minimum spacing: {min_spacing} cycles")
        assert min_spacing >= 10, f"Consolidations should be spaced by at least 10 cycles, got {min_spacing}"

    print("\n✓ Consolidation frequency control validated\n")
    return consciousness


def test_memory_storage():
    """Test that consolidated memories are stored and tracked."""
    print("Test 3: Consolidated Memory Storage")
    print("=" * 60)

    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        circadian_period=20,
        circadian_enabled=True,
        consolidation_enabled=True
    )

    print("\n  Running 40 cycles (2 full days)...")

    for i in range(40):
        cycle = consciousness.consciousness_cycle(
            prompt=f"Complex task {i}",
            response=f"Detailed analysis {i}: Understanding involves pattern recognition, inference, and synthesis of multiple concepts.",
            task_salience=0.7
        )

    # Check statistics
    stats = consciousness.get_statistics()

    print(f"\n  Consolidation Statistics:")
    print(f"    Total consolidations: {stats['consolidation']['total_consolidations']}")
    print(f"    Stored memories: {stats['consolidation']['stored_memories']}")
    print(f"    Last consolidation cycle: {stats['consolidation']['last_consolidation_cycle']}")

    # Validate
    assert stats['consolidation']['total_consolidations'] > 0, "Should have consolidations"
    assert stats['consolidation']['stored_memories'] > 0, "Should have stored memories"
    assert stats['consolidation']['stored_memories'] == stats['consolidation']['total_consolidations'], \
        "Stored memories should match consolidation count"

    # Check consolidated memories directly
    assert len(consciousness.consolidated_memories) > 0, "Should have consolidated memories stored"

    print(f"\n  Sample consolidated memory (most recent):")
    recent_memory = consciousness.consolidated_memories[-1]
    print(f"    Patterns extracted: {len(recent_memory.patterns)}")
    print(f"    Quality learnings: {len(recent_memory.quality_learnings)}")
    print(f"    Creative associations: {len(recent_memory.creative_associations)}")
    print(f"    Cycles processed: {recent_memory.cycles_processed}")

    print("\n✓ Memory storage validated\n")
    return consciousness


def test_integration_with_cycle():
    """Test complete integration with consciousness cycle."""
    print("Test 4: Integration with Consciousness Cycle")
    print("=" * 60)

    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        circadian_period=15,
        circadian_enabled=True,
        consolidation_enabled=True
    )

    print("\n  Simulating realistic consciousness cycles...")

    cycle_with_consolidation = None

    for i in range(30):
        # Vary content to create interesting patterns
        if i % 3 == 0:
            response = f"Technical explanation {i}: Neural networks use backpropagation with gradient descent."
        elif i % 3 == 1:
            response = f"Conceptual insight {i}: Understanding emerges from pattern recognition across multiple domains."
        else:
            response = f"Practical application {i}: These principles apply to real-world problem solving."

        cycle = consciousness.consciousness_cycle(
            prompt=f"Query {i}",
            response=response,
            task_salience=0.6 + (i % 3) * 0.1
        )

        # Capture first consolidation for detailed inspection
        if cycle.consolidation_triggered and cycle_with_consolidation is None:
            cycle_with_consolidation = cycle

    # Validate we captured a consolidation
    assert cycle_with_consolidation is not None, "Should have at least one consolidation"

    print(f"\n  Consolidation Event Details:")
    print(f"    Cycle number: {consciousness.cycles.index(cycle_with_consolidation)}")
    print(f"    Circadian phase: {cycle_with_consolidation.circadian_phase}")
    print(f"    Metabolic state: {cycle_with_consolidation.metabolic_state.value}")
    print(f"    Quality score: {cycle_with_consolidation.quality_score.normalized:.3f}")
    print(f"    Consolidation triggered: {cycle_with_consolidation.consolidation_triggered}")

    # Validate consolidation fields are populated
    assert cycle_with_consolidation.consolidation_triggered == True
    assert cycle_with_consolidation.consolidated_memory is not None
    assert cycle_with_consolidation.circadian_phase == CircadianPhase.DEEP_NIGHT.value

    memory = cycle_with_consolidation.consolidated_memory
    print(f"\n  Consolidated Memory Content:")
    print(f"    Patterns: {len(memory.patterns)}")
    print(f"    Quality learnings: {len(memory.quality_learnings)}")
    print(f"    Associations: {len(memory.creative_associations)}")

    # Print sample pattern
    if memory.patterns:
        pattern = memory.patterns[0]
        print(f"\n  Sample Pattern:")
        print(f"    Type: {pattern.pattern_type}")
        print(f"    Strength: {pattern.strength:.3f}")
        print(f"    Examples: {len(pattern.examples)} cycles")

    print("\n✓ Integration with consciousness cycle validated\n")
    return consciousness


def test_consolidation_disabled():
    """Test that consolidation can be disabled."""
    print("Test 5: Consolidation Disable Flag")
    print("=" * 60)

    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        circadian_period=20,
        circadian_enabled=True,
        consolidation_enabled=False  # DISABLED
    )

    print("\n  Running 40 cycles with consolidation disabled...")

    consolidation_count = 0

    for i in range(40):
        cycle = consciousness.consciousness_cycle(
            prompt=f"Task {i}",
            response=f"Content {i}",
            task_salience=0.5
        )

        if cycle.consolidation_triggered:
            consolidation_count += 1

    print(f"\n  Consolidations triggered: {consolidation_count}")
    assert consolidation_count == 0, "No consolidations should trigger when disabled"

    # Check statistics
    stats = consciousness.get_statistics()
    assert stats['consolidation']['total_consolidations'] == 0

    print("\n✓ Consolidation disable flag validated\n")
    return consciousness


def run_all_tests():
    """Run all scheduled consolidation tests."""
    print("\n" + "=" * 60)
    print("SAGE Scheduled Memory Consolidation Tests - Session 50")
    print("=" * 60)
    print()

    try:
        # Test 1: Triggering during DEEP_NIGHT
        test_consolidation_triggering()

        # Test 2: Frequency control
        test_consolidation_frequency()

        # Test 3: Memory storage
        test_memory_storage()

        # Test 4: Integration
        test_integration_with_cycle()

        # Test 5: Disable flag
        test_consolidation_disabled()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Session 50 Achievement:")
        print("- DREAM consolidation integrated with circadian rhythm")
        print("- Consolidation triggers automatically during DEEP_NIGHT")
        print("- Memory patterns extracted and stored biologically-timed")
        print("- Frequency control prevents over-consolidation")
        print("- Complete integration with consciousness cycle")
        print("- Biological parallel: Memory consolidation during deep sleep")
        print()

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
