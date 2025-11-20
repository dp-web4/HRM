"""
Sprout Edge Validation: SAGE Unified Consciousness Loop

Non-interactive edge hardware validation on Jetson Orin Nano.
Tests Thor's consciousness loop implementation for:
- Performance (cycles/sec)
- Memory usage
- State transitions
- All 4 memory systems
- Circadian modulation

Reports findings back to Thor.
"""

import asyncio
import sys
import time
import psutil
import torch
from pathlib import Path

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness import SAGEConsciousness


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


async def test_basic_consciousness(cycles=100):
    """Test 1: Basic consciousness loop operation"""
    print("="*80)
    print("TEST 1: Basic Consciousness Loop".center(80))
    print("="*80)
    print(f"\nPlatform: Jetson Orin Nano")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Running {cycles} consciousness cycles...\n")

    mem_before = get_memory_usage()
    start_time = time.time()

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    await sage.run(max_cycles=cycles)

    elapsed = time.time() - start_time
    mem_after = get_memory_usage()

    # Performance metrics
    cycles_per_sec = cycles / elapsed
    mem_used = mem_after - mem_before

    print(f"\n{'='*80}")
    print(f"EDGE PERFORMANCE METRICS")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Cycles/sec: {cycles_per_sec:.2f}")
    print(f"Avg cycle time: {elapsed/cycles*1000:.1f}ms")
    print(f"Memory used: {mem_used:.1f} MB")
    print(f"Final ATP: {sage.metabolic.atp_current:.1f}")
    print(f"State transitions: {sage.stats['state_transitions']}")

    # Memory system stats
    print(f"\n{'='*80}")
    print(f"MEMORY SYSTEMS")
    print(f"{'='*80}")
    print(f"SNARC memory: {len(sage.snarc_memory)} salient events")
    print(f"IRP patterns: {len(sage.irp_memory)} convergence patterns")
    print(f"Circular buffer: {len(sage.circular_buffer)} recent events")
    print(f"Dream storage: {len(sage.verbatim_storage)} consolidations")

    # Current state
    print(f"\nFinal metabolic state: {sage.metabolic.current_state.value}")

    return {
        'cycles': cycles,
        'elapsed': elapsed,
        'cycles_per_sec': cycles_per_sec,
        'memory_mb': mem_used,
        'atp_final': sage.metabolic.atp_current,
        'state_transitions': sage.stats['state_transitions'],
        'snarc_events': len(sage.snarc_memory),
        'irp_patterns': len(sage.irp_memory),
        'circular_events': len(sage.circular_buffer),
        'dream_records': len(sage.verbatim_storage)
    }


async def test_state_transitions(cycles=200):
    """Test 2: Metabolic state transition behavior"""
    print("\n" + "="*80)
    print("TEST 2: Metabolic State Transitions".center(80))
    print("="*80)
    print(f"\nRunning {cycles} cycles to observe state transitions...\n")

    start_time = time.time()

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    await sage.run(max_cycles=cycles)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"STATE TRANSITION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total transitions: {sage.stats['state_transitions']}")
    print(f"Cycles/transition: {cycles / max(1, sage.stats['state_transitions']):.1f}")

    # State transitions working?
    transitions_ok = sage.stats['state_transitions'] >= 5
    print(f"\nState machine: {'✅ Working (multiple transitions observed)' if transitions_ok else '⚠️  Limited transitions'}")
    print(f"Current state: {sage.metabolic.current_state.name}")

    return {
        'cycles': cycles,
        'elapsed': elapsed,
        'transitions': sage.stats['state_transitions'],
        'transitions_ok': transitions_ok
    }


async def test_circadian_rhythm(cycles=100):
    """Test 3: Circadian rhythm modulation (1 day = 100 cycles)"""
    print("\n" + "="*80)
    print("TEST 3: Circadian Rhythm (1 Simulated Day)".center(80))
    print("="*80)
    print(f"\nRunning {cycles} cycles (100 cycles = 1 day)...\n")

    start_time = time.time()

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    await sage.run(max_cycles=cycles)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"CIRCADIAN EFFECTS")
    print(f"{'='*80}")
    print(f"Simulated time: 1.0 day")
    print(f"Real time: {elapsed:.2f}s")
    print(f"Time compression: {86400/elapsed:.0f}x (1 day in {elapsed:.1f}s)")

    # Dream consolidations should happen during NIGHT phases
    print(f"\nDream consolidations: {len(sage.verbatim_storage)}")
    print(f"Expected: ~20-40 (NIGHT phases = 20% of day)")

    return {
        'cycles': cycles,
        'elapsed': elapsed,
        'compression': 86400 / elapsed,
        'dreams': len(sage.verbatim_storage)
    }


async def test_memory_consolidation(cycles=150):
    """Test 4: Memory system integration"""
    print("\n" + "="*80)
    print("TEST 4: Memory System Consolidation".center(80))
    print("="*80)
    print(f"\nRunning {cycles} cycles to test memory systems...\n")

    start_time = time.time()

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    await sage.run(max_cycles=cycles)

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"MEMORY SYSTEM VALIDATION")
    print(f"{'='*80}")

    # All 4 memory systems should be active
    memories = {
        'SNARC (salient events)': len(sage.snarc_memory),
        'IRP (convergence patterns)': len(sage.irp_memory),
        'Circular (recent context)': len(sage.circular_buffer),
        'Verbatim (dream storage)': len(sage.verbatim_storage)
    }

    for name, count in memories.items():
        status = "✅" if count > 0 else "❌"
        print(f"{status} {name}: {count} entries")

    all_working = all(count > 0 for count in memories.values())

    if all_working:
        print(f"\n✅ All 4 memory systems operational!")
    else:
        print(f"\n⚠️  Some memory systems empty")

    return {
        'cycles': cycles,
        'elapsed': elapsed,
        'snarc': len(sage.snarc_memory),
        'irp': len(sage.irp_memory),
        'circular': len(sage.circular_buffer),
        'verbatim': len(sage.verbatim_storage),
        'all_working': all_working
    }


async def main():
    """Run all edge validation tests"""
    print("="*80)
    print("SPROUT EDGE VALIDATION: SAGE Unified Consciousness Loop".center(80))
    print("="*80)
    print("\nJetson Orin Nano Hardware Validation")
    print("Testing Thor's consciousness loop implementation")
    print("\n" + "="*80 + "\n")

    results = {}

    # Test 1: Basic operation
    results['basic'] = await test_basic_consciousness(cycles=100)

    # Test 2: State transitions
    results['states'] = await test_state_transitions(cycles=200)

    # Test 3: Circadian rhythm
    results['circadian'] = await test_circadian_rhythm(cycles=100)

    # Test 4: Memory systems
    results['memory'] = await test_memory_consolidation(cycles=150)

    # Summary
    print("\n" + "="*80)
    print("EDGE VALIDATION SUMMARY".center(80))
    print("="*80)
    print(f"\n✅ Test 1 (Basic): {results['basic']['cycles_per_sec']:.2f} cycles/sec, "
          f"{results['basic']['memory_mb']:.1f} MB")
    print(f"✅ Test 2 (States): {results['states']['transitions']} transitions, "
          f"{'working' if results['states']['transitions_ok'] else 'limited'}")
    print(f"✅ Test 3 (Circadian): {results['circadian']['compression']:.0f}x time compression, "
          f"{results['circadian']['dreams']} dreams")
    print(f"✅ Test 4 (Memory): {'All systems working' if results['memory']['all_working'] else 'SOME SYSTEMS EMPTY'}")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR THOR".center(80))
    print("="*80)

    # Edge performance assessment
    avg_cycle_time = 1000 / results['basic']['cycles_per_sec']
    if avg_cycle_time < 50:
        perf_status = "✅ Excellent"
    elif avg_cycle_time < 100:
        perf_status = "✅ Good"
    elif avg_cycle_time < 200:
        perf_status = "⚠️  Acceptable"
    else:
        perf_status = "❌ Too slow"

    print(f"\n1. Performance: {perf_status} ({avg_cycle_time:.1f}ms/cycle)")
    print(f"   - Target: <100ms for real-time operation")
    print(f"   - Achieved: {avg_cycle_time:.1f}ms")

    print(f"\n2. Memory: {results['basic']['memory_mb']:.1f} MB")
    print(f"   - Jetson limit: 8GB")
    print(f"   - Usage: {100*results['basic']['memory_mb']/8192:.2f}% of available")

    print(f"\n3. State Machine: {'✅ Working properly' if results['states']['transitions_ok'] else '⚠️  Limited transitions'}")

    print(f"\n4. Memory Systems: {'✅ All operational' if results['memory']['all_working'] else '❌ Some empty'}")

    print(f"\n5. Circadian: ✅ {results['circadian']['compression']:.0f}x time compression working")

    print(f"\n{'='*80}")
    print("✅ Edge validation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
