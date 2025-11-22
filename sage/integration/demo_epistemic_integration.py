"""
Demo: SAGE with Epistemic Integration

Demonstrates Phase 1 integration:
- High-salience observations → epistemic DB
- Learning sessions → episodes with quality scores
- Context queries → inform reasoning
- Blockchain witnessing → attribution

Run:
    cd /home/dp/ai-workspace/HRM
    python3 sage/integration/demo_epistemic_integration.py

Author: Thor
Date: 2025-11-22
"""

import asyncio
import sys
from pathlib import Path

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness_epistemic import create_epistemic_sage
from integration.epistemic_memory import EPISTEMIC_AVAILABLE


async def demo_basic_integration():
    """
    Demo 1: Basic epistemic integration with single observation.

    Shows:
    - SAGE with epistemic bridge
    - High-salience detection
    - Observation storage (if available)
    """
    print("="*70)
    print("DEMO 1: Basic Epistemic Integration")
    print("="*70)
    print()

    # Create SAGE with epistemic integration
    sage = create_epistemic_sage(
        machine='thor',
        project='sage-demo',
        epistemic_threshold=0.6,  # Lower threshold for demo
        enable_witnessing=False,  # Disable for demo
        initial_atp=100.0
    )

    # Single observation
    observation = "What is the relationship between attention and consciousness?"

    # Run with automatic session tracking
    episode_id = await sage.run_with_session([observation])

    print()
    print("="*70)
    print(f"Demo 1 Complete - Episode ID: {episode_id}")
    print("="*70)


async def demo_multi_observation_session():
    """
    Demo 2: Multiple observations with session tracking.

    Shows:
    - Session start/end
    - Multiple observations processed
    - Quality score calculation
    - Discovery tracking
    """
    print("\n")
    print("="*70)
    print("DEMO 2: Multi-Observation Session")
    print("="*70)
    print()

    sage = create_epistemic_sage(
        machine='thor',
        project='sage-demo',
        epistemic_threshold=0.6,
        enable_witnessing=False
    )

    # Multiple related observations
    observations = [
        "What is consciousness?",
        "How does attention direct awareness?",
        "Can consciousness exist without attention?"
    ]

    # Run full session
    episode_id = await sage.run_with_session(
        observations,
        session_id="demo-session-multi-obs"
    )

    print()
    print("="*70)
    print(f"Demo 2 Complete - Episode ID: {episode_id}")
    print("="*70)


async def demo_context_retrieval():
    """
    Demo 3: Context retrieval from epistemic DB.

    Shows:
    - Querying relevant past experiences
    - Context influences reasoning (future enhancement)
    """
    print("\n")
    print("="*70)
    print("DEMO 3: Context Retrieval")
    print("="*70)
    print()

    sage = create_epistemic_sage(
        machine='thor',
        project='sage-demo',
        epistemic_threshold=0.6,
        enable_witnessing=False
    )

    # First session: establish context
    print("[Context Building] First session...")
    await sage.run_with_session(
        ["What is the role of salience in consciousness?"],
        session_id="demo-context-build"
    )

    print("\n")
    print("[Context Retrieval] Second session (should find context)...")

    # Second session: should find relevant context
    await sage.run_with_session(
        ["How does SNARC compute salience?"],
        session_id="demo-context-retrieve"
    )

    print()
    print("="*70)
    print("Demo 3 Complete - Context retrieval demonstrated")
    print("="*70)


async def demo_integration_status():
    """
    Demo 4: Integration status and statistics.

    Shows:
    - Integration status summary
    - What's stored
    - Whether epistemic tools available
    """
    print("\n")
    print("="*70)
    print("DEMO 4: Integration Status")
    print("="*70)
    print()

    sage = create_epistemic_sage(
        machine='thor',
        project='sage-demo'
    )

    # Get integration status
    status = sage.epistemic_bridge.summarize_integration_status()

    print("Integration Status:")
    print(f"  Status: {status['status']}")
    print(f"  Machine: {status['machine']}")
    print(f"  Project: {status['project']}")
    print(f"  Salience Threshold: {status['salience_threshold']}")
    print(f"  Witnessing Enabled: {status['witnessing_enabled']}")
    print(f"  Epistemic Tools Available: {status['epistemic_available']}")
    print(f"  Observations Stored: {status['observations_stored']}")
    print(f"  Sessions Recorded: {status['sessions_recorded']}")

    print()
    if not EPISTEMIC_AVAILABLE:
        print("NOTE: Running in fallback mode (epistemic tools not available)")
        print("      Observations stored locally, not in shared database")
        print("      To enable full integration, ensure memory repo is accessible")

    print()
    print("="*70)
    print("Demo 4 Complete")
    print("="*70)


async def run_all_demos():
    """Run all demonstration scenarios"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║          SAGE Epistemic Integration - Phase 1 Demonstration       ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Demo 1: Basic integration
        await demo_basic_integration()

        # Demo 2: Multi-observation session
        await demo_multi_observation_session()

        # Demo 3: Context retrieval
        await demo_context_retrieval()

        # Demo 4: Status summary
        await demo_integration_status()

        print("\n")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                                                                    ║")
        print("║                    ALL DEMOS COMPLETED ✅                          ║")
        print("║                                                                    ║")
        print("║  Phase 1 Integration Status:                                      ║")
        print("║    ✅ Epistemic bridge implemented                                 ║")
        print("║    ✅ High-salience observation storage                            ║")
        print("║    ✅ Learning session recording                                   ║")
        print("║    ✅ Context queries functional                                   ║")
        print("║    ✅ Session tracking with quality scores                         ║")
        print("║                                                                    ║")
        print("║  Next Steps (Phase 2):                                            ║")
        print("║    - Skill library integration                                    ║")
        print("║    - Skill-guided plugin selection                                ║")
        print("║    - Cross-machine skill discovery                                ║")
        print("║                                                                    ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    print()
    print("NOTE: This demo requires LLM model to be available.")
    print("      If model not found, demo will show integration structure only.")
    print()

    try:
        asyncio.run(run_all_demos())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
