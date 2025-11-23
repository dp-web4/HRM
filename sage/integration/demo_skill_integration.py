"""
Demo: SAGE with Skill Integration

Demonstrates Phase 2 integration:
- Skill queries for applicable skills
- Skill-guided IRP execution
- Pattern discovery from successful executions
- New skill creation

Run:
    cd /home/dp/ai-workspace/HRM
    python3 sage/integration/demo_skill_integration.py

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

from core.sage_consciousness_skills import create_skill_sage
from integration.skill_learning import SKILLS_AVAILABLE


async def demo_skill_application():
    """
    Demo 1: Applying skills from library.

    Shows:
    - Skill query for situation
    - Skill-guided execution
    - Application tracking
    """
    print("="*70)
    print("DEMO 1: Skill Application")
    print("="*70)
    print()

    sage = create_skill_sage(
        machine='thor',
        project='sage-demo',
        enable_skill_application=True,
        enable_skill_discovery=False,  # Disable for focused demo
        epistemic_threshold=0.6,
        enable_witnessing=False
    )

    # Observation that should match a skill (if any exist)
    observations = [
        "How should I approach designing an attention mechanism?"
    ]

    result = await sage.run_with_skill_learning(
        observations,
        session_id="demo-skill-app"
    )

    print()
    print("="*70)
    print(f"Demo 1 Complete")
    print(f"Skills applied: {result['skills_applied']}")
    print("="*70)


async def demo_pattern_discovery():
    """
    Demo 2: Discovering patterns from successful executions.

    Shows:
    - Successful execution tracking
    - Pattern recording
    - Repeat pattern detection
    """
    print("\n")
    print("="*70)
    print("DEMO 2: Pattern Discovery")
    print("="*70)
    print()

    sage = create_skill_sage(
        machine='thor',
        project='sage-demo',
        enable_skill_application=False,  # Disable for focused demo
        enable_skill_discovery=True,
        epistemic_threshold=0.6,
        enable_witnessing=False
    )

    # Similar observations to create repeating pattern
    observations = [
        "What is consciousness?",
        "What is awareness?",
        "What is sentience?"
    ]

    result = await sage.run_with_skill_learning(
        observations,
        session_id="demo-pattern-disc"
    )

    print()
    print("="*70)
    print(f"Demo 2 Complete")
    print(f"Patterns recorded: {result['patterns_recorded']}")
    print("="*70)


async def demo_skill_creation():
    """
    Demo 3: Creating new skill from repeated pattern.

    Shows:
    - Pattern repetition threshold
    - Skill creation
    - Blockchain witnessing
    """
    print("\n")
    print("="*70)
    print("DEMO 3: Skill Creation (Simulated)")
    print("="*70)
    print()

    sage = create_skill_sage(
        machine='thor',
        project='sage-demo',
        enable_skill_discovery=True,
        epistemic_threshold=0.6,
        enable_witnessing=False
    )

    # Note: This would require 3+ successful executions of same pattern
    # For demo, we just show the tracking mechanism

    print("To create a skill, a pattern must repeat 3+ times successfully.")
    print("The skill manager tracks patterns automatically:")
    print()

    # Show pattern tracking
    from integration.skill_learning import DiscoveredPattern

    example_pattern = DiscoveredPattern(
        pattern_id="example123",
        strategy="Focused llm_reasoning processing with rapid convergence",
        plugins_sequence=['llm_reasoning'],
        convergence_profile={'iterations': 3, 'final_energy': 0.2},
        success_count=3,
        failure_count=0,
        average_quality=0.85,
        contexts=['What is X?', 'What is Y?', 'What is Z?']
    )

    print(f"Example Pattern:")
    print(f"  ID: {example_pattern.pattern_id}")
    print(f"  Strategy: {example_pattern.strategy}")
    print(f"  Success: {example_pattern.success_count}/{example_pattern.success_count + example_pattern.failure_count}")
    print(f"  Quality: {example_pattern.average_quality:.2f}")
    print(f"  Contexts: {len(example_pattern.contexts)} similar situations")
    print()
    print(f"When success_count >= 3: Automatic skill creation!")

    print()
    print("="*70)
    print("Demo 3 Complete")
    print("="*70)


async def demo_full_integration():
    """
    Demo 4: Full skill integration (discovery + application).

    Shows:
    - Complete skill learning cycle
    - Statistics tracking
    - Session summary
    """
    print("\n")
    print("="*70)
    print("DEMO 4: Full Integration")
    print("="*70)
    print()

    sage = create_skill_sage(
        machine='thor',
        project='sage-demo',
        enable_skill_application=True,
        enable_skill_discovery=True,
        epistemic_threshold=0.6,
        enable_witnessing=False
    )

    observations = [
        "How does attention direct consciousness?",
        "What role does salience play?",
        "Can consciousness exist without attention?"
    ]

    result = await sage.run_with_skill_learning(
        observations,
        session_id="demo-full"
    )

    print()
    print("="*70)
    print(f"Demo 4 Complete - Full Integration")
    print(f"  Skills applied: {result['skills_applied']}")
    print(f"  Patterns recorded: {result['patterns_recorded']}")
    print(f"  Total skills created: {result['total_skills_created']}")
    print(f"  Average quality: {result['average_quality']:.2f}")
    print("="*70)


async def demo_skill_statistics():
    """
    Demo 5: Skill manager statistics.

    Shows:
    - Skill application tracking
    - Pattern discovery counts
    - Quality metrics
    """
    print("\n")
    print("="*70)
    print("DEMO 5: Skill Statistics")
    print("="*70)
    print()

    sage = create_skill_sage(
        machine='thor',
        project='sage-demo'
    )

    # Get current statistics
    stats = sage.skill_manager.get_skill_statistics()

    print("Skill Manager Statistics:")
    print(f"  Applications: {stats['applications_count']}")
    print(f"  Successful: {stats['successful_applications']}")
    print(f"  Discovered patterns: {stats['discovered_patterns']}")
    print(f"  Skills created: {stats['skills_created']}")
    print(f"  Average quality: {stats['average_quality']:.2f}")

    print()
    if not SKILLS_AVAILABLE:
        print("NOTE: Running in fallback mode (skill tools not available)")
        print("      Skills stored locally, not in shared database")
        print("      To enable full integration, ensure memory repo is accessible")

    print()
    print("="*70)
    print("Demo 5 Complete")
    print("="*70)


async def run_all_demos():
    """Run all demonstration scenarios"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║         SAGE Skill Integration - Phase 2 Demonstration            ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        # Demo 1: Skill application
        await demo_skill_application()

        # Demo 2: Pattern discovery
        await demo_pattern_discovery()

        # Demo 3: Skill creation
        await demo_skill_creation()

        # Demo 4: Full integration
        await demo_full_integration()

        # Demo 5: Statistics
        await demo_skill_statistics()

        print("\n")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                                                                    ║")
        print("║                    ALL DEMOS COMPLETED ✅                          ║")
        print("║                                                                    ║")
        print("║  Phase 2 Integration Status:                                      ║")
        print("║    ✅ Skill query manager implemented                              ║")
        print("║    ✅ Skill-to-IRP guidance translation                            ║")
        print("║    ✅ Pattern discovery from executions                            ║")
        print("║    ✅ Automatic skill creation                                     ║")
        print("║    ✅ Skill application tracking                                   ║")
        print("║    ✅ Session statistics                                           ║")
        print("║                                                                    ║")
        print("║  Skill Learning Cycle:                                            ║")
        print("║    1. Query applicable skills                                     ║")
        print("║    2. Apply skill-guided execution                                ║")
        print("║    3. Record successful patterns                                  ║")
        print("║    4. Create skills from repeated patterns                        ║")
        print("║    5. Share skills across machines (via epistemic DB)             ║")
        print("║                                                                    ║")
        print("║  Next Steps (Phase 3):                                            ║")
        print("║    - Replace blockchain witnessing stubs                          ║")
        print("║    - Implement Merkle batch witnessing                            ║")
        print("║    - Add security validation                                      ║")
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
    print("NOTE: This demo shows skill integration structure.")
    print("      Full skill discovery requires repeated pattern execution.")
    print("      LLM model required for actual consciousness loop.")
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
