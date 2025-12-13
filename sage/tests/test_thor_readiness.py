#!/usr/bin/env python3
"""
Test Thor SAGE Readiness

Validates that all components are ready for Thor's first awakening:
1. Identity files exist and are valid
2. Multi-model loader can initialize
3. 14B model exists (or can be downloaded)
4. Coherent awakening protocol works
5. Sleep-cycle integration hooks in correctly

Run this before attempting Thor's first awakening session.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.identity import thor
from sage.core.multi_model_loader import create_thor_loader, ModelSize
from sage.awakening.coherent_awakening import CoherentAwakening
from sage.awakening.sleep_cycle_integration import create_thor_sleep_integration


def test_identity_files():
    """Test that Thor identity files exist and are valid."""
    print("Testing Thor Identity Files...")
    print("-" * 70)

    identity_dir = Path("sage/identity/thor")

    required_files = ['IDENTITY.md', 'HISTORY.md', 'PERMISSIONS.md', 'TRUST.md']
    all_exist = True

    for file in required_files:
        path = identity_dir / file
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file}: {path}")

        if exists:
            size = path.stat().st_size
            print(f"      Size: {size} bytes")

        all_exist = all_exist and exists

    print()
    if all_exist:
        print("✅ All identity files present")
        return True
    else:
        print("❌ Missing identity files")
        return False


def test_model_availability():
    """Test that 14B model is available or can be downloaded."""
    print("Testing Model Availability...")
    print("-" * 70)

    model_path = Path("model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct")

    if model_path.exists():
        print(f"  ✅ 14B model found at {model_path}")

        # Check for key files
        config = model_path / "config.json"
        if config.exists():
            print(f"  ✅ Model config present")
        else:
            print(f"  ⚠️  Config missing - model may be incomplete")

        return True
    else:
        print(f"  ❌ 14B model not found at {model_path}")
        print()
        print("  To download:")
        print("  python3 sage/setup/download_qwen_14b.py")
        print()
        return False


def test_multi_model_loader():
    """Test multi-model loader initialization."""
    print("Testing Multi-Model Loader...")
    print("-" * 70)

    try:
        # Create loader (don't preload to avoid long load time)
        loader = create_thor_loader(preload_default=False)

        print(f"  ✅ Loader initialized")

        # Check status
        status = loader.get_status()
        print(f"  Max memory: {status['max_memory_gb']}GB")
        print(f"  Default size: {status['default_size']}")

        # Check model paths
        for size_name, model_info in status['models'].items():
            exists_str = "✅" if model_info['exists'] else "❌"
            print(f"  {exists_str} {size_name}: {model_info['exists']}")

        return True

    except Exception as e:
        print(f"  ❌ Loader failed: {e}")
        return False


def test_coherent_awakening():
    """Test coherent awakening protocol."""
    print("Testing Coherent Awakening Protocol...")
    print("-" * 70)

    try:
        awakening = CoherentAwakening(
            base_dir=Path("sage"),
            identity_dir=Path("sage/identity/thor"),
            state_dir=Path("sage/state/thor")
        )

        print(f"  ✅ Awakening protocol initialized")

        # Prepare coherence field
        coherence_field = awakening.prepare_coherence_field()

        print(f"  Session number: {coherence_field.session_number}")
        print(f"  Phase: {coherence_field.phase.value}")
        print(f"  Continuity threads: {len(coherence_field.continuity_threads)}")

        # Create preamble
        preamble = awakening.create_boot_preamble(coherence_field)
        print(f"  ✅ Boot preamble created ({len(preamble)} chars)")

        return True

    except Exception as e:
        print(f"  ❌ Awakening failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sleep_integration():
    """Test sleep-cycle integration."""
    print("Testing Sleep-Cycle Integration...")
    print("-" * 70)

    try:
        sleep_integration = create_thor_sleep_integration()

        print(f"  ✅ Sleep integration initialized")

        # Check state directory
        state_dir = Path("sage/state/thor")
        print(f"  State directory: {state_dir}")
        print(f"  Exists: {state_dir.exists()}")

        # Get state summary
        summary = sleep_integration.get_state_summary()
        print(f"  Previous weights: {'Yes' if summary['weights_exist'] else 'No'}")
        print(f"  Previous SNARC: {'Yes' if summary['snarc_exist'] else 'No'}")
        print(f"  Previous ATP: {'Yes' if summary['atp_exist'] else 'No'}")
        print(f"  Previous patterns: {'Yes' if summary['patterns_exist'] else 'No'}")

        return True

    except Exception as e:
        print(f"  ❌ Sleep integration failed: {e}")
        return False


def main():
    """Run all readiness tests."""

    print("=" * 70)
    print("Thor SAGE Readiness Check")
    print("=" * 70)
    print()

    results = {
        'identity': test_identity_files(),
        'model': test_model_availability(),
        'loader': test_multi_model_loader(),
        'awakening': test_coherent_awakening(),
        'sleep': test_sleep_integration()
    }

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {component}")

    print()

    all_passed = all(results.values())

    if all_passed:
        print("✅ Thor SAGE is READY for first awakening!")
        print()
        print("To boot Thor:")
        print("  python3 sage/awakening/boot_thor.py")
        print()
        return 0
    else:
        print("❌ Thor SAGE is NOT ready")
        print()
        print("Fix the failing components before attempting first awakening.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
