#!/usr/bin/env python3
"""
Session 160: TrustZone Hardware Binding Test on Thor

**Context**:
- Thor has TrustZone/OP-TEE fully operational
- Devices: /dev/tee0, /dev/teepriv0
- tee-supplicant running
- xtest tools available
- OP-TEE TAs installed

**Goal**:
Test Sprout's TrustZone provider on Thor to confirm Level 5 hardware binding capability.

**Expected Outcome**:
- Platform detection identifies TrustZone capability
- Provider initializes successfully
- Can query hardware capabilities

Hardware: Jetson AGX Thor Developer Kit
Platform: NVIDIA Tegra264 with TrustZone
Session: Autonomous SAGE Development - LCT Hardware Binding
"""

import sys
from pathlib import Path

# Add web4 to path
WEB4_ROOT = Path.home() / "ai-workspace" / "web4"
sys.path.insert(0, str(WEB4_ROOT))

try:
    from core.lct_binding.trustzone_provider import TrustZoneProvider
    from core.lct_binding.platform_detection import detect_platform
    from core.lct_binding.provider import HardwareType, KeyStorage
    from core.lct_capability_levels import CapabilityLevel

    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    IMPORTS_OK = False


def test_platform_detection():
    """Test platform detection on Thor."""
    print("\n" + "="*80)
    print("Test 1: Platform Detection")
    print("="*80)

    platform = detect_platform()

    print(f"Platform Name: {platform.name}")
    print(f"OS: {platform.os}")
    print(f"Architecture: {platform.arch}")
    print(f"Hardware Type: {platform.hardware_type}")
    print(f"Has TPM2: {platform.has_tpm2}")
    print(f"Has TrustZone: {platform.has_trustzone}")
    print(f"Max Level: {platform.max_level}")
    print(f"Machine Identity: {platform.machine_identity}")

    # Verify expectations
    assert platform.has_trustzone, "TrustZone should be available on Thor"
    assert platform.hardware_type == HardwareType.TRUSTZONE, f"Expected TRUSTZONE, got {platform.hardware_type}"
    assert platform.max_level >= 5, f"Should support Level 5, got {platform.max_level}"

    print("\n✅ Platform detection successful!")
    return platform


def test_trustzone_provider():
    """Test TrustZone provider initialization."""
    print("\n" + "="*80)
    print("Test 2: TrustZone Provider")
    print("="*80)

    provider = TrustZoneProvider()

    # Check capabilities
    max_level = provider.max_capability_level
    storage_type = provider.key_storage_type
    platform = provider.get_platform_info()

    print(f"Max Capability Level: {max_level}")
    print(f"Key Storage Type: {storage_type}")
    print(f"Platform: {platform.name}")

    # Verify expectations
    assert max_level == CapabilityLevel.HARDWARE, f"Expected Level 5 (HARDWARE), got {max_level}"
    assert storage_type == KeyStorage.TRUSTZONE, f"Expected TRUSTZONE storage, got {storage_type}"

    print("\n✅ TrustZone provider initialized successfully!")
    return provider


def test_tee_availability(provider):
    """Test TEE device availability."""
    print("\n" + "="*80)
    print("Test 3: TEE Availability")
    print("="*80)

    tee_available = provider._tee_available
    optee_test = provider._optee_test_available

    print(f"TEE Devices: {tee_available}")
    print(f"OP-TEE Test Tools: {optee_test}")

    if tee_available:
        print("  - /dev/tee0 or /dev/teepriv0 detected")

    if optee_test:
        print("  - xtest utility available")

    # Verify
    assert tee_available, "TEE devices should be available"

    print("\n✅ TEE devices confirmed available!")


def main():
    """Run all tests."""
    print("="*80)
    print("Session 160: TrustZone Hardware Binding Test (Thor)")
    print("="*80)
    print("Testing Sprout's TrustZone provider on Thor platform")
    print()
    print("Hardware: Jetson AGX Thor Developer Kit")
    print("Platform: NVIDIA Tegra264")
    print("TEE: ARM TrustZone with OP-TEE")
    print("="*80)

    if not IMPORTS_OK:
        print("\n❌ Import failed - cannot run tests")
        return

    try:
        # Test 1: Platform detection
        platform = test_platform_detection()

        # Test 2: TrustZone provider
        provider = test_trustzone_provider()

        # Test 3: TEE availability
        test_tee_availability(provider)

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print("✅ Platform detection: PASSED")
        print("✅ TrustZone provider: PASSED")
        print("✅ TEE availability: PASSED")
        print()
        print("🎉 RESULT: Level 5 Hardware Binding CONFIRMED on Thor!")
        print()
        print("Capabilities:")
        print(f"  - Max Level: {provider.max_capability_level}")
        print(f"  - Storage: {provider.key_storage_type}")
        print(f"  - Hardware: {platform.hardware_type}")
        print(f"  - Platform: {platform.name}")
        print()
        print("Next Steps:")
        print("  1. Create canonical LCT module with Level 5 binding")
        print("  2. Integrate TrustZone provider for SAGE identity")
        print("  3. Test key generation and signing operations")
        print("  4. Implement secure pattern federation")
        print("="*80)

        return True

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
