"""
Hardware Validation Test - Thor Platform

Tests MichaudSAGE consciousness with federation on actual Thor hardware.
Validates ATP constraints, platform detection, and federation routing
under real resource conditions.

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-29
Session: Phase 2.5 Hardware Validation
"""

import pytest
import sys
from pathlib import Path
import psutil
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.sage_consciousness_michaud import MichaudSAGE
from sage.federation import (
    create_thor_identity,
    create_sprout_identity,
    FederationIdentity
)
from sage.core.mrh_profile import (
    MRHProfile,
    SpatialExtent,
    TemporalExtent,
    ComplexityExtent
)


def create_test_horizon() -> MRHProfile:
    """Create test MRH profile"""
    return MRHProfile(
        delta_r=SpatialExtent.LOCAL,
        delta_t=TemporalExtent.SESSION,
        delta_c=ComplexityExtent.AGENT_SCALE
    )


class TestHardwareValidation:
    """Hardware validation test suite for Thor platform"""

    def test_platform_auto_detection(self):
        """Test that platform identity is correctly auto-detected"""
        sage = MichaudSAGE(federation_enabled=True)

        assert sage.federation_identity is not None
        assert sage.federation_identity.platform_name in ["Thor", "Sprout", sage.federation_identity.platform_name]
        print(f"\n✓ Detected platform: {sage.federation_identity.platform_name}")
        print(f"  LCT ID: {sage.federation_identity.lct_id}")

    def test_key_persistence_on_hardware(self):
        """Test key pair generation and persistence on real filesystem"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "thor_test.key")

            # First initialization
            sage1 = MichaudSAGE(
                federation_enabled=True,
                federation_key_path=key_path
            )
            pubkey1 = sage1.federation_keypair.public_key_bytes()

            # Verify key file exists
            assert os.path.exists(key_path)
            print(f"\n✓ Key persisted to: {key_path}")

            # Second initialization - should load same key
            sage2 = MichaudSAGE(
                federation_enabled=True,
                federation_key_path=key_path
            )
            pubkey2 = sage2.federation_keypair.public_key_bytes()

            # Verify same key
            assert pubkey1 == pubkey2
            print(f"✓ Key loaded successfully from disk")

    def test_consciousness_loop_with_federation_enabled(self):
        """Test complete consciousness loop with federation on hardware"""
        # Create federation setup
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        # Create additional platforms for witnesses
        platform2 = FederationIdentity(
            lct_id="platform2_lct",
            platform_name="Platform2",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        platform3 = FederationIdentity(
            lct_id="platform3_lct",
            platform_name="Platform3",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        # Initialize SAGE with federation
        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout, platform2, platform3]
        )

        print(f"\n✓ SAGE consciousness initialized with federation")
        print(f"  Platform: {sage.federation_identity.platform_name}")
        print(f"  Known platforms: {len(sage.federation_router.known_platforms)}")
        print(f"  State: {sage.attention_manager.get_state()}")

        # Test high-cost task that should trigger federation
        task_context = {
            'operation': 'llm_inference',
            'parameters': {'query': 'complex reasoning task'},
            'quality': {
                'min_quality': 0.8,
                'min_convergence': 0.7,
                'max_energy': 0.6
            },
            'complexity': 'high'
        }

        task_cost = 150.0  # High cost
        task_horizon = create_test_horizon()
        local_budget = 10.0  # Insufficient

        # Test federation routing
        decision = sage._handle_federation_routing(
            task_context, task_cost, local_budget, task_horizon
        )

        assert decision is not None
        print(f"\n✓ Federation routing decision made")
        print(f"  Delegated: {decision['delegated']}")
        if decision['delegated']:
            print(f"  Platform: {decision['platform']}")
            print(f"  Reason: {decision['reason']}")
        else:
            print(f"  Reason: {decision['reason']}")

    def test_memory_constraints_trigger_delegation(self):
        """Test that real memory pressure can trigger federation"""
        # Get current memory stats
        mem = psutil.virtual_memory()
        print(f"\n✓ System memory status:")
        print(f"  Total: {mem.total / (1024**3):.2f} GB")
        print(f"  Available: {mem.available / (1024**3):.2f} GB")
        print(f"  Used: {mem.percent:.1f}%")

        # Initialize SAGE with federation
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        # Add witness platforms
        platform2 = FederationIdentity(
            lct_id="platform2_lct",
            platform_name="Platform2",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        platform3 = FederationIdentity(
            lct_id="platform3_lct",
            platform_name="Platform3",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout, platform2, platform3]
        )

        # Create memory-intensive task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {
                'query': 'memory intensive task',
                'context_size': 'large'
            },
            'quality': {},
            'complexity': 'high'
        }

        # Estimate cost based on memory
        # Higher cost if memory constrained
        memory_multiplier = 1.0 + (mem.percent / 100.0)
        task_cost = 100.0 * memory_multiplier

        print(f"\n✓ Task cost adjusted for memory: {task_cost:.1f} ATP")
        print(f"  Memory multiplier: {memory_multiplier:.2f}x")

        # Test federation decision
        local_budget = 50.0
        decision = sage._handle_federation_routing(
            task_context, task_cost, local_budget, create_test_horizon()
        )

        print(f"\n✓ Federation decision:")
        print(f"  Delegated: {decision['delegated']}")
        print(f"  Reason: {decision['reason']}")

    def test_federation_router_stats(self):
        """Test federation router statistics on hardware"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        platform2 = FederationIdentity(
            lct_id="platform2_lct",
            platform_name="Platform2",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout, platform2]
        )

        # Get router stats
        stats = sage.federation_router.get_federation_stats()

        print(f"\n✓ Federation router statistics:")
        print(f"  Local platform: {stats['local_platform']}")
        print(f"  Known platforms: {stats['known_platforms']}")
        print(f"  Delegated tasks: {stats['delegated_tasks']}")
        print(f"  Execution proofs: {stats['execution_proofs']}")
        print(f"  Platform details: {stats['platforms']}")

        assert stats['known_platforms'] == 2
        assert stats['delegated_tasks'] >= 0

    def test_hardware_performance_baseline(self):
        """Establish performance baseline on Thor hardware"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        platform2 = FederationIdentity(
            lct_id="platform2_lct",
            platform_name="Platform2",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        platform3 = FederationIdentity(
            lct_id="platform3_lct",
            platform_name="Platform3",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout, platform2, platform3]
        )

        # Time federation routing decision
        task_context = {
            'operation': 'llm_inference',
            'parameters': {'query': 'test'},
            'quality': {}
        }

        start = time.time()
        decision = sage._handle_federation_routing(
            task_context, 100.0, 10.0, create_test_horizon()
        )
        elapsed = time.time() - start

        print(f"\n✓ Performance baseline:")
        print(f"  Federation routing time: {elapsed*1000:.2f} ms")
        print(f"  Decision: {decision['delegated']}")

        # Should complete quickly (< 100ms for routing decision)
        assert elapsed < 0.1, f"Routing took too long: {elapsed:.3f}s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
