#!/usr/bin/env python3
"""
Standalone SAGE-Totality runner without external dependencies.
Uses only Python standard library.
"""

import sys
import json
from pathlib import Path

# Add totality_min to path
sys.path.insert(0, str(Path(__file__).parent / "totality_min"))

from totality_core import TotalityStore, Scheme, Canvas
from transforms import ContextTransform, SemanticTransform

class StandaloneSageTotality:
    """Run SAGE-Totality operations without web service."""
    
    def __init__(self):
        self.store = TotalityStore()
        self.context_transform = ContextTransform()
        self.semantic_transform = SemanticTransform()
        self._seed_data()
    
    def _seed_data(self):
        """Initialize with basic schemes and canvas."""
        # Add basic schemes
        grasp = self.store.add_scheme(
            "grasp(object)",
            [{"name": "object", "value": None}],
            activation=0.7
        )
        place = self.store.add_scheme(
            "place(object, location)",
            [{"name": "object"}, {"name": "location"}],
            activation=0.5
        )
        
        # Add initial canvas
        canvas = self.store.add_canvas(
            "pick-and-place scenario",
            entities=[grasp, place],
            links=[{"src": grasp.id, "dst": place.id, "relation": "enables"}]
        )
        
        print(f"✅ Seeded {len(self.store.schemes)} schemes, {len(self.store.canvases)} canvas")
    
    def run_sleep_cycle(self, cycles: int = 3):
        """Run a simulated sleep cycle with augmentation."""
        print(f"\n🌙 Starting sleep cycle ({cycles} iterations)...")
        
        for i in range(cycles):
            print(f"\n  Cycle {i+1}/{cycles}:")
            
            # 1. Read current state
            state = self.store.read(activation_min=0.3)
            print(f"    📖 Read: {len(state['schemes'])} schemes, {len(state['canvases'])} canvases")
            
            # 2. Imagine new canvases
            if state['schemes']:
                scheme = state['schemes'][0]
                new_canvases = self.store.imagine(scheme['id'], count=2)
                print(f"    💭 Imagined: {len(new_canvases)} new canvases")
            
            # 3. Activate important schemes
            for scheme in state['schemes'][:2]:
                self.store.activate(scheme['id'], delta=0.1)
                print(f"    ⚡ Activated: {scheme['label']} (+0.1)")
            
            # 4. Write abstraction
            if i == cycles - 1:  # Only on last cycle
                new_scheme = self.store.add_scheme(
                    f"abstract_pattern_{i}",
                    [{"name": "entity"}, {"name": "context"}],
                    activation=0.8
                )
                print(f"    ✍️  Wrote: {new_scheme.label}")
        
        # Final snapshot
        snapshot = self.store.snapshot()
        print(f"\n📸 Final state: {len(snapshot['schemes'])} schemes, {len(snapshot['canvases'])} canvases")
        return snapshot
    
    def run_tests(self):
        """Run basic integration tests."""
        print("\n🧪 Running tests...")
        
        tests_passed = 0
        tests_total = 4
        
        # Test 1: Read
        try:
            result = self.store.read()
            assert len(result['schemes']) > 0
            print("  ✅ Test 1: Read operation")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 1: Read failed - {e}")
        
        # Test 2: Imagine
        try:
            schemes = list(self.store.schemes.values())
            if schemes:
                canvases = self.store.imagine(schemes[0].id, count=1)
                assert len(canvases) > 0
                print("  ✅ Test 2: Imagine operation")
                tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 2: Imagine failed - {e}")
        
        # Test 3: Activate
        try:
            if schemes:
                old_activation = schemes[0].activation
                self.store.activate(schemes[0].id, delta=0.2)
                assert schemes[0].activation > old_activation
                print("  ✅ Test 3: Activate operation")
                tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 3: Activate failed - {e}")
        
        # Test 4: Write
        try:
            new_scheme = self.store.add_scheme("test_scheme", [{"name": "test"}])
            assert new_scheme.id in self.store.schemes
            print("  ✅ Test 4: Write operation")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 4: Write failed - {e}")
        
        print(f"\n📊 Tests passed: {tests_passed}/{tests_total}")
        return tests_passed == tests_total


def main():
    """Main entry point."""
    print("🚀 SAGE-Totality Standalone Runner")
    print("=" * 40)
    
    # Load configuration if available
    config_file = Path(__file__).parent / "machine_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            settings = config.get("settings", {})
            print(f"📦 Machine: {config.get('machine', {}).get('machine_profile', 'unknown')}")
            print(f"⚙️  Device: {settings.get('device', 'cpu')}")
    
    # Initialize and run
    sage = StandaloneSageTotality()
    
    # Run tests
    sage.run_tests()
    
    # Run sleep cycle
    cycles = settings.get("sleep_cycle_count", 3) if 'settings' in locals() else 3
    cycles = min(cycles, 5)  # Limit for standalone demo
    sage.run_sleep_cycle(cycles)
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
