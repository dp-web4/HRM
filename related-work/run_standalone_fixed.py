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
from transforms import context_shift, semantic_variation

class StandaloneSageTotality:
    """Run SAGE-Totality operations without web service."""
    
    def __init__(self):
        self.store = TotalityStore()
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
    
    def demonstrate_augmentation(self):
        """Demonstrate how augmentation works with Totality structures."""
        print("\n🔬 Demonstrating Augmentation Engine Integration:")
        print("-" * 40)
        
        # Get a scheme to augment
        schemes = list(self.store.schemes.values())
        if not schemes:
            return
        
        base_scheme = schemes[0]
        print(f"Base scheme: {base_scheme.label}")
        
        # Demonstrate different types of augmentation
        augmentations = []
        
        # 1. Semantic variation (like HRM's value permutations)
        semantic_aug = semantic_variation(base_scheme.label)
        print(f"  → Semantic: {semantic_aug}")
        augmentations.append(semantic_aug)
        
        # 2. Context shift (like HRM's translational shifts)
        contexts = ["kitchen", "workshop", "laboratory"]
        for ctx in contexts:
            ctx_aug = context_shift(base_scheme.label, ctx)
            print(f"  → Context [{ctx}]: {ctx_aug}")
            augmentations.append(ctx_aug)
        
        # 3. Demonstrate dream-like recombination
        print("\n💭 Dream-like recombination:")
        for i, aug in enumerate(augmentations[:3]):
            # Create new canvases from augmented schemes
            new_scheme = self.store.add_scheme(
                aug,
                [{"name": "entity", "value": f"aug_{i}"}],
                activation=0.6
            )
            canvas = self.store.add_canvas(
                f"Dream variation {i+1}: {aug}",
                entities=[new_scheme],
                provenance_ops=["augmented", "imagined"]
            )
            print(f"  Created canvas: {canvas.description}")
    
    def run_sleep_cycle(self, cycles: int = 3):
        """Run a simulated sleep cycle with augmentation."""
        print(f"\n🌙 Starting sleep cycle ({cycles} iterations)...")
        print("=" * 40)
        
        initial_schemes = len(self.store.schemes)
        initial_canvases = len(self.store.canvases)
        
        for i in range(cycles):
            print(f"\n  Cycle {i+1}/{cycles}:")
            
            # 1. Read current state (H-level processing)
            state = self.store.read(activation_min=0.3)
            print(f"    📖 Read: {len(state['schemes'])} active schemes")
            
            # 2. Imagine new canvases (dream augmentation)
            if state['schemes']:
                scheme = state['schemes'][0]
                new_canvases = self.store.imagine(scheme['id'], count=2)
                print(f"    💭 Imagined: {len(new_canvases)} new canvases")
                
                # Show what was imagined
                for c_id in new_canvases[:1]:  # Show first one
                    canvas = self.store.canvases.get(c_id)
                    if canvas:
                        print(f"       → {canvas.description[:50]}")
            
            # 3. Activate important schemes (attention mechanism)
            for scheme in state['schemes'][:2]:
                old_act = scheme['activation']
                self.store.activate(scheme['id'], delta=0.1)
                print(f"    ⚡ Activated: {scheme['label'][:30]} ({old_act:.2f} → {min(1.0, old_act + 0.1):.2f})")
            
            # 4. Consolidate learning (write abstraction)
            if i == cycles - 1:  # Only on last cycle
                # This represents the distilled wisdom from dreams
                new_scheme = self.store.add_scheme(
                    f"learned_pattern: combine(grasp, place)",
                    [{"name": "sequence"}, {"name": "goal"}],
                    activation=0.8
                )
                print(f"    ✍️  Consolidated: {new_scheme.label}")
        
        # Final statistics
        print(f"\n📊 Sleep Cycle Results:")
        print(f"  Schemes: {initial_schemes} → {len(self.store.schemes)} (+{len(self.store.schemes) - initial_schemes})")
        print(f"  Canvases: {initial_canvases} → {len(self.store.canvases)} (+{len(self.store.canvases) - initial_canvases})")
        
        # Show activation changes (learning effect)
        print(f"\n🧠 Learning Effect (activation changes):")
        for scheme in list(self.store.schemes.values())[:3]:
            print(f"  {scheme.label[:40]}: {scheme.activation:.2f}")
    
    def demonstrate_dual_training(self):
        """Show how H-level and L-level training differ."""
        print("\n🎭 Dual Training Demonstration:")
        print("-" * 40)
        
        print("H-Level (Strategic/Dream):")
        print("  • Processes: Abstract patterns, relationships")
        print("  • Training: Large batches during sleep")
        print("  • Example: 'grasp enables place' pattern")
        
        print("\nL-Level (Tactical/Continuous):")
        print("  • Processes: Motor commands, precise timing")
        print("  • Training: Small incremental updates")
        print("  • Example: grip force = 0.7N for object_weight")
        
        # Simulate the difference
        h_scheme = self.store.add_scheme(
            "strategic: sequence(grasp, move, place)",
            [{"name": "steps"}, {"name": "goal"}],
            activation=0.5
        )
        print(f"\n  Created H-level scheme: {h_scheme.label}")
        
        # L-level would be continuous small adjustments (simulated)
        print("  L-level adjustments (simulated):")
        for i in range(3):
            adjustment = 0.01 * (i + 1)
            print(f"    t={i}: grip_force += {adjustment:.3f}")
    
    def run_tests(self):
        """Run basic integration tests."""
        print("\n🧪 Running Integration Tests...")
        print("-" * 40)
        
        tests_passed = 0
        tests_total = 5
        
        # Test 1: Read operation
        try:
            result = self.store.read()
            assert len(result['schemes']) > 0
            print("  ✅ Test 1: Read operation works")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 1: Read failed - {e}")
        
        # Test 2: Imagine (dream generation)
        try:
            schemes = list(self.store.schemes.values())
            if schemes:
                canvases = self.store.imagine(schemes[0].id, count=1)
                assert len(canvases) > 0
                print("  ✅ Test 2: Imagine/dream generation works")
                tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 2: Imagine failed - {e}")
        
        # Test 3: Activate (attention)
        try:
            if schemes:
                scheme = schemes[0]
                old_activation = scheme.activation
                self.store.activate(scheme.id, delta=0.2)
                new_activation = self.store.schemes[scheme.id].activation
                assert new_activation != old_activation
                print("  ✅ Test 3: Activation/attention works")
                tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 3: Activate failed - {e}")
        
        # Test 4: Write (consolidation)
        try:
            new_scheme = self.store.add_scheme("test_consolidation", [{"name": "test"}])
            assert new_scheme.id in self.store.schemes
            print("  ✅ Test 4: Write/consolidation works")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 4: Write failed - {e}")
        
        # Test 5: Augmentation integration
        try:
            original_text = "grasp(object)"
            augmented = semantic_variation(original_text)
            assert augmented != original_text
            print("  ✅ Test 5: Augmentation transform works")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Test 5: Augmentation failed - {e}")
        
        print(f"\n📊 Tests passed: {tests_passed}/{tests_total}")
        return tests_passed == tests_total


def main():
    """Main entry point."""
    print("🚀 SAGE-Totality Integration Demo")
    print("=" * 50)
    
    # Load configuration if available
    config_file = Path(__file__).parent / "machine_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            machine = config.get("machine", {})
            settings = config.get("settings", {})
            print(f"📦 Machine: {machine.get('machine_profile', 'unknown')}")
            print(f"🖥️  Host: {machine.get('hostname', 'unknown')}")
            print(f"⚙️  Device: {settings.get('device', 'cpu')}")
            print(f"🎮 GPU: {machine.get('gpu_name', 'Not detected')}")
            
            # Show what this means for SAGE
            if settings.get('device') == 'cuda':
                print(f"✨ CUDA enabled - can run full HRM on GPU")
            print(f"💤 Sleep cycles configured: {settings.get('sleep_cycle_count', 'default')}")
            print(f"🔄 Augmentations per dream: {settings.get('augmentation_count', 'default')}")
    else:
        settings = {}
    
    print("=" * 50)
    
    # Initialize SAGE-Totality integration
    sage = StandaloneSageTotality()
    
    # Run tests
    if sage.run_tests():
        print("✅ All integration tests passed!\n")
    
    # Demonstrate augmentation
    sage.demonstrate_augmentation()
    
    # Demonstrate dual training
    sage.demonstrate_dual_training()
    
    # Run sleep cycle (limited for demo)
    cycles = min(settings.get("sleep_cycle_count", 3), 5)
    sage.run_sleep_cycle(cycles)
    
    print("\n" + "=" * 50)
    print("✨ Integration demo complete!")
    print("\nKey Insights Demonstrated:")
    print("  1. Totality provides structured cognitive workspace")
    print("  2. Augmentation transforms schemes like dreams transform memories")
    print("  3. Dual training: H-level (dreams) vs L-level (practice)")
    print("  4. Sleep cycles consolidate patterns into wisdom")
    print("\nNext steps: Run on Jetson/Legion with full GPU acceleration")


if __name__ == "__main__":
    main()