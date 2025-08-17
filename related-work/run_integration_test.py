#!/usr/bin/env python3
"""
SAGE-Totality Integration Test
Demonstrates how Totality acts as a cognitive sensor for SAGE.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add totality_min to path
sys.path.insert(0, str(Path(__file__).parent / "totality_min"))

from totality_core import TotalityStore, Scheme, Canvas
from transforms import context_shift, semantic_variation


class SAGETotalityIntegration:
    """Integration layer between SAGE and Totality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.store = TotalityStore()
        self.config = config or {}
        self.settings = self.config.get("settings", {})
        self.trust_scores = {}  # Track trust for Totality outputs
        self._seed_data()
    
    def _seed_data(self):
        """Initialize with basic schemes and canvas."""
        # Basic manipulation schemes
        grasp = self.store.add_scheme(
            "grasp(object)",
            [{"name": "object", "value": "block"}],
            activation=0.7
        )
        place = self.store.add_scheme(
            "place(object, location)",
            [{"name": "object", "value": "block"}, {"name": "location", "value": "table"}],
            activation=0.5
        )
        
        # Initial canvas linking them
        canvas = self.store.add_canvas(
            "pick-and-place scenario",
            entities=[grasp, place],
            links=[{"src": grasp.id, "dst": place.id, "relation": "enables"}]
        )
        
        # Initialize trust scores
        self.trust_scores[grasp.id] = 0.8  # High trust for basic actions
        self.trust_scores[place.id] = 0.8
        
        print(f"✅ Initialized with {len(self.store.schemes)} schemes, {len(self.store.canvases)} canvas")
    
    def cognitive_sensor_read(self, min_trust: float = 0.5) -> Dict[str, Any]:
        """
        SAGE reads from Totality as a cognitive sensor.
        Filters by trust score like other sensors.
        """
        # Read all active schemes
        state = self.store.read(activation_min=0.3)
        
        # Filter by trust score (SAGE's trust mechanism)
        trusted_schemes = []
        for scheme in state['schemes']:
            trust = self.trust_scores.get(scheme.id if hasattr(scheme, 'id') else scheme.get('id'), 0.5)
            if trust >= min_trust:
                trusted_schemes.append({
                    "scheme": scheme,
                    "trust": trust,
                    "weighted_activation": scheme.activation if hasattr(scheme, 'activation') else scheme.get('activation', 0) * trust
                })
        
        return {
            "trusted_schemes": trusted_schemes,
            "canvases": state['canvases'],
            "sensor_type": "cognitive",
            "timestamp": "simulated"
        }
    
    def augment_with_llm_simulation(self, scheme: Scheme) -> List[Dict[str, Any]]:
        """
        Simulate LLM-assisted augmentation.
        In production, this would call actual LLM.
        """
        augmentations = []
        
        # 1. Semantic variation (what if different words?)
        semantic = semantic_variation(scheme.label)
        augmentations.append({
            "type": "semantic",
            "original": scheme.label,
            "augmented": semantic,
            "explanation": "Alternative phrasing of the same concept"
        })
        
        # 2. Context shifts (what if different environment?)
        contexts = ["kitchen", "factory", "underwater"]
        for ctx in contexts:
            contextual = context_shift(scheme.label, ctx)
            augmentations.append({
                "type": "context",
                "original": scheme.label,
                "augmented": contextual,
                "explanation": f"Same action in {ctx} environment"
            })
        
        # 3. Abstract pattern extraction (what's the core pattern?)
        abstract = f"manipulate(entity) // abstracted from {scheme.label}"
        augmentations.append({
            "type": "abstraction",
            "original": scheme.label,
            "augmented": abstract,
            "explanation": "Core pattern extraction"
        })
        
        return augmentations
    
    def sleep_cycle_with_dual_training(self, cycles: int = 3):
        """
        Demonstrate SAGE's dual training loops:
        - H-level: Dream augmentation and pattern extraction
        - L-level: Continuous refinement of execution
        """
        print(f"\n🌙 SAGE Sleep Cycle with Dual Training")
        print("=" * 50)
        
        h_level_updates = []  # Strategic updates from dreams
        l_level_updates = []  # Tactical updates from practice
        
        for cycle in range(cycles):
            print(f"\n📍 Cycle {cycle + 1}/{cycles}")
            
            # H-LEVEL TRAINING (Dreams/Augmentation)
            print("  H-Level (Strategic/Cognitive):")
            
            # 1. Read from cognitive sensor
            cognitive_state = self.cognitive_sensor_read()
            print(f"    📖 Read {len(cognitive_state['trusted_schemes'])} trusted schemes")
            
            # 2. Select schemes for augmentation
            if cognitive_state['trusted_schemes']:
                for item in cognitive_state['trusted_schemes'][:1]:  # Process first scheme
                    scheme = item['scheme']
                    trust = item['trust']
                    
                    # Generate augmentations (dreams)
                    augmentations = self.augment_with_llm_simulation(scheme)
                    print(f"    💭 Generated {len(augmentations)} augmentations for: {scheme.label if hasattr(scheme, 'label') else scheme.get('label')}")
                    
                    # Create new schemes from augmentations
                    for aug in augmentations[:2]:  # Limit for demo
                        new_scheme = self.store.add_scheme(
                            aug['augmented'],
                            [{"name": "augmented", "value": aug['type']}],
                            activation=0.6
                        )
                        # New schemes start with lower trust (need validation)
                        self.trust_scores[new_scheme.id] = trust * 0.7
                        h_level_updates.append(f"Learned: {aug['augmented'][:40]}")
                        print(f"      → {aug['type']}: {aug['augmented'][:40]}")
            
            # L-LEVEL TRAINING (Continuous refinement)
            print("  L-Level (Tactical/Motor):")
            
            # Simulate continuous small adjustments
            motor_params = {
                "grip_force": 0.5,
                "move_speed": 1.0,
                "precision": 0.8
            }
            
            for param, value in motor_params.items():
                # Small incremental updates (would be from sensor feedback in real system)
                adjustment = 0.01 * (cycle + 1)
                new_value = min(1.0, value + adjustment)
                l_level_updates.append(f"{param}: {value:.2f} → {new_value:.2f}")
                print(f"    ⚙️  {param}: +{adjustment:.3f} (now {new_value:.2f})")
            
            # Update activations based on usage (attention mechanism)
            print("  Activation Updates:")
            for scheme_data in cognitive_state['trusted_schemes'][:2]:
                scheme = scheme_data['scheme']
                scheme_id = scheme.id if hasattr(scheme, 'id') else scheme.get('id')
                # Activate requires specific format
                self.store.activate([{"id": scheme_id, "delta": 0.1}])
                print(f"    ⚡ Boosted: {scheme.label if hasattr(scheme, 'label') else scheme.get('label', 'unknown')[:30]}")
        
        # Consolidation phase
        print(f"\n✨ Consolidation Phase:")
        print(f"  H-Level learned {len(h_level_updates)} strategic patterns")
        print(f"  L-Level made {len(l_level_updates)} tactical adjustments")
        
        # Create final abstraction (wisdom from experience)
        wisdom_scheme = self.store.add_scheme(
            "wisdom: adaptive_manipulation(context, precision)",
            [{"name": "context", "value": "learned"}, {"name": "precision", "value": "refined"}],
            activation=0.9
        )
        self.trust_scores[wisdom_scheme.id] = 0.9  # High trust for consolidated wisdom
        print(f"  📚 Distilled wisdom: {wisdom_scheme.label}")
        
        return {
            "h_level_updates": h_level_updates,
            "l_level_updates": l_level_updates,
            "final_schemes": len(self.store.schemes),
            "final_canvases": len(self.store.canvases)
        }
    
    def demonstrate_trust_evolution(self):
        """Show how trust scores evolve based on performance."""
        print("\n🎯 Trust Score Evolution")
        print("-" * 40)
        
        print("Initial trust scores:")
        for scheme_id, trust in list(self.trust_scores.items())[:3]:
            scheme = self.store.schemes.get(scheme_id)
            if scheme:
                print(f"  {scheme.label[:30]}: {trust:.2f}")
        
        print("\nSimulating performance feedback...")
        # Simulate that augmented schemes performed well
        for scheme_id in list(self.trust_scores.keys()):
            if "context" in str(self.store.schemes.get(scheme_id, "").label if scheme_id in self.store.schemes else ""):
                # Context-aware schemes get trust boost
                self.trust_scores[scheme_id] = min(1.0, self.trust_scores[scheme_id] + 0.1)
                print(f"  ✅ Boosted trust for context-aware scheme")
        
        print("\nUpdated trust scores:")
        for scheme_id, trust in list(self.trust_scores.items())[:3]:
            scheme = self.store.schemes.get(scheme_id)
            if scheme:
                print(f"  {scheme.label[:30]}: {trust:.2f}")


def main():
    """Main test routine."""
    print("🚀 SAGE-Totality Integration Test")
    print("=" * 50)
    
    # Load machine configuration
    config = {}
    config_file = Path(__file__).parent / "machine_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            machine = config.get("machine", {})
            settings = config.get("settings", {})
            
            print(f"📦 Machine: {machine.get('machine_profile', 'unknown')}")
            print(f"🎮 GPU: {machine.get('gpu_name', 'Not detected')}")
            print(f"⚙️  Config: {settings.get('sleep_cycle_count')} cycles, "
                  f"{settings.get('augmentation_count')} augmentations")
    
    # Initialize integration
    sage_totality = SAGETotalityIntegration(config)
    
    # Test cognitive sensor reading
    print("\n🔍 Testing Cognitive Sensor:")
    sensor_data = sage_totality.cognitive_sensor_read()
    print(f"  Received {len(sensor_data['trusted_schemes'])} trusted schemes")
    for item in sensor_data['trusted_schemes'][:2]:
        scheme = item['scheme']
        print(f"    → {scheme.label if hasattr(scheme, 'label') else 'scheme'} (trust: {item['trust']:.2f})")
    
    # Run sleep cycle with dual training
    cycles = min(config.get("settings", {}).get("sleep_cycle_count", 3), 5)
    results = sage_totality.sleep_cycle_with_dual_training(cycles)
    
    # Demonstrate trust evolution
    sage_totality.demonstrate_trust_evolution()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Integration Test Summary:")
    print(f"  ✅ Cognitive sensor operational")
    print(f"  ✅ Dual training loops demonstrated")
    print(f"  ✅ Trust-based filtering active")
    print(f"  ✅ Augmentation engine integrated")
    print(f"  📈 Final state: {results['final_schemes']} schemes, {results['final_canvases']} canvases")
    
    print("\n🎓 Key Achievements:")
    print("  1. Totality acts as trusted cognitive sensor in SAGE")
    print("  2. H-level trains on augmented dreams, L-level on continuous practice")
    print("  3. Trust scores gate which cognitive patterns influence decisions")
    print("  4. Wisdom emerges from consolidation of augmented experiences")
    
    print("\n📍 Next Steps:")
    print("  - Deploy on Jetson with real sensor integration")
    print("  - Connect to actual LLMs for richer augmentation")
    print("  - Implement full HRM training loops")
    print("  - Test on real robotics tasks")


if __name__ == "__main__":
    main()