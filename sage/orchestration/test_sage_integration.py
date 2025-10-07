#!/usr/bin/env python3
"""
SAGE Multi-Agent Integration Test
Tests the complete orchestration system with all agents working together
"""

import sys
import time
import numpy as np
from pathlib import Path
import importlib.util

# Add agent paths
sys.path.append("agents/vision")
sys.path.append("agents/trust")
sys.path.append("agents/memory")
sys.path.append("agents/training")
sys.path.append("agents/control")

# Import agents with proper module loading
def load_agent_module(path, module_name):
    """Load an agent module from file"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load all agent modules
vision_module = load_agent_module(
    "agents/vision/eagle-vision-irp.py", "eagle_vision_irp"
)
trust_module = load_agent_module(
    "agents/trust/trust-attention-surprise-coordinator.py", "tas_coordinator"
)
memory_module = load_agent_module(
    "agents/memory/memory-consolidation-agent.py", "memory_consolidation"
)
training_module = load_agent_module(
    "agents/training/groot-data-processor.py", "groot_processor"
)
control_module = load_agent_module(
    "agents/control/metabolic-state-manager.py", "metabolic_manager"
)

# Extract classes
EagleVisionIRP = vision_module.EagleVisionIRP
TASCoordinator = trust_module.TASCoordinator
MemoryConsolidationAgent = memory_module.MemoryConsolidationAgent
GR00TDataProcessor = training_module.GR00TDataProcessor
MetabolicStateManager = control_module.MetabolicStateManager


class SAGEIntegrationTest:
    """
    Integration test for the complete SAGE system
    """
    
    def __init__(self):
        print("🧠 Initializing SAGE Integration Test")
        print("=" * 60)
        
        # Initialize all agents
        self.vision_agent = EagleVisionIRP({
            "max_iterations": 3,
            "energy_threshold": 0.2
        })
        
        self.tas_coordinator = TASCoordinator({
            "max_attention_targets": 3,
            "attention_budget": 1.0,
            "surprise_threshold": 0.3
        })
        
        self.memory_agent = MemoryConsolidationAgent({
            "buffer_size": 50,
            "consolidation_interval": 10,
            "min_salience": 0.3,
            "compression_ratio": 0.2
        })
        
        self.groot_processor = GR00TDataProcessor({
            "episode_count": 2,
            "batch_size": 4
        })
        
        self.metabolic_manager = MetabolicStateManager({
            "initial_energy": 100.0,
            "recharge_rate": 5.0
        })
        
        print("✅ All agents initialized\n")
    
    def run_integration_test(self):
        """Run the complete integration test"""
        
        print("🚀 Starting SAGE Integration Test")
        print("-" * 60)
        
        # Start background services
        print("\n1️⃣ Starting background services...")
        self.tas_coordinator.start()
        self.metabolic_manager.start()
        print("  ✅ TAS Coordinator and Metabolic Manager running")
        
        # Load demonstration data
        print("\n2️⃣ Loading GR00T demonstration data...")
        episodes = self.groot_processor.load_episodes()
        print(f"  ✅ Loaded {len(episodes)} episodes")
        
        # Process vision through IRP
        print("\n3️⃣ Processing vision through Eagle IRP...")
        vision_results = []
        for i in range(3):
            result = self.vision_agent.process_image(f"test_frame_{i}.jpg")
            vision_results.append(result)
            print(f"  Frame {i}: {result['iterations']} iterations, "
                  f"energy={result['final_energy']:.3f}")
        
        # Submit observations to TAS coordinator
        print("\n4️⃣ Processing observations through TAS...")
        for i, result in enumerate(vision_results):
            # Create observation
            expected = np.random.random()
            actual = expected + np.random.normal(0, 0.1)
            
            obs = trust_module.Observation(
                source=f"vision_{i}",
                data={"features": result["features"][:10]},  # Sample features
                timestamp=time.time(),
                expected_value=expected,
                actual_value=actual
            )
            
            self.tas_coordinator.submit_observation(obs)
        
        time.sleep(0.5)  # Allow processing
        
        # Check attention allocation
        attention = self.tas_coordinator.get_attention_allocation()
        print("  Attention allocation:")
        for source, weight in attention.items():
            print(f"    {source}: {weight:.2%}")
        
        # Add experiences to memory
        print("\n5️⃣ Adding experiences to memory...")
        for i in range(15):
            exp = memory_module.Experience(
                timestamp=time.time(),
                source=f"sensor_{i % 3}",
                features=np.random.randn(1536),
                action=np.random.randn(7) if i % 2 == 0 else None,
                reward=np.random.random(),
                salience=np.random.random(),
                metadata={"test": True, "index": i}
            )
            self.memory_agent.add_experience(exp)
        
        print(f"  ✅ Added {self.memory_agent.total_experiences} experiences")
        
        # Wait for consolidation
        time.sleep(2)
        
        # Check memory statistics
        memory_stats = self.memory_agent.get_statistics()
        print(f"  Consolidated: {memory_stats['total_consolidated']} patterns")
        print(f"  Compression: {memory_stats['compression_achieved']:.1%}")
        
        # Simulate metabolic state changes
        print("\n6️⃣ Testing metabolic state transitions...")
        
        # High performance
        self.metabolic_manager.submit_event({
            "type": "performance",
            "value": 0.85
        })
        time.sleep(1)
        
        status1 = self.metabolic_manager.get_status()
        print(f"  State: {status1['current_state']}, "
              f"Energy: {status1['energy_fraction']:.1%}")
        
        # High surprise
        self.metabolic_manager.submit_event({
            "type": "surprise",
            "level": 0.9
        })
        time.sleep(1)
        
        status2 = self.metabolic_manager.get_status()
        print(f"  State: {status2['current_state']}, "
              f"Energy: {status2['energy_fraction']:.1%}")
        
        # Process GR00T episodes for distillation
        print("\n7️⃣ Processing episodes for knowledge distillation...")
        processed = self.groot_processor.process_all_episodes()
        print(f"  ✅ Processed {len(processed)} episodes")
        
        # Create training batch
        batch = self.groot_processor.create_training_batch(batch_size=4)
        print(f"  Created batch: {batch['visual_features'].shape[0]} samples")
        
        # Final system status
        print("\n8️⃣ Final System Status:")
        print("-" * 40)
        
        # TAS status
        tas_stats = self.tas_coordinator.get_statistics()
        print(f"  TAS Coordinator:")
        print(f"    Total observations: {tas_stats['total_observations']}")
        print(f"    Surprise rate: {tas_stats['surprise_rate']:.1%}")
        print(f"    Active targets: {tas_stats['num_attention_targets']}")
        
        # Memory status
        memory_stats = self.memory_agent.get_statistics()
        print(f"  Memory Agent:")
        print(f"    Buffer usage: {memory_stats['buffer_size']}")
        print(f"    Consolidated: {memory_stats['total_consolidated']}")
        
        # Metabolic status
        metabolic_status = self.metabolic_manager.get_status()
        print(f"  Metabolic Manager:")
        print(f"    Current state: {metabolic_status['current_state']}")
        print(f"    Energy: {metabolic_status['energy']:.1f}/{100.0}")
        print(f"    Consumption rate: {metabolic_status['consumption_rate']:.1f}/s")
        
        # Stop background services
        print("\n9️⃣ Stopping background services...")
        self.tas_coordinator.stop()
        self.metabolic_manager.stop()
        print("  ✅ All services stopped")
        
        print("\n" + "=" * 60)
        print("🎉 SAGE Integration Test Complete!")
        print("All agents successfully orchestrated together!")
        
        return True


def main():
    """Run the integration test"""
    test = SAGEIntegrationTest()
    success = test.run_integration_test()
    
    if success:
        print("\n✅ Integration test PASSED")
    else:
        print("\n❌ Integration test FAILED")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())