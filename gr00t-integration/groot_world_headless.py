#!/usr/bin/env python3
"""
GR00T Headless World Simulation
================================
Runs world simulation without GUI, saves outputs directly.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from groot_world_sim import GR00TWorldSimulator
import numpy as np

def main():
    print("\n" + "="*60)
    print("GR00T Simulated World View - Headless Mode")
    print("="*60)
    
    # Create simulator
    print("\n🔧 Initializing GR00T world simulator...")
    sim = GR00TWorldSimulator()
    
    # Test different commands
    commands = [
        "Pick up the red cube and place it at the goal",
        "Stack all cubes in order of size",
        "Avoid the obstacle while moving to the target"
    ]
    
    for i, command in enumerate(commands):
        print(f"\n📝 Task {i+1}: {command}")
        
        # Run perception
        sim.simulate_perception(command)
        
        # Save visualization
        output_file = f"groot_world_{i+1}.png"
        sim.render_world_view(output_file)
        print(f"   ✅ Saved visualization to {output_file}")
        
    # Generate summary stats
    print("\n📊 Simulation Statistics:")
    print(f"   Objects in world: {len(sim.objects)}")
    print(f"   Trajectory waypoints: {len(sim.trajectory_plan)}")
    print(f"   Robot position: {sim.robot_state.position}")
    
    if sim.predicted_actions is not None:
        print(f"   Action dimensions: {sim.predicted_actions.shape[1]}")
        print(f"   Max action value: {np.max(np.abs(sim.predicted_actions)):.3f}")
        
    if sim.vision_features is not None:
        print(f"   Vision feature dim: {sim.vision_features.shape[1]}")
        
    print("\n✅ Headless simulation complete!")
    print("   View the generated PNG files to see the world visualizations")
    

if __name__ == "__main__":
    main()