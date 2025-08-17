#!/usr/bin/env python3
"""
GR00T Simulated World View Renderer
====================================
Renders GR00T's internal world model and perception state.
Based on NVIDIA's Isaac Sim visualizations for embodied AI.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Arrow, FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
from load_groot_full import GR00TModel, load_groot_model

@dataclass
class WorldObject:
    """Object in GR00T's world model"""
    name: str
    position: np.ndarray  # [x, y, z]
    size: np.ndarray      # [width, height, depth]
    color: str
    object_type: str      # 'static', 'dynamic', 'target', 'obstacle'
    confidence: float     # Trust/confidence score
    
@dataclass
class RobotState:
    """Robot's state in the world"""
    position: np.ndarray
    orientation: float  # Angle in radians
    gripper_state: float  # 0=open, 1=closed
    joint_angles: np.ndarray
    velocity: np.ndarray
    
class GR00TWorldSimulator:
    """Simulates and renders GR00T's internal world representation"""
    
    def __init__(self, model: Optional[GR00TModel] = None, device: str = "cuda"):
        """Initialize world simulator
        
        Args:
            model: Pre-loaded GR00T model
            device: Device for computation
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model if not provided
        if model is None:
            print("📦 Loading GR00T model...")
            self.model = load_groot_model(self.device)
        else:
            self.model = model.to(self.device)
            
        self.model.eval()
        
        # World state
        self.objects: List[WorldObject] = []
        self.robot_state = RobotState(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=0.0,
            gripper_state=0.0,
            joint_angles=np.zeros(7),  # 7-DOF arm
            velocity=np.zeros(3)
        )
        
        # Perception state
        self.vision_features = None
        self.predicted_actions = None
        self.attention_map = None
        self.trajectory_plan = []
        
        # Simulation parameters
        self.world_bounds = np.array([[-2, 2], [-2, 2], [0, 2]])  # [x, y, z] bounds
        self.time_step = 0
        self.running = False
        
        # Initialize world with demo objects
        self._init_demo_world()
        
    def _init_demo_world(self):
        """Initialize a demo world with objects"""
        # Table
        self.objects.append(WorldObject(
            name="table",
            position=np.array([0.0, 0.0, 0.5]),
            size=np.array([1.5, 1.0, 0.05]),
            color='brown',
            object_type='static',
            confidence=1.0
        ))
        
        # Target objects (cubes to manipulate)
        colors = ['red', 'blue', 'green']
        for i, color in enumerate(colors):
            self.objects.append(WorldObject(
                name=f"{color}_cube",
                position=np.array([-0.4 + i*0.4, 0.3, 0.55]),
                size=np.array([0.1, 0.1, 0.1]),
                color=color,
                object_type='target',
                confidence=0.9 - i*0.1
            ))
            
        # Goal positions (transparent)
        for i, color in enumerate(colors):
            self.objects.append(WorldObject(
                name=f"{color}_goal",
                position=np.array([-0.4 + i*0.4, -0.3, 0.55]),
                size=np.array([0.12, 0.12, 0.02]),
                color=color,
                object_type='goal',
                confidence=0.5
            ))
            
        # Obstacles
        self.objects.append(WorldObject(
            name="obstacle",
            position=np.array([0.6, 0.0, 0.8]),
            size=np.array([0.15, 0.15, 0.4]),
            color='gray',
            object_type='obstacle',
            confidence=0.95
        ))
        
    def simulate_perception(self, language_command: str = "Pick up the red cube and place it at the goal"):
        """Simulate GR00T's perception and planning
        
        Args:
            language_command: Natural language instruction
        """
        print(f"\n🧠 Processing command: '{language_command}'")
        
        # Generate mock visual input (in real system, this would be camera feed)
        batch_size = 1
        visual_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # Tokenize language (mock)
        language_ids = torch.randint(0, 50000, (batch_size, 50)).to(self.device)
        
        # Get robot proprioceptive state
        proprio_state = torch.from_numpy(
            np.concatenate([
                self.robot_state.position,
                [self.robot_state.orientation],
                self.robot_state.joint_angles,
                [self.robot_state.gripper_state]
            ])
        ).float().unsqueeze(0).to(self.device)
        
        # Pad to expected dimension
        if proprio_state.shape[1] < 64:
            proprio_state = torch.cat([
                proprio_state,
                torch.zeros(1, 64 - proprio_state.shape[1]).to(self.device)
            ], dim=1)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(visual_input, language_ids, proprio_state)
            
        # Store perception results
        self.vision_features = outputs['vision_features'].cpu().numpy()
        self.predicted_actions = outputs['actions'].cpu().numpy()
        
        # Generate attention map (simulate what the model is "looking at")
        self.attention_map = self._generate_attention_map()
        
        # Generate trajectory plan
        self.trajectory_plan = self._plan_trajectory()
        
        print(f"✅ Perception complete. Generated {len(self.trajectory_plan)} waypoints")
        
    def _generate_attention_map(self) -> np.ndarray:
        """Generate attention heatmap over world objects"""
        attention = np.zeros((20, 20))
        
        # Focus attention on target objects
        for obj in self.objects:
            if obj.object_type == 'target':
                # Map object position to grid
                x_idx = int((obj.position[0] - self.world_bounds[0, 0]) / 
                           (self.world_bounds[0, 1] - self.world_bounds[0, 0]) * 20)
                y_idx = int((obj.position[1] - self.world_bounds[1, 0]) / 
                           (self.world_bounds[1, 1] - self.world_bounds[1, 0]) * 20)
                
                x_idx = np.clip(x_idx, 0, 19)
                y_idx = np.clip(y_idx, 0, 19)
                
                # Add Gaussian blob
                for i in range(max(0, x_idx-3), min(20, x_idx+4)):
                    for j in range(max(0, y_idx-3), min(20, y_idx+4)):
                        dist = np.sqrt((i-x_idx)**2 + (j-y_idx)**2)
                        attention[j, i] += obj.confidence * np.exp(-dist/2)
                        
        return attention / (attention.max() + 1e-8)
        
    def _plan_trajectory(self) -> List[np.ndarray]:
        """Generate trajectory waypoints for task execution"""
        waypoints = []
        
        # Find red cube
        red_cube = next((obj for obj in self.objects if obj.name == "red_cube"), None)
        red_goal = next((obj for obj in self.objects if obj.name == "red_goal"), None)
        
        if red_cube and red_goal:
            # Approach position (above cube)
            approach_pos = red_cube.position.copy()
            approach_pos[2] += 0.2
            waypoints.append(approach_pos)
            
            # Grasp position
            waypoints.append(red_cube.position.copy())
            
            # Lift position
            lift_pos = red_cube.position.copy()
            lift_pos[2] += 0.3
            waypoints.append(lift_pos)
            
            # Move to above goal
            goal_approach = red_goal.position.copy()
            goal_approach[2] += 0.3
            waypoints.append(goal_approach)
            
            # Place at goal
            waypoints.append(red_goal.position.copy())
            
        return waypoints
        
    def render_world_view(self, save_path: Optional[str] = None):
        """Render the complete world view with multiple panels"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('GR00T Simulated World View', fontsize=16, fontweight='bold')
        
        # 3D World View
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        self._render_3d_world(ax1)
        
        # Top-down view with trajectory
        ax2 = fig.add_subplot(2, 3, 2)
        self._render_top_view(ax2)
        
        # Attention heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        self._render_attention(ax3)
        
        # Action prediction
        ax4 = fig.add_subplot(2, 3, 4)
        self._render_actions(ax4)
        
        # Vision features
        ax5 = fig.add_subplot(2, 3, 5)
        self._render_vision_features(ax5)
        
        # Trust/Confidence scores
        ax6 = fig.add_subplot(2, 3, 6)
        self._render_confidence(ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 World view saved to {save_path}")
        
        plt.show()
        
    def _render_3d_world(self, ax):
        """Render 3D world view"""
        ax.set_title('3D World Model')
        ax.set_xlim(self.world_bounds[0])
        ax.set_ylim(self.world_bounds[1])
        ax.set_zlim(self.world_bounds[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Draw objects
        for obj in self.objects:
            if obj.object_type == 'static':
                alpha = 0.3
            elif obj.object_type == 'goal':
                alpha = 0.2
            else:
                alpha = 0.7 * obj.confidence
                
            # Draw as box
            self._draw_3d_box(ax, obj.position, obj.size, obj.color, alpha)
            
        # Draw robot
        ax.scatter(*self.robot_state.position, s=100, c='black', marker='o')
        
        # Draw trajectory
        if self.trajectory_plan:
            waypoints = np.array(self.trajectory_plan)
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                   'g--', linewidth=2, alpha=0.7, label='Planned Path')
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                      c='green', s=50, alpha=0.7)
                      
        ax.legend()
        
    def _draw_3d_box(self, ax, position, size, color, alpha):
        """Draw a 3D box"""
        x, y, z = position - size/2
        dx, dy, dz = size
        
        # Define the vertices of the box
        xx = [x, x, x+dx, x+dx, x, x, x+dx, x+dx]
        yy = [y, y+dy, y+dy, y, y, y+dy, y+dy, y]
        zz = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
        
        vertices = list(zip(xx, yy, zz))
        
        # Define the 6 faces
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black')
        ax.add_collection3d(poly)
        
    def _render_top_view(self, ax):
        """Render top-down view with trajectory"""
        ax.set_title('Top-Down View & Navigation')
        ax.set_xlim(self.world_bounds[0])
        ax.set_ylim(self.world_bounds[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Draw objects from above
        for obj in self.objects:
            if obj.object_type != 'goal':
                rect = Rectangle(
                    (obj.position[0] - obj.size[0]/2, 
                     obj.position[1] - obj.size[1]/2),
                    obj.size[0], obj.size[1],
                    facecolor=obj.color,
                    alpha=0.7 * obj.confidence,
                    edgecolor='black'
                )
                ax.add_patch(rect)
                ax.text(obj.position[0], obj.position[1], obj.name.split('_')[0],
                       ha='center', va='center', fontsize=8)
            else:
                # Draw goals as dashed rectangles
                rect = Rectangle(
                    (obj.position[0] - obj.size[0]/2, 
                     obj.position[1] - obj.size[1]/2),
                    obj.size[0], obj.size[1],
                    facecolor='none',
                    edgecolor=obj.color,
                    linestyle='--',
                    linewidth=2
                )
                ax.add_patch(rect)
                
        # Draw robot
        robot_circle = Circle(self.robot_state.position[:2], 0.05,
                             facecolor='black', edgecolor='yellow', linewidth=2)
        ax.add_patch(robot_circle)
        
        # Draw orientation arrow
        arrow_len = 0.15
        arrow = Arrow(self.robot_state.position[0], self.robot_state.position[1],
                     arrow_len * np.cos(self.robot_state.orientation),
                     arrow_len * np.sin(self.robot_state.orientation),
                     width=0.05, color='yellow')
        ax.add_patch(arrow)
        
        # Draw trajectory
        if self.trajectory_plan:
            waypoints = np.array(self.trajectory_plan)
            ax.plot(waypoints[:, 0], waypoints[:, 1], 'g--', linewidth=2, alpha=0.7)
            ax.scatter(waypoints[:, 0], waypoints[:, 1], c='green', s=30, zorder=5)
            
            # Number waypoints
            for i, wp in enumerate(waypoints):
                ax.text(wp[0]+0.05, wp[1]+0.05, str(i+1),
                       fontsize=8, color='green', fontweight='bold')
                       
        ax.grid(True, alpha=0.3)
        
    def _render_attention(self, ax):
        """Render attention heatmap"""
        ax.set_title('Attention Map')
        
        if self.attention_map is not None:
            im = ax.imshow(self.attention_map, cmap='hot', interpolation='bilinear',
                          extent=[self.world_bounds[0, 0], self.world_bounds[0, 1],
                                 self.world_bounds[1, 0], self.world_bounds[1, 1]])
            plt.colorbar(im, ax=ax, label='Attention Weight')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        else:
            ax.text(0.5, 0.5, 'No attention data\nRun perception first',
                   ha='center', va='center', transform=ax.transAxes)
            
    def _render_actions(self, ax):
        """Render predicted actions"""
        ax.set_title('Predicted Actions')
        
        if self.predicted_actions is not None:
            actions = self.predicted_actions[0][:16]  # Show first 16 actions
            indices = np.arange(len(actions))
            colors = ['green' if a > 0 else 'red' for a in actions]
            
            ax.bar(indices, actions, color=colors, alpha=0.7)
            ax.set_xlabel('Action Dimension')
            ax.set_ylabel('Action Value')
            ax.grid(True, alpha=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, 'No action predictions\nRun perception first',
                   ha='center', va='center', transform=ax.transAxes)
            
    def _render_vision_features(self, ax):
        """Render vision feature analysis"""
        ax.set_title('Vision Features (PCA)')
        
        if self.vision_features is not None:
            # Simulate PCA of vision features
            features = self.vision_features[0]
            
            # Create 2D projection (mock PCA)
            n_points = 50
            theta = np.linspace(0, 2*np.pi, n_points)
            
            # Different clusters for different object types
            for i, (label, color) in enumerate([('Objects', 'blue'), 
                                                ('Goals', 'green'),
                                                ('Obstacles', 'red')]):
                r = 1 + 0.3 * np.random.randn(n_points)
                x = r * np.cos(theta) + i * 2 - 2
                y = r * np.sin(theta) + np.random.randn(n_points) * 0.2
                ax.scatter(x, y, c=color, alpha=0.5, label=label, s=20)
                
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No vision features\nRun perception first',
                   ha='center', va='center', transform=ax.transAxes)
            
    def _render_confidence(self, ax):
        """Render object confidence scores"""
        ax.set_title('Object Confidence/Trust Scores')
        
        # Get object names and confidences
        names = [obj.name for obj in self.objects if obj.object_type != 'static']
        confidences = [obj.confidence for obj in self.objects if obj.object_type != 'static']
        colors_list = [obj.color for obj in self.objects if obj.object_type != 'static']
        
        if names:
            y_pos = np.arange(len(names))
            ax.barh(y_pos, confidences, color=colors_list, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('Confidence Score')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add threshold line
            ax.axvline(x=0.7, color='black', linestyle='--', alpha=0.5, label='Trust Threshold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No objects detected',
                   ha='center', va='center', transform=ax.transAxes)
            
    def animate_execution(self, duration: int = 10):
        """Animate task execution in real-time"""
        fig = plt.figure(figsize=(12, 8))
        
        # Create subplots
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)
        
        def update(frame):
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Update robot position along trajectory
            if self.trajectory_plan and len(self.trajectory_plan) > 0:
                waypoint_idx = min(frame // 20, len(self.trajectory_plan) - 1)
                target_pos = self.trajectory_plan[waypoint_idx]
                
                # Interpolate position
                alpha = 0.1
                self.robot_state.position = (1 - alpha) * self.robot_state.position + alpha * target_pos
                
                # Update gripper state
                if waypoint_idx == 1:  # Grasping
                    self.robot_state.gripper_state = min(1.0, self.robot_state.gripper_state + 0.1)
                elif waypoint_idx == 4:  # Releasing
                    self.robot_state.gripper_state = max(0.0, self.robot_state.gripper_state - 0.1)
                    
            # Render 3D world
            self._render_3d_world(ax1)
            ax1.set_title(f'Task Execution (t={frame/10:.1f}s)')
            
            # Render top view
            self._render_top_view(ax2)
            ax2.set_title('Navigation View')
            
            # Add status text
            status = f"Gripper: {'Closed' if self.robot_state.gripper_state > 0.5 else 'Open'}"
            ax2.text(0.02, 0.98, status, transform=ax2.transAxes,
                    fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
                    
            return ax1, ax2
            
        # Create animation
        anim = FuncAnimation(fig, update, frames=duration*10, interval=100, blit=False)
        
        plt.show()
        return anim
        

def main():
    """Main demo of GR00T world simulation"""
    print("\n" + "="*60)
    print("GR00T Simulated World View Demo")
    print("="*60)
    
    # Create simulator
    sim = GR00TWorldSimulator()
    
    # Run perception on a task
    sim.simulate_perception("Pick up the red cube and place it at the goal")
    
    # Render static world view
    print("\n📊 Rendering world view...")
    sim.render_world_view("groot_world_view.png")
    
    # Animate execution
    print("\n🎬 Starting task execution animation...")
    print("   Close the window to continue")
    anim = sim.animate_execution(duration=10)
    
    print("\n✅ Demo complete!")
    

if __name__ == "__main__":
    main()