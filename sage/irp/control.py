"""
Control IRP Plugin Implementation
Version: 1.0 (2025-08-23)

Trajectory refinement with hard constraint projection for control tasks.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from .base import IRPPlugin, IRPState


class ControlIRP(IRPPlugin):
    """
    Control plugin for trajectory planning with constraint satisfaction.
    
    Key features:
    - Iterative trajectory refinement
    - Hard constraint projection for dynamics/safety
    - Feasibility-aware halting
    - Diffuser-style planning with guarantees
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Control IRP with dynamics model and constraints.
        
        Config parameters:
            - state_dim: Dimension of state space (default 4)
            - action_dim: Dimension of action space (default 2)
            - horizon: Planning horizon (default 50)
            - dt: Time step (default 0.1)
            - constraints: Dict of constraint functions
            - dynamics_model: Forward dynamics model
            - device: Compute device
        """
        super().__init__(config)
        
        self.state_dim = config.get('state_dim', 4)
        self.action_dim = config.get('action_dim', 2)
        self.horizon = config.get('horizon', 50)
        self.dt = config.get('dt', 0.1)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dynamics and constraints
        self.dynamics_model = config.get('dynamics_model') or self._default_dynamics
        self.constraints = config.get('constraints', {})
        
        # Default constraints if none provided
        if not self.constraints:
            self.constraints = {
                'state_bounds': lambda s: torch.clamp(s, -10, 10),
                'action_bounds': lambda a: torch.clamp(a, -1, 1),
                'obstacle_avoidance': self._default_obstacle_constraint
            }
        
        # Trajectory refiner network
        self.refiner = self._build_refiner()
        
        # Learning rate schedule for refinement
        self.lr_schedule = self._compute_lr_schedule()
        
        # Feasibility margin
        self.feasibility_margin = config.get('feasibility_margin', 0.1)
        
    def _build_refiner(self) -> nn.Module:
        """Build trajectory refinement network."""
        input_dim = (self.state_dim + self.action_dim) * self.horizon + 2 * self.state_dim + 1
        hidden_dim = 256
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim * self.horizon)
        )
    
    def _compute_lr_schedule(self) -> List[float]:
        """Compute learning rate schedule for refinement steps."""
        max_steps = self.config.get('max_iterations', 100)
        # Exponential decay
        return [0.1 * (0.95 ** i) for i in range(max_steps)]
    
    def _default_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Default simple dynamics model (double integrator).
        
        Args:
            state: Current state [B, state_dim]
            action: Control action [B, action_dim]
            
        Returns:
            Next state [B, state_dim]
        """
        # Simple double integrator dynamics
        # state = [position, velocity], action = acceleration
        batch_size = state.shape[0]
        
        if self.state_dim == 4 and self.action_dim == 2:
            # 2D double integrator
            pos = state[:, :2]
            vel = state[:, 2:]
            
            new_vel = vel + action * self.dt
            new_pos = pos + new_vel * self.dt
            
            return torch.cat([new_pos, new_vel], dim=-1)
        else:
            # Generic linear dynamics
            A = torch.eye(self.state_dim).to(self.device)
            B = torch.zeros(self.state_dim, self.action_dim).to(self.device)
            B[:self.action_dim, :] = torch.eye(self.action_dim).to(self.device) * self.dt
            
            return state @ A.t() + action @ B.t()
    
    def _default_obstacle_constraint(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Default obstacle avoidance constraint.
        
        Args:
            trajectory: State trajectory [B, H, state_dim]
            
        Returns:
            Constrained trajectory
        """
        # Simple circular obstacle at origin with radius 2
        positions = trajectory[..., :2]  # Extract positions
        distances = torch.norm(positions, dim=-1, keepdim=True)
        
        # Push away from obstacle if too close
        min_distance = 2.5
        violation_mask = distances < min_distance
        
        if violation_mask.any():
            # Compute repulsion direction
            repulsion = positions / (distances + 1e-6)
            repulsion = repulsion * (min_distance - distances).clamp(min=0)
            
            # Apply repulsion
            positions = positions + repulsion * violation_mask.float()
            trajectory[..., :2] = positions
        
        return trajectory
    
    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize trajectory from start and goal states.
        
        Args:
            x0: Dict with 'start' and 'goal' states
            task_ctx: Task context with constraints and objectives
            
        Returns:
            Initial IRPState with random trajectory
        """
        start_state = x0['start']
        goal_state = x0['goal']
        
        # Convert to tensors
        if isinstance(start_state, np.ndarray):
            start_state = torch.from_numpy(start_state).float()
        if isinstance(goal_state, np.ndarray):
            goal_state = torch.from_numpy(goal_state).float()
        
        start_state = start_state.to(self.device)
        goal_state = goal_state.to(self.device)
        
        # Add batch dimension if needed
        if start_state.dim() == 1:
            start_state = start_state.unsqueeze(0)
        if goal_state.dim() == 1:
            goal_state = goal_state.unsqueeze(0)
        
        batch_size = start_state.shape[0]
        
        # Initialize with random trajectory
        states = torch.zeros(batch_size, self.horizon, self.state_dim).to(self.device)
        actions = torch.randn(batch_size, self.horizon, self.action_dim).to(self.device) * 0.1
        
        # Linear interpolation as initial guess
        for t in range(self.horizon):
            alpha = t / (self.horizon - 1)
            states[:, t] = (1 - alpha) * start_state + alpha * goal_state
        
        # Store trajectory
        trajectory = {
            'states': states,
            'actions': actions
        }
        
        # Metadata
        meta = {
            'start_state': start_state,
            'goal_state': goal_state,
            'task_ctx': task_ctx,
            'cost_history': [],
            'feasibility_history': []
        }
        
        return IRPState(
            x=trajectory,
            step_idx=0,
            energy_val=None,
            meta=meta
        )
    
    def energy(self, state: IRPState) -> float:
        """
        Compute trajectory cost (energy).
        
        Includes:
        - Action cost (control effort)
        - Terminal cost (distance to goal)
        - Constraint violation penalties
        
        Args:
            state: Current refinement state with trajectory
            
        Returns:
            Scalar energy (trajectory cost)
        """
        trajectory = state.x
        states = trajectory['states']
        actions = trajectory['actions']
        goal = state.meta['goal_state']
        
        # Action cost (control effort)
        action_cost = torch.sum(actions ** 2) * 0.01
        
        # Terminal cost (distance to goal)
        terminal_state = states[:, -1]
        terminal_cost = torch.sum((terminal_state - goal) ** 2)
        
        # Constraint violation cost
        violation_cost = 0.0
        
        # Check dynamics consistency
        for t in range(self.horizon - 1):
            predicted_next = self.dynamics_model(states[:, t], actions[:, t])
            actual_next = states[:, t + 1]
            violation_cost += torch.sum((predicted_next - actual_next) ** 2)
        
        # Total cost
        total_cost = action_cost + terminal_cost + violation_cost * 10
        
        # Track history
        state.meta['cost_history'].append(total_cost.item())
        
        return float(total_cost.item())
    
    def project(self, state: IRPState) -> IRPState:
        """
        Project trajectory onto constraint manifold.
        
        Ensures:
        - Dynamics feasibility
        - State/action bounds
        - Obstacle avoidance
        
        Args:
            state: State with trajectory to project
            
        Returns:
            State with feasible trajectory
        """
        trajectory = state.x
        states = trajectory['states']
        actions = trajectory['actions']
        start = state.meta['start_state']
        
        # Project actions to bounds
        if 'action_bounds' in self.constraints:
            actions = self.constraints['action_bounds'](actions)
        
        # Forward simulate with projected actions to ensure dynamics consistency
        projected_states = torch.zeros_like(states)
        projected_states[:, 0] = start
        
        for t in range(self.horizon - 1):
            next_state = self.dynamics_model(projected_states[:, t], actions[:, t])
            
            # Apply state constraints
            if 'state_bounds' in self.constraints:
                next_state = self.constraints['state_bounds'](next_state)
            
            projected_states[:, t + 1] = next_state
        
        # Apply obstacle constraints to entire trajectory
        if 'obstacle_avoidance' in self.constraints:
            projected_states = self.constraints['obstacle_avoidance'](projected_states)
        
        # Update trajectory
        state.x = {
            'states': projected_states,
            'actions': actions
        }
        
        return state
    
    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Execute one trajectory refinement step.
        
        Args:
            state: Current state with trajectory
            noise_schedule: Optional noise schedule
            
        Returns:
            Refined trajectory state
        """
        trajectory = state.x
        states = trajectory['states']
        actions = trajectory['actions']
        start = state.meta['start_state']
        goal = state.meta['goal_state']
        step_idx = state.step_idx
        
        # Flatten trajectory for refiner input
        traj_flat = torch.cat([
            states.flatten(1),
            actions.flatten(1),
            start,
            goal,
            torch.tensor([[step_idx / 100.0]]).to(self.device)
        ], dim=-1)
        
        # Get refinement gradient
        with torch.no_grad():
            action_update = self.refiner(traj_flat)
            action_update = action_update.reshape(actions.shape)
        
        # Apply update with learning rate
        lr = self.lr_schedule[min(step_idx, len(self.lr_schedule) - 1)]
        new_actions = actions - lr * action_update
        
        # Add small noise for exploration (decreasing with steps)
        noise_scale = 0.1 * (1.0 - step_idx / self.config.get('max_iterations', 100))
        new_actions = new_actions + torch.randn_like(new_actions) * noise_scale
        
        # Create new state
        new_trajectory = {
            'states': states.clone(),
            'actions': new_actions
        }
        
        new_state = IRPState(
            x=new_trajectory,
            step_idx=step_idx + 1,
            energy_val=None,
            meta=state.meta
        )
        
        # Project to ensure feasibility
        new_state = self.project(new_state)
        
        return new_state
    
    def is_feasible(self, state: IRPState) -> bool:
        """
        Check if trajectory satisfies all constraints.
        
        Args:
            state: State with trajectory to check
            
        Returns:
            True if trajectory is feasible
        """
        trajectory = state.x
        states = trajectory['states']
        actions = trajectory['actions']
        start = state.meta['start_state']
        
        # Check initial condition
        if torch.norm(states[:, 0] - start) > 1e-3:
            return False
        
        # Check dynamics consistency
        for t in range(self.horizon - 1):
            predicted = self.dynamics_model(states[:, t], actions[:, t])
            actual = states[:, t + 1]
            if torch.norm(predicted - actual) > self.feasibility_margin:
                return False
        
        # Check bounds
        if torch.any(torch.abs(actions) > 1.0 + self.feasibility_margin):
            return False
        
        if torch.any(torch.abs(states) > 10.0 + self.feasibility_margin):
            return False
        
        return True
    
    def halt(self, history: List[IRPState]) -> bool:
        """
        Halt when trajectory is feasible and cost converges.
        
        Args:
            history: Refinement history
            
        Returns:
            True if should halt
        """
        if not history:
            return False
        
        current_state = history[-1]
        
        # Only consider halting if feasible
        if not self.is_feasible(current_state):
            # Don't halt if not feasible (unless max iterations)
            if len(history) >= self.config.get('max_iterations', 100):
                return True
            return False
        
        # Track feasibility
        current_state.meta['feasibility_history'].append(True)
        
        # Check cost convergence
        cost_history = current_state.meta.get('cost_history', [])
        if len(cost_history) >= 5:
            recent_costs = cost_history[-5:]
            cost_variance = np.var(recent_costs)
            if cost_variance < 0.01:
                return True
        
        return super().halt(history)
    
    def get_trajectory(self, state: IRPState) -> Dict[str, Any]:
        """
        Extract final trajectory and metrics.
        
        Args:
            state: Final refined state
            
        Returns:
            Dictionary with trajectory and metrics
        """
        trajectory = state.x
        
        return {
            'states': trajectory['states'].cpu().numpy(),
            'actions': trajectory['actions'].cpu().numpy(),
            'is_feasible': self.is_feasible(state),
            'final_cost': state.meta['cost_history'][-1] if state.meta['cost_history'] else float('inf'),
            'refinement_steps': state.step_idx,
            'terminal_error': torch.norm(
                trajectory['states'][:, -1] - state.meta['goal_state']
            ).item()
        }