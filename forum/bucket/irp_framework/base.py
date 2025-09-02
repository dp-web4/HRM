"""
IRP (Iterative Refinement Primitive) Base Interface
Version: 1.0 (2025-08-23)

Base classes and utilities for implementing IRP plugins.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import numpy as np


@dataclass
class IRPState:
    """State container for IRP refinement process."""
    x: Any                      # Plugin-specific state (latent, tokens, trajectory, etc.)
    step_idx: int = 0
    energy_val: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class IRPPlugin:
    """
    Base class for all IRP implementations.
    
    Every IRP plugin must define four invariants:
    1. State space: The representation being refined
    2. Noise model: How uncertainty is represented  
    3. Energy/distance metric: Measure of refinement progress
    4. Coherence contribution: Impact on system-level coherence
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize IRP plugin with configuration.
        
        Args:
            config: Plugin-specific configuration including:
                - entity_id: Unique identifier for this plugin instance
                - max_iterations: Maximum refinement steps allowed
                - halt_eps: Energy slope threshold for halting
                - halt_K: Number of steps to check for convergence
                - trust_weight: Initial trust weight
                - device: Compute device (cpu/cuda/jetson)
        """
        self.config = config
        self.entity_id = config.get('entity_id', f'{self.__class__.__name__}_{id(self)}')
        self.trust_weight = config.get('trust_weight', 1.0)
        self.energy_history = []
        self.time_history = []
        
    # ----- Core IRP Contract (must override) -----
    
    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize refinement state from input.
        
        Args:
            x0: Initial input (image, text, trajectory, etc.)
            task_ctx: Task context including objectives and constraints
            
        Returns:
            Initialized IRPState for refinement
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement init_state()")
    
    def energy(self, state: IRPState) -> float:
        """
        Compute energy/distance metric for current state.
        
        Lower energy indicates better refinement.
        Used for convergence detection and trust scoring.
        
        Args:
            state: Current refinement state
            
        Returns:
            Scalar energy value (lower is better)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement energy()")
    
    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Execute one refinement iteration.
        
        Args:
            state: Current state to refine
            noise_schedule: Optional schedule for noise injection/removal
            
        Returns:
            Updated state after one refinement step
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement step()")
    
    # ----- Optional Overrides -----
    
    def project(self, state: IRPState) -> IRPState:
        """
        Enforce constraints on state (dynamics/safety/feasibility).
        
        Default implementation is pass-through.
        Override for constraint satisfaction problems.
        
        Args:
            state: State to project onto constraint manifold
            
        Returns:
            Projected state satisfying constraints
        """
        return state
    
    def halt(self, history: List[IRPState]) -> bool:
        """
        Determine if refinement should stop.
        
        Default: halt when energy slope < eps for K steps
        OR maximum iterations reached.
        
        Args:
            history: List of states from refinement process
            
        Returns:
            True if refinement should halt
        """
        eps = self.config.get('halt_eps', 1e-4)
        K = self.config.get('halt_K', 3)
        max_iter = self.config.get('max_iterations', 100)
        
        if len(history) >= max_iter:
            return True
            
        if len(history) < K + 1:
            return False
            
        # Check energy slope over last K steps
        recent_energies = [s.energy_val or self.energy(s) for s in history[-(K+1):]]
        slope = abs(recent_energies[-1] - recent_energies[0]) / len(recent_energies)
        
        return slope < eps
    
    def get_halt_reason(self, history: List[IRPState]) -> str:
        """Determine why refinement halted."""
        if not history:
            return "no_history"
            
        eps = self.config.get('halt_eps', 1e-4)
        K = self.config.get('halt_K', 3)
        max_iter = self.config.get('max_iterations', 100)
        
        if len(history) >= max_iter:
            return "max_steps"
            
        if len(history) >= K + 1:
            recent_energies = [s.energy_val or self.energy(s) for s in history[-(K+1):]]
            slope = abs(recent_energies[-1] - recent_energies[0]) / len(recent_energies)
            if slope < eps:
                return "slope<eps"
                
        return "unknown"
    
    # ----- Telemetry and Trust -----
    
    def compute_trust_metrics(self, history: List[IRPState]) -> Dict[str, float]:
        """
        Compute trust metrics from refinement history.
        
        Args:
            history: List of states from refinement
            
        Returns:
            Dictionary with trust metrics:
                - monotonicity_ratio: Fraction of steps with energy decrease
                - dE_variance: Variance of energy changes
                - convergence_rate: Rate of energy decrease
        """
        if len(history) < 2:
            return {
                'monotonicity_ratio': 0.0,
                'dE_variance': float('inf'),
                'convergence_rate': 0.0
            }
            
        energies = [s.energy_val or self.energy(s) for s in history]
        dE_values = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        
        # Monotonicity: fraction of steps with energy decrease
        monotonic_steps = sum(1 for dE in dE_values if dE < 0)
        monotonicity = monotonic_steps / len(dE_values) if dE_values else 0.0
        
        # Variance of energy changes
        dE_variance = np.var(dE_values) if len(dE_values) > 1 else 0.0
        
        # Convergence rate (average energy decrease per step)
        total_decrease = energies[0] - energies[-1]
        convergence_rate = total_decrease / len(history) if len(history) > 0 else 0.0
        
        return {
            'monotonicity_ratio': float(monotonicity),
            'dE_variance': float(dE_variance),
            'convergence_rate': float(convergence_rate)
        }
    
    def emit_telemetry(self, state: IRPState, history: List[IRPState]) -> Dict[str, Any]:
        """
        Generate telemetry for Web4 integration.
        
        Args:
            state: Current state
            history: Full refinement history
            
        Returns:
            Telemetry dictionary conforming to schema
        """
        current_energy = state.energy_val or self.energy(state)
        
        # Compute dE if we have history
        dE = None
        if history and len(history) > 1:
            prev_energy = history[-2].energy_val or self.energy(history[-2])
            dE = current_energy - prev_energy
            
        # Compute timing
        time_ms = 0.0
        if history and len(history) > 1:
            time_ms = (state.timestamp - history[0].timestamp) * 1000
            
        # Compute ATP (simplified: proportional to steps and time)
        ATP_spent = len(history) * self.config.get('ATP_per_step', 0.1)
        
        return {
            'entity_id': self.entity_id,
            'plugin': self.__class__.__name__.lower().replace('plugin', ''),
            'step_idx': state.step_idx,
            'E': float(current_energy),
            'dE': float(dE) if dE is not None else None,
            'steps': len(history),
            'halt_reason': self.get_halt_reason(history) if self.halt(history) else None,
            'trust': self.compute_trust_metrics(history),
            'budget': {
                'ATP_spent': float(ATP_spent),
                'time_ms': float(time_ms),
                'memory_mb': state.meta.get('memory_mb', 0.0)
            },
            'LRC_context': self.config.get('LRC_context', None)
        }
    
    # ----- Convenience Methods -----
    
    def refine(self, x0: Any, task_ctx: Dict[str, Any] = None, 
               max_steps: Optional[int] = None) -> tuple[IRPState, List[IRPState]]:
        """
        Complete refinement process from initial input.
        
        Args:
            x0: Initial input
            task_ctx: Task context
            max_steps: Override max iterations
            
        Returns:
            Tuple of (final_state, history)
        """
        task_ctx = task_ctx or {}
        max_steps = max_steps or self.config.get('max_iterations', 100)
        
        # Initialize
        state = self.init_state(x0, task_ctx)
        state.energy_val = self.energy(state)
        history = [state]
        
        # Refine
        for step in range(max_steps):
            if self.halt(history):
                break
                
            # One refinement step
            state = self.step(state, noise_schedule=None)
            state = self.project(state)  # Apply constraints
            state.step_idx = step + 1
            state.energy_val = self.energy(state)
            state.timestamp = time.time()
            
            history.append(state)
            
        return state, history