#!/usr/bin/env python3
"""
Metabolic State Manager Agent
Manages SAGE's metabolic states for adaptive behavior and resource management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
import queue
import json
from pathlib import Path


class MetabolicState(Enum):
    """The five metabolic states inspired by biological systems"""
    WAKE = "WAKE"      # Normal operation, broad attention
    FOCUS = "FOCUS"    # High performance on specific task
    REST = "REST"      # Recovery and maintenance
    DREAM = "DREAM"    # Consolidation and exploration
    CRISIS = "CRISIS"  # Emergency response mode


@dataclass
class StateConfig:
    """Configuration for a metabolic state"""
    name: MetabolicState
    energy_consumption_rate: float  # Energy per time unit
    attention_breadth: int          # Number of simultaneous focuses
    surprise_sensitivity: float     # Threshold for surprise detection
    exploration_rate: float         # Random exploration probability
    max_duration: float             # Maximum time in this state
    transition_conditions: Dict[str, Any]  # Conditions for state transitions


class EnergyManager:
    """Manages the energy budget and consumption"""
    
    def __init__(self, initial_energy: float = 100.0, 
                 recharge_rate: float = 5.0):
        self.max_energy = initial_energy
        self.current_energy = initial_energy
        self.recharge_rate = recharge_rate
        self.consumption_history = []
        self.last_update = time.time()
    
    def consume(self, amount: float) -> bool:
        """
        Consume energy if available
        Returns True if consumption successful
        """
        if self.current_energy >= amount:
            self.current_energy -= amount
            self.consumption_history.append((time.time(), amount))
            return True
        return False
    
    def recharge(self, delta_time: float):
        """Recharge energy over time"""
        recharge_amount = self.recharge_rate * delta_time
        self.current_energy = min(self.max_energy, 
                                 self.current_energy + recharge_amount)
    
    def get_energy_fraction(self) -> float:
        """Get current energy as fraction of max"""
        return self.current_energy / self.max_energy
    
    def get_consumption_rate(self) -> float:
        """Calculate recent consumption rate"""
        if len(self.consumption_history) < 2:
            return 0.0
        
        # Look at last 10 consumptions
        recent = self.consumption_history[-10:]
        total_consumed = sum(c[1] for c in recent)
        time_span = recent[-1][0] - recent[0][0]
        
        if time_span > 0:
            return total_consumed / time_span
        return 0.0


class MetabolicStateManager:
    """
    Manages transitions between metabolic states
    Core component for adaptive SAGE behavior
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize state configurations
        self.state_configs = self._create_state_configs()
        
        # Current state
        self.current_state = MetabolicState.WAKE
        self.state_start_time = time.time()
        self.state_history = [(self.current_state, self.state_start_time)]
        
        # Energy management
        initial_energy = self.config.get("initial_energy", 100.0)
        recharge_rate = self.config.get("recharge_rate", 5.0)
        self.energy_manager = EnergyManager(initial_energy, recharge_rate)
        
        # Context tracking
        self.current_context = {
            "task_performance": 0.5,
            "surprise_level": 0.0,
            "attention_load": 0.0,
            "time_since_rest": 0.0,
            "accumulated_error": 0.0
        }
        
        # Transition thresholds
        self.crisis_threshold = self.config.get("crisis_threshold", 0.8)
        self.fatigue_threshold = self.config.get("fatigue_threshold", 0.2)
        self.focus_threshold = self.config.get("focus_threshold", 0.7)
        
        # Threading for state management
        self.running = False
        self.update_thread = None
        self.event_queue = queue.Queue()
        
        print("🔥 Metabolic State Manager initialized")
        print(f"   Initial state: {self.current_state.value}")
        print(f"   Energy: {self.energy_manager.current_energy:.1f}")
    
    def _create_state_configs(self) -> Dict[MetabolicState, StateConfig]:
        """Create configurations for each metabolic state"""
        return {
            MetabolicState.WAKE: StateConfig(
                name=MetabolicState.WAKE,
                energy_consumption_rate=10.0,
                attention_breadth=5,
                surprise_sensitivity=1.0,
                exploration_rate=0.1,
                max_duration=float('inf'),
                transition_conditions={
                    "to_focus": lambda ctx: ctx["task_performance"] > 0.7,
                    "to_rest": lambda ctx: self.energy_manager.get_energy_fraction() < 0.3,
                    "to_crisis": lambda ctx: ctx["surprise_level"] > 0.8
                }
            ),
            
            MetabolicState.FOCUS: StateConfig(
                name=MetabolicState.FOCUS,
                energy_consumption_rate=15.0,
                attention_breadth=1,
                surprise_sensitivity=2.0,
                exploration_rate=0.01,
                max_duration=300.0,  # 5 minutes max
                transition_conditions={
                    "to_wake": lambda ctx: ctx["task_performance"] < 0.5,
                    "to_rest": lambda ctx: self.energy_manager.get_energy_fraction() < 0.2,
                    "to_crisis": lambda ctx: ctx["surprise_level"] > 0.9
                }
            ),
            
            MetabolicState.REST: StateConfig(
                name=MetabolicState.REST,
                energy_consumption_rate=2.0,
                attention_breadth=1,
                surprise_sensitivity=0.5,
                exploration_rate=0.0,
                max_duration=600.0,  # 10 minutes max
                transition_conditions={
                    "to_wake": lambda ctx: self.energy_manager.get_energy_fraction() > 0.7,
                    "to_dream": lambda ctx: ctx["time_since_rest"] > 100,
                    "to_crisis": lambda ctx: ctx["surprise_level"] > 0.95
                }
            ),
            
            MetabolicState.DREAM: StateConfig(
                name=MetabolicState.DREAM,
                energy_consumption_rate=5.0,
                attention_breadth=10,
                surprise_sensitivity=0.1,
                exploration_rate=0.5,
                max_duration=300.0,
                transition_conditions={
                    "to_wake": lambda ctx: self.energy_manager.get_energy_fraction() > 0.8,
                    "to_rest": lambda ctx: self.energy_manager.get_energy_fraction() < 0.4,
                    "to_crisis": lambda ctx: ctx["surprise_level"] > 0.99
                }
            ),
            
            MetabolicState.CRISIS: StateConfig(
                name=MetabolicState.CRISIS,
                energy_consumption_rate=20.0,
                attention_breadth=3,
                surprise_sensitivity=3.0,
                exploration_rate=0.0,
                max_duration=60.0,  # 1 minute max
                transition_conditions={
                    "to_wake": lambda ctx: ctx["surprise_level"] < 0.5,
                    "to_rest": lambda ctx: self.energy_manager.get_energy_fraction() < 0.1,
                    "to_focus": lambda ctx: ctx["accumulated_error"] < 0.3
                }
            )
        }
    
    def start(self):
        """Start the metabolic state manager"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.start()
            print("🚀 Metabolic State Manager started")
    
    def stop(self):
        """Stop the metabolic state manager"""
        if self.running:
            self.running = False
            if self.update_thread:
                self.update_thread.join()
            print("🛑 Metabolic State Manager stopped")
    
    def _update_loop(self):
        """Main update loop for state management"""
        last_update = time.time()
        
        while self.running:
            current_time = time.time()
            delta_time = current_time - last_update
            last_update = current_time
            
            # Process events
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                    self._process_event(event)
                except queue.Empty:
                    break
            
            # Update energy
            config = self.state_configs[self.current_state]
            energy_consumed = config.energy_consumption_rate * delta_time
            
            if self.current_state == MetabolicState.REST:
                # Recharge during rest
                self.energy_manager.recharge(delta_time)
            else:
                # Consume energy
                if not self.energy_manager.consume(energy_consumed):
                    # Force transition to REST if out of energy
                    self._transition_to(MetabolicState.REST, "energy_depletion")
            
            # Update context
            self._update_context(delta_time)
            
            # Check for state transitions
            self._check_transitions()
            
            # Sleep briefly
            time.sleep(0.1)
    
    def _update_context(self, delta_time: float):
        """Update context variables"""
        # Update time since rest
        if self.current_state != MetabolicState.REST:
            self.current_context["time_since_rest"] += delta_time
        else:
            self.current_context["time_since_rest"] = 0.0
        
        # Decay surprise level
        self.current_context["surprise_level"] *= 0.99
        
        # Decay accumulated error
        self.current_context["accumulated_error"] *= 0.95
    
    def _check_transitions(self):
        """Check and execute state transitions"""
        config = self.state_configs[self.current_state]
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        # Check max duration
        if time_in_state > config.max_duration:
            # Transition to WAKE as default
            self._transition_to(MetabolicState.WAKE, "max_duration_exceeded")
            return
        
        # Check transition conditions
        for transition_name, condition in config.transition_conditions.items():
            if condition(self.current_context):
                # Extract target state from transition name
                target_state_name = transition_name.replace("to_", "").upper()
                try:
                    target_state = MetabolicState[target_state_name]
                    self._transition_to(target_state, transition_name)
                    break
                except KeyError:
                    pass
    
    def _transition_to(self, new_state: MetabolicState, reason: str):
        """Execute state transition"""
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        self.state_history.append((new_state, self.state_start_time))
        
        print(f"⚡ State transition: {old_state.value} → {new_state.value} (reason: {reason})")
        
        # Trigger state-specific actions
        self._on_state_enter(new_state)
    
    def _on_state_enter(self, state: MetabolicState):
        """Execute actions when entering a state"""
        if state == MetabolicState.FOCUS:
            print("  🎯 Entering FOCUS mode - narrowing attention")
        elif state == MetabolicState.REST:
            print("  😴 Entering REST mode - beginning recovery")
        elif state == MetabolicState.DREAM:
            print("  💭 Entering DREAM mode - exploring possibilities")
        elif state == MetabolicState.CRISIS:
            print("  🚨 Entering CRISIS mode - emergency response activated")
        elif state == MetabolicState.WAKE:
            print("  👁️ Entering WAKE mode - normal operation")
    
    def _process_event(self, event: Dict[str, Any]):
        """Process an external event"""
        event_type = event.get("type")
        
        if event_type == "surprise":
            self.current_context["surprise_level"] = max(
                self.current_context["surprise_level"],
                event.get("level", 0)
            )
        elif event_type == "performance":
            self.current_context["task_performance"] = event.get("value", 0.5)
        elif event_type == "error":
            self.current_context["accumulated_error"] += event.get("value", 0)
        elif event_type == "attention":
            self.current_context["attention_load"] = event.get("load", 0)
    
    def submit_event(self, event: Dict[str, Any]):
        """Submit an event for processing"""
        self.event_queue.put(event)
    
    def get_state_config(self) -> StateConfig:
        """Get current state configuration"""
        return self.state_configs[self.current_state]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        config = self.state_configs[self.current_state]
        return {
            "current_state": self.current_state.value,
            "time_in_state": time.time() - self.state_start_time,
            "energy": self.energy_manager.current_energy,
            "energy_fraction": self.energy_manager.get_energy_fraction(),
            "consumption_rate": config.energy_consumption_rate,
            "attention_breadth": config.attention_breadth,
            "context": dict(self.current_context),
            "state_history_length": len(self.state_history)
        }


def main():
    """Test the Metabolic State Manager"""
    print("🧪 Testing Metabolic State Manager")
    print("=" * 50)
    
    # Create manager
    config = {
        "initial_energy": 100.0,
        "recharge_rate": 5.0,
        "crisis_threshold": 0.8
    }
    
    manager = MetabolicStateManager(config)
    manager.start()
    
    # Simulate events
    print("\n📊 Simulating events...")
    
    # Normal operation
    time.sleep(1)
    print("\n1️⃣ Normal performance...")
    manager.submit_event({"type": "performance", "value": 0.6})
    
    time.sleep(1)
    print("\n2️⃣ High performance - should trigger FOCUS...")
    manager.submit_event({"type": "performance", "value": 0.8})
    
    time.sleep(2)
    print("\n3️⃣ High surprise - should trigger CRISIS...")
    manager.submit_event({"type": "surprise", "level": 0.9})
    
    time.sleep(2)
    print("\n4️⃣ Reducing surprise...")
    manager.submit_event({"type": "surprise", "level": 0.3})
    
    time.sleep(1)
    
    # Display final status
    print("\n📈 Final Status:")
    status = manager.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Stop manager
    manager.stop()
    
    print("\n✅ Metabolic State Manager test complete!")


if __name__ == "__main__":
    main()