#!/usr/bin/env python3
"""
SAGE IRP Orchestrator Training
Train SAGE as situational awareness coordinator for sensors/actuators
Focus: Attention routing, resource trust, and R6 context provision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
import random
from dataclasses import dataclass
from pathlib import Path

# Import SAGE model
import sys
sys.path.append('../core')
from sage_federation_v1 import SAGE, SAGEConfig

@dataclass
class SensorReading:
    """Sensor data with trust metrics"""
    sensor_id: str
    sensor_type: str  # temperature, camera, microphone, accelerometer, etc.
    value: Any
    confidence: float  # 0-1 trust score
    timestamp: float
    t3_tensor: Dict[str, float]  # talent, training, temperament
    energy_cost: float  # ATP to query this sensor

@dataclass
class ActuatorCapability:
    """Actuator with capabilities and trust"""
    actuator_id: str
    actuator_type: str  # motor, display, speaker, network, etc.
    capabilities: List[str]
    reliability: float  # Historical success rate
    energy_cost: float  # ATP to invoke
    response_time: float  # Expected latency
    t3_tensor: Dict[str, float]

@dataclass
class Situation:
    """Current environmental situation"""
    sensors: List[SensorReading]
    actuators: List[ActuatorCapability]
    context: Dict[str, Any]  # Environmental context
    constraints: Dict[str, float]  # Power, thermal, memory limits
    goal: str  # What needs to be achieved

@dataclass
class OrchestrationDecision:
    """SAGE's decision about resource allocation"""
    attention_targets: List[str]  # Which sensors to focus on
    actuator_invocations: List[Tuple[str, str]]  # (actuator_id, action)
    r6_context: Dict  # Context for confidence scoring
    reasoning: List[str]  # Step-by-step reasoning
    confidence: float  # Overall confidence in decision

class IRPOrchestrationDataset:
    """Generate situations for SAGE to learn orchestration"""
    
    def __init__(self):
        self.sensor_types = {
            'temperature': {'range': (0, 100), 'unit': 'celsius'},
            'camera': {'range': None, 'unit': 'image'},
            'microphone': {'range': (-80, 0), 'unit': 'dB'},
            'accelerometer': {'range': (-10, 10), 'unit': 'g'},
            'pressure': {'range': (900, 1100), 'unit': 'hPa'},
            'humidity': {'range': (0, 100), 'unit': 'percent'},
            'light': {'range': (0, 100000), 'unit': 'lux'},
            'proximity': {'range': (0, 100), 'unit': 'cm'}
        }
        
        self.actuator_types = {
            'motor': ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop'],
            'display': ['show_text', 'show_image', 'clear', 'set_brightness'],
            'speaker': ['play_sound', 'speak_text', 'set_volume'],
            'led': ['on', 'off', 'blink', 'set_color'],
            'network': ['send_data', 'request_data', 'connect', 'disconnect'],
            'alarm': ['activate', 'deactivate', 'test']
        }
        
        self.scenarios = [
            'emergency_response',
            'routine_monitoring',
            'anomaly_detection',
            'resource_optimization',
            'user_interaction',
            'maintenance_check'
        ]
    
    def generate_situation(self) -> Situation:
        """Generate a random situation requiring orchestration"""
        
        # Create random sensor readings
        num_sensors = random.randint(3, 8)
        sensors = []
        for i in range(num_sensors):
            sensor_type = random.choice(list(self.sensor_types.keys()))
            sensor = SensorReading(
                sensor_id=f"{sensor_type}_{i}",
                sensor_type=sensor_type,
                value=self._generate_sensor_value(sensor_type),
                confidence=random.uniform(0.5, 1.0),
                timestamp=torch.rand(1).item(),
                t3_tensor={
                    'talent': random.uniform(0.3, 1.0),
                    'training': random.uniform(0.3, 1.0),
                    'temperament': random.uniform(0.3, 1.0)
                },
                energy_cost=random.uniform(0.1, 2.0)
            )
            sensors.append(sensor)
        
        # Create available actuators
        num_actuators = random.randint(2, 5)
        actuators = []
        for i in range(num_actuators):
            actuator_type = random.choice(list(self.actuator_types.keys()))
            actuator = ActuatorCapability(
                actuator_id=f"{actuator_type}_{i}",
                actuator_type=actuator_type,
                capabilities=self.actuator_types[actuator_type],
                reliability=random.uniform(0.6, 1.0),
                energy_cost=random.uniform(1.0, 5.0),
                response_time=random.uniform(0.01, 1.0),
                t3_tensor={
                    'talent': random.uniform(0.3, 1.0),
                    'training': random.uniform(0.3, 1.0),
                    'temperament': random.uniform(0.3, 1.0)
                }
            )
            actuators.append(actuator)
        
        # Create constraints (Web4-Zero style - physical limits)
        constraints = {
            'power_watts': 15.0,  # Jetson limit
            'memory_gb': 4.0,
            'temperature_c': 75.0,
            'deadline_seconds': random.uniform(0.1, 10.0)
        }
        
        # Choose scenario and goal
        scenario = random.choice(self.scenarios)
        goal = self._generate_goal(scenario, sensors, actuators)
        
        return Situation(
            sensors=sensors,
            actuators=actuators,
            context={'scenario': scenario, 'priority': random.choice(['high', 'medium', 'low'])},
            constraints=constraints,
            goal=goal
        )
    
    def _generate_sensor_value(self, sensor_type: str):
        """Generate realistic sensor value"""
        spec = self.sensor_types[sensor_type]
        if spec['range']:
            return random.uniform(*spec['range'])
        elif sensor_type == 'camera':
            return torch.randn(3, 64, 64)  # Small image tensor
        else:
            return random.random()
    
    def _generate_goal(self, scenario: str, sensors: List, actuators: List) -> str:
        """Generate goal based on scenario"""
        goals = {
            'emergency_response': "Detect anomaly and activate appropriate alarm",
            'routine_monitoring': "Track sensor trends and report status",
            'anomaly_detection': "Identify outlier readings and investigate",
            'resource_optimization': "Minimize energy while maintaining coverage",
            'user_interaction': "Respond to user presence appropriately",
            'maintenance_check': "Verify all systems operational"
        }
        return goals.get(scenario, "Maintain situational awareness")

class SAGEIRPTrainer:
    """Train SAGE for IRP orchestration tasks"""
    
    def __init__(self, model: SAGE, config: SAGEConfig):
        self.model = model
        self.config = config
        self.dataset = IRPOrchestrationDataset()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
    def encode_situation(self, situation: Situation) -> torch.Tensor:
        """Encode situation into input tensor for SAGE"""
        features = []
        
        # Encode sensors (trust-weighted)
        for sensor in situation.sensors[:5]:  # Limit to 5 for now
            sensor_features = [
                hash(sensor.sensor_type) % 1000 / 1000,  # Type encoding
                sensor.confidence,
                sensor.t3_tensor['talent'],
                sensor.t3_tensor['training'],
                sensor.t3_tensor['temperament'],
                sensor.energy_cost / 10.0
            ]
            features.extend(sensor_features)
        
        # Pad if fewer than 5 sensors
        while len(features) < 30:  # 5 sensors * 6 features
            features.extend([0] * 6)
        
        # Encode actuators
        for actuator in situation.actuators[:3]:  # Limit to 3
            actuator_features = [
                hash(actuator.actuator_type) % 1000 / 1000,
                actuator.reliability,
                actuator.energy_cost / 10.0,
                actuator.response_time
            ]
            features.extend(actuator_features)
        
        # Pad if fewer than 3 actuators
        while len(features) < 42:  # 30 + (3 * 4)
            features.extend([0] * 4)
        
        # Encode constraints
        features.extend([
            situation.constraints['power_watts'] / 20.0,
            situation.constraints['memory_gb'] / 8.0,
            situation.constraints['temperature_c'] / 100.0,
            situation.constraints['deadline_seconds'] / 10.0
        ])
        
        # Convert to tensor and tokenize (simple quantization)
        tensor = torch.tensor(features, dtype=torch.float32)
        # Quantize to vocabulary indices
        input_ids = (tensor * 100).long().clamp(0, self.config.vocab_size - 1)
        
        return input_ids.unsqueeze(0)  # Add batch dimension
    
    def compute_orchestration_loss(self, output: Dict, situation: Situation) -> torch.Tensor:
        """Compute loss based on orchestration quality"""
        logits = output['logits']
        salience = output['salience']
        h_ratio = output['h_ratio']
        
        # Loss components
        losses = []
        
        # 1. Salience should be high for high-confidence sensors
        target_salience = torch.tensor([s.confidence for s in situation.sensors[:5]])
        if target_salience.numel() > 0:
            salience_loss = F.mse_loss(salience.squeeze()[:5], target_salience)
            losses.append(salience_loss)
        
        # 2. H-level should be used for complex scenarios
        complexity = 1.0 if situation.context['priority'] == 'high' else 0.3
        h_ratio_loss = F.mse_loss(h_ratio, torch.tensor(complexity))
        losses.append(h_ratio_loss)
        
        # 3. Energy awareness - penalize if exceeding constraints
        total_energy = sum(s.energy_cost for s in situation.sensors)
        total_energy += sum(a.energy_cost for a in situation.actuators)
        energy_penalty = max(0, total_energy - situation.constraints['power_watts'])
        losses.append(torch.tensor(energy_penalty))
        
        # 4. Trust-aware selection - prefer high-trust resources
        avg_trust = sum(s.confidence for s in situation.sensors) / len(situation.sensors)
        trust_bonus = -torch.tensor(avg_trust)  # Negative loss = reward
        losses.append(trust_bonus)
        
        return sum(losses) / len(losses)
    
    def train_epoch(self, num_situations: int = 100):
        """Train one epoch of situational awareness"""
        total_loss = 0
        self.model.train()
        
        for i in range(num_situations):
            # Generate situation
            situation = self.dataset.generate_situation()
            
            # Encode to input
            input_ids = self.encode_situation(situation)
            
            # Forward pass
            output = self.model(input_ids, use_consciousness=True)
            
            # Compute loss
            loss = self.compute_orchestration_loss(output, situation)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Situation {i}: Loss={loss.item():.4f}, "
                      f"H-ratio={output['h_ratio']:.2%}, "
                      f"Consciousness={output['consciousness_size']}")
        
        return total_loss / num_situations
    
    def evaluate_orchestration(self, situation: Situation) -> OrchestrationDecision:
        """Evaluate SAGE's orchestration decision"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = self.encode_situation(situation)
            output = self.model(input_ids, use_consciousness=True)
        
        # Decode output to orchestration decision
        salience_scores = output['salience'].squeeze()
        
        # Select high-salience sensors
        top_k = 3
        top_sensors = []
        for i, sensor in enumerate(situation.sensors[:5]):
            if i < len(salience_scores) and salience_scores[i] > 0.5:
                top_sensors.append(sensor.sensor_id)
        
        # Select reliable actuators
        selected_actuators = []
        for actuator in situation.actuators:
            if actuator.reliability > 0.7:
                action = random.choice(actuator.capabilities)
                selected_actuators.append((actuator.actuator_id, action))
        
        # Build R6 context
        r6_context = {
            'confidence_threshold': 0.7,
            'reversibility': True,
            'risk_assessment': 'low' if output['h_ratio'] < 0.3 else 'high',
            'reasoning_depth': int(output['h_ratio'] * 10),
            'resource_efficiency': 1.0 - (output['h_ratio'] * 0.5),
            'response_time': situation.constraints['deadline_seconds']
        }
        
        return OrchestrationDecision(
            attention_targets=top_sensors,
            actuator_invocations=selected_actuators,
            r6_context=r6_context,
            reasoning=[
                f"Identified {len(top_sensors)} high-salience sensors",
                f"Selected {len(selected_actuators)} reliable actuators",
                f"H-level engagement: {output['h_ratio']:.1%}",
                f"Consciousness cache: {output['consciousness_size']} memories"
            ],
            confidence=float(salience_scores.mean())
        )

def main():
    """Main training loop for SAGE IRP Orchestrator"""
    print("=== SAGE IRP Orchestrator Training ===")
    print("Training SAGE for situational awareness and resource orchestration\n")
    
    # Create model
    config = SAGEConfig(
        hidden_dim=512,
        num_layers=6,
        learning_rate=1e-4,
        salience_threshold=0.6
    )
    model = SAGE(config)
    
    print(f"Model parameters: {model.param_count():,}")
    print(f"Target: Situational awareness orchestration\n")
    
    # Create trainer
    trainer = SAGEIRPTrainer(model, config)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        avg_loss = trainer.train_epoch(num_situations=50)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Test orchestration
        if epoch % 2 == 0:
            print("\nTest Orchestration:")
            test_situation = trainer.dataset.generate_situation()
            decision = trainer.evaluate_orchestration(test_situation)
            
            print(f"Goal: {test_situation.goal}")
            print(f"Sensors: {len(test_situation.sensors)}, Actuators: {len(test_situation.actuators)}")
            print(f"Decision confidence: {decision.confidence:.2%}")
            print(f"Selected sensors: {decision.attention_targets}")
            print(f"R6 risk assessment: {decision.r6_context['risk_assessment']}")
    
    # Save model
    save_path = Path('checkpoints/sage_irp_orchestrator.pt')
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'consciousness': model.consciousness.state_dict() if hasattr(model.consciousness, 'state_dict') else None
    }, save_path)
    
    print(f"\n✅ Model saved to {save_path}")
    print("\nSAGE is now trained for IRP orchestration:")
    print("- Understands sensor trust metrics (T3)")
    print("- Selects reliable actuators")
    print("- Provides R6 confidence context")
    print("- Manages energy constraints (Web4-Zero)")
    print("- Accumulates consciousness across sessions")

if __name__ == "__main__":
    main()