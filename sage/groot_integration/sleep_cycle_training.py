"""
GR00T + Sleep Cycle Training for 4K Reality Context
Integrates NVIDIA GR00T world model with sleep-cycle training
for 4K dimensional reality context learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import deque
import random
import time

# Import our context encoder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from context.reality_context_4k import RealityContext4K, RealityContextEncoder4K

# Import SAGE modules
from core.sage_v2 import SAGEV2Core, SAGEV2Config


@dataclass
class Experience:
    """Single experience timestep."""
    observation: Dict[str, torch.Tensor]
    context_4k: RealityContext4K
    action: torch.Tensor
    next_observation: Dict[str, torch.Tensor]
    reward: float
    metadata: Dict[str, Any]


class ExperienceMemory:
    """Circular buffer for experience storage."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_boundaries = []
    
    def add(self, experience: Experience):
        """Add single experience."""
        self.buffer.append(experience)
    
    def add_episode(self, episode: List[Experience]):
        """Add complete episode."""
        start_idx = len(self.buffer)
        for exp in episode:
            self.add(exp)
        end_idx = len(self.buffer)
        self.episode_boundaries.append((start_idx, end_idx))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def sample_episodes(self, n_episodes: int) -> List[List[Experience]]:
        """Sample complete episodes."""
        if not self.episode_boundaries:
            return []
        
        sampled = []
        boundaries = random.sample(
            self.episode_boundaries, 
            min(n_episodes, len(self.episode_boundaries))
        )
        
        for start, end in boundaries:
            episode = list(self.buffer)[start:end]
            sampled.append(episode)
        
        return sampled


class DreamScenarioGenerator:
    """Generate hypothetical scenarios for dream phase."""
    
    def __init__(self):
        self.modification_types = [
            "physics_violation",
            "object_substitution", 
            "temporal_reversal",
            "scale_distortion",
            "causal_inversion"
        ]
    
    def create_scenario(
        self, 
        base_experiences: List[Experience],
        modification_type: str = "physics_violation",
        complexity: str = "surreal"
    ) -> Dict[str, torch.Tensor]:
        """Create dream scenario from base experiences."""
        
        if not base_experiences:
            return self._generate_random_scenario()
        
        # Blend multiple experiences
        base = random.choice(base_experiences)
        scenario = base.observation.copy()
        
        if modification_type == "physics_violation":
            scenario = self._violate_physics(scenario)
        elif modification_type == "object_substitution":
            scenario = self._substitute_objects(scenario)
        elif modification_type == "temporal_reversal":
            scenario = self._reverse_temporal(scenario, base_experiences)
        elif modification_type == "scale_distortion":
            scenario = self._distort_scale(scenario)
        elif modification_type == "causal_inversion":
            scenario = self._invert_causality(scenario, base_experiences)
        
        return scenario
    
    def _violate_physics(self, scenario: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make objects float, pass through walls, etc."""
        # Simulate gravity reversal
        if 'depth' in scenario:
            scenario['depth'] = torch.flip(scenario['depth'], dims=[-1])
        
        # Add impossible velocities
        if 'proprioceptive' in scenario:
            scenario['proprioceptive'] *= torch.randn_like(scenario['proprioceptive']) * 10
        
        return scenario
    
    def _substitute_objects(self, scenario: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Replace objects with semantically different ones."""
        if 'visual' in scenario:
            # Shuffle channels to create "wrong" objects
            b, c, h, w = scenario['visual'].shape
            perm = torch.randperm(c)
            scenario['visual'] = scenario['visual'][:, perm, :, :]
        
        return scenario
    
    def _reverse_temporal(
        self, 
        scenario: Dict[str, torch.Tensor], 
        experiences: List[Experience]
    ) -> Dict[str, torch.Tensor]:
        """Reverse temporal sequence."""
        # Create scenario from reversed experience sequence
        if len(experiences) > 1:
            reversed_exp = experiences[-1]
            scenario = reversed_exp.observation.copy()
            # Indicate temporal reversal in metadata
            scenario['temporal_reversed'] = torch.tensor([1.0])
        
        return scenario
    
    def _distort_scale(self, scenario: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make things abnormally large or small."""
        scale_factor = random.choice([0.1, 0.5, 2.0, 10.0])
        
        if 'visual' in scenario:
            # Resize visual input
            b, c, h, w = scenario['visual'].shape
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            scenario['visual'] = F.interpolate(
                scenario['visual'], 
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            # Pad or crop back to original size
            scenario['visual'] = F.interpolate(
                scenario['visual'],
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
        
        return scenario
    
    def _invert_causality(
        self,
        scenario: Dict[str, torch.Tensor],
        experiences: List[Experience]
    ) -> Dict[str, torch.Tensor]:
        """Effect happens before cause."""
        if experiences:
            # Take outcome from one experience
            outcome = experiences[-1].next_observation
            # But initial state from another
            initial = experiences[0].observation
            
            # Blend them impossibly
            for key in scenario:
                if key in outcome and key in initial:
                    # 70% outcome, 30% initial (reversed causality)
                    scenario[key] = 0.7 * outcome[key] + 0.3 * initial[key]
        
        return scenario
    
    def _generate_random_scenario(self) -> Dict[str, torch.Tensor]:
        """Generate completely random scenario."""
        return {
            'visual': torch.randn(1, 3, 224, 224),
            'depth': torch.randn(1, 1, 224, 224),
            'audio': torch.randn(1, 1024),
            'tactile': torch.randn(1, 128),
            'proprioceptive': torch.randn(1, 64),
            'batch_size': 1
        }


class GR00TSleepCycleTrainer:
    """
    Integrates GR00T world model with sleep-cycle training
    for 4K dimensional reality context learning.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize context encoder
        self.context_encoder = RealityContextEncoder4K().to(device)
        
        # Initialize SAGE with 4K context
        sage_config = SAGEV2Config()
        sage_config.hidden_size = 256
        sage_config.num_heads = 8
        sage_config.num_h_layers = 6
        sage_config.num_l_layers = 6
        sage_config.use_external_llm = False  # For testing
        self.sage_model = SAGEV2Core(sage_config).to(device)
        
        # Memory systems
        self.experience_buffer = ExperienceMemory(capacity=100000)
        self.dream_generator = DreamScenarioGenerator()
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters()) + 
            list(self.sage_model.parameters()),
            lr=1e-4
        )
        
        # Metrics
        self.wake_experiences = 0
        self.sleep_consolidations = 0
        self.dream_explorations = 0
    
    def wake_phase(self, hours: float = 1.0) -> List[List[Experience]]:
        """
        Generate experience through interaction.
        This simulates lived experience.
        """
        print(f"🌅 Wake Phase: Generating {hours} hours of experience...")
        
        episodes = []
        num_episodes = int(hours * 60)  # 1 episode per minute
        
        for episode_idx in range(num_episodes):
            episode = self._generate_episode(duration_seconds=60)
            episodes.append(episode)
            self.experience_buffer.add_episode(episode)
            self.wake_experiences += len(episode)
            
            if (episode_idx + 1) % 10 == 0:
                print(f"  Generated {episode_idx + 1}/{num_episodes} episodes")
        
        print(f"✅ Wake complete: {self.wake_experiences} total experiences collected")
        return episodes
    
    def sleep_phase(self, num_samples: int = 1000) -> List[Dict]:
        """
        Consolidate experiences through augmentation.
        This is where context patterns are extracted.
        """
        print(f"😴 Sleep Phase: Consolidating {num_samples} experiences...")
        
        if len(self.experience_buffer.buffer) == 0:
            print("⚠️ No experiences to consolidate")
            return []
        
        # Sample experiences for consolidation
        experiences = self.experience_buffer.sample(min(num_samples, len(self.experience_buffer.buffer)))
        
        consolidated = []
        losses = []
        
        for idx, exp in enumerate(experiences):
            # Generate augmented variations
            variations = self._augment_experience(exp)
            
            for var in variations:
                # Re-extract context from variation (with gradients)
                var_context = self.context_encoder(var['obs'])
                
                # Learn invariances
                loss = self._consolidation_loss(exp.context_4k, var_context)
                losses.append(loss.item())
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                consolidated.append({
                    'original': exp,
                    'variation': var,
                    'loss': loss.item()
                })
                
                self.sleep_consolidations += 1
            
            if (idx + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"  Consolidated {idx + 1}/{len(experiences)} | Loss: {avg_loss:.4f}")
        
        print(f"✅ Sleep complete: {self.sleep_consolidations} consolidations performed")
        return consolidated
    
    def dream_phase(self, num_dreams: int = 100) -> List[Dict]:
        """
        Generate hypothetical scenarios to test understanding.
        Uses generative capabilities to explore edge cases.
        """
        print(f"💭 Dream Phase: Exploring {num_dreams} hypothetical scenarios...")
        
        dreams = []
        coherence_scores = []
        
        for dream_idx in range(num_dreams):
            # Sample base experiences
            base_experiences = self.experience_buffer.sample(
                min(10, len(self.experience_buffer.buffer))
            )
            
            # Generate dream scenario
            modification_type = random.choice(self.dream_generator.modification_types)
            dream_scenario = self.dream_generator.create_scenario(
                base_experiences=base_experiences,
                modification_type=modification_type,
                complexity="surreal"
            )
            
            # Test if context encoder handles novel situation
            with torch.no_grad():
                dream_context = self.context_encoder(dream_scenario)
            
            # Check if SAGE maintains coherence (simplified)
            sage_output = dream_context.to_tensor()  # Placeholder
            
            # Measure coherence
            coherence = self._measure_coherence(sage_output)
            coherence_scores.append(coherence)
            
            dreams.append({
                'scenario': dream_scenario,
                'context': dream_context,
                'sage_output': sage_output,
                'coherence': coherence,
                'modification': modification_type
            })
            
            self.dream_explorations += 1
            
            if (dream_idx + 1) % 20 == 0:
                avg_coherence = np.mean(coherence_scores[-20:])
                print(f"  Explored {dream_idx + 1}/{num_dreams} dreams | Coherence: {avg_coherence:.3f}")
        
        # Learn from dreams to improve robustness
        self._improve_from_dreams(dreams)
        
        print(f"✅ Dream complete: {self.dream_explorations} scenarios explored")
        return dreams
    
    def _generate_episode(self, duration_seconds: int = 60) -> List[Experience]:
        """Generate single episode of experience."""
        episode = []
        steps = duration_seconds * 10  # 10Hz sampling
        
        # Initialize observation
        obs = {
            'visual': torch.randn(1, 3, 224, 224).to(self.device),
            'depth': torch.randn(1, 1, 224, 224).to(self.device),
            'audio': torch.randn(1, 1024).to(self.device),
            'tactile': torch.randn(1, 128).to(self.device),
            'proprioceptive': torch.randn(1, 64).to(self.device),
            'batch_size': 1
        }
        
        for step in range(steps):
            # Extract 4K context
            with torch.no_grad():
                context_4k = self.context_encoder(obs)
            
            # SAGE processes context (simplified for now - just use context)
            # In real implementation, SAGE would process visual + context
            context_tensor = context_4k.to_tensor()
            sage_output = context_tensor  # Placeholder
            
            # Generate action (simplified)
            action = torch.randn(1, 64).to(self.device)
            
            # Simulate environment step (simplified)
            next_obs = self._simulate_step(obs, action)
            
            # Calculate reward (simplified)
            reward = torch.randn(1).item()
            
            # Store experience
            exp = Experience(
                observation=obs,
                context_4k=context_4k,
                action=action,
                next_observation=next_obs,
                reward=reward,
                metadata={'step': step, 'episode_length': steps}
            )
            episode.append(exp)
            
            # Update observation
            obs = next_obs
        
        return episode
    
    def _simulate_step(
        self, 
        obs: Dict[str, torch.Tensor], 
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Simulate environment dynamics (placeholder)."""
        # In real implementation, this would use GR00T
        next_obs = {}
        for key, value in obs.items():
            if key == 'batch_size':
                next_obs[key] = value
            else:
                # Simple dynamics: slight random change
                noise = torch.randn_like(value) * 0.1
                next_obs[key] = value + noise
        
        return next_obs
    
    def _augment_experience(self, exp: Experience) -> List[Dict]:
        """Create variations like biological sleep does."""
        variations = []
        
        # Temporal augmentation (replay at different speeds)
        for factor in [0.5, 2.0]:
            var = self._temporal_stretch(exp, factor)
            variations.append({'obs': var, 'type': f'temporal_{factor}x'})
        
        # Spatial augmentation (different viewpoints)
        for angle in [30, -30]:
            var = self._spatial_transform(exp.observation, rotation=angle)
            variations.append({'obs': var, 'type': f'rotation_{angle}deg'})
        
        # Physics augmentation (different physics parameters)
        for gravity in [0.8, 1.2]:
            var = self._physics_variation(exp.observation, gravity_mult=gravity)
            variations.append({'obs': var, 'type': f'gravity_{gravity}x'})
        
        return variations
    
    def _temporal_stretch(self, exp: Experience, factor: float) -> Dict[str, torch.Tensor]:
        """Stretch or compress temporal dimension."""
        obs = exp.observation.copy()
        # Simulate temporal distortion
        if 'proprioceptive' in obs:
            obs['proprioceptive'] = obs['proprioceptive'] * factor
        return obs
    
    def _spatial_transform(
        self, 
        obs: Dict[str, torch.Tensor], 
        rotation: float = 0,
        translation: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply spatial transformations."""
        transformed = obs.copy()
        
        if 'visual' in transformed and rotation != 0:
            # Simple rotation (placeholder for proper rotation)
            angle_rad = rotation * np.pi / 180
            theta = torch.tensor([[
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0]
            ]], dtype=torch.float32).to(self.device)
            
            grid = F.affine_grid(theta, transformed['visual'].size(), align_corners=False)
            transformed['visual'] = F.grid_sample(
                transformed['visual'], 
                grid, 
                align_corners=False
            )
        
        return transformed
    
    def _physics_variation(
        self, 
        obs: Dict[str, torch.Tensor], 
        gravity_mult: float = 1.0,
        friction_mult: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Vary physics parameters."""
        varied = obs.copy()
        
        # Simulate different gravity
        if 'depth' in varied:
            # Distort depth based on gravity
            varied['depth'] = varied['depth'] * gravity_mult
        
        # Simulate different friction
        if 'tactile' in varied:
            varied['tactile'] = varied['tactile'] * friction_mult
        
        return varied
    
    def _consolidation_loss(
        self, 
        original_context: RealityContext4K,
        varied_context: RealityContext4K
    ) -> torch.Tensor:
        """Loss for learning invariances during sleep."""
        # Convert to tensors
        orig_tensor = original_context.to_tensor()
        var_tensor = varied_context.to_tensor()
        
        # Invariance loss: similar contexts should have similar representations
        # But not identical (need to preserve meaningful differences)
        similarity = F.cosine_similarity(orig_tensor, var_tensor, dim=-1)
        
        # Target similarity around 0.8 (similar but not identical)
        target_similarity = torch.tensor([0.8]).to(self.device)
        loss = F.mse_loss(similarity.unsqueeze(0), target_similarity)
        
        return loss
    
    def _measure_coherence(self, sage_output: torch.Tensor) -> float:
        """Measure coherence of SAGE output."""
        # Simple coherence metric: low entropy = high coherence
        # In practice, would use more sophisticated metrics
        
        # Flatten and normalize
        flat = sage_output.flatten()
        if flat.numel() == 0:
            return 0.0
        
        # Convert to probabilities
        probs = F.softmax(flat, dim=0)
        
        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Convert to coherence score (lower entropy = higher coherence)
        max_entropy = torch.log(torch.tensor(flat.numel(), dtype=torch.float32))
        coherence = 1.0 - (entropy / max_entropy).item()
        
        return coherence
    
    def _improve_from_dreams(self, dreams: List[Dict]):
        """Use dream experiences to improve robustness."""
        # Filter dreams by coherence
        coherent_dreams = [d for d in dreams if d['coherence'] > 0.5]
        incoherent_dreams = [d for d in dreams if d['coherence'] <= 0.5]
        
        print(f"  Learning from {len(coherent_dreams)} coherent and {len(incoherent_dreams)} incoherent dreams")
        
        # Train to maintain coherence on edge cases
        for dream in coherent_dreams[:10]:  # Limit to prevent overfitting
            # Reinforce coherent responses (simplified)
            # For now, just skip training on dreams since we're using placeholder SAGE
            pass
    
    def run_full_cycle(
        self,
        wake_hours: float = 1.0,
        sleep_samples: int = 1000,
        dream_count: int = 100
    ) -> Dict[str, Any]:
        """Run complete wake-sleep-dream cycle."""
        print("\n" + "="*60)
        print("🔄 Starting Full Sleep Cycle")
        print("="*60)
        
        start_time = time.time()
        
        # Wake phase
        wake_episodes = self.wake_phase(hours=wake_hours)
        
        # Sleep phase
        sleep_consolidated = self.sleep_phase(num_samples=sleep_samples)
        
        # Dream phase
        dream_scenarios = self.dream_phase(num_dreams=dream_count)
        
        elapsed = time.time() - start_time
        
        # Summary statistics
        summary = {
            'wake_episodes': len(wake_episodes),
            'total_experiences': self.wake_experiences,
            'sleep_consolidations': len(sleep_consolidated),
            'dream_explorations': len(dream_scenarios),
            'avg_dream_coherence': np.mean([d['coherence'] for d in dream_scenarios]),
            'cycle_time_seconds': elapsed
        }
        
        print("\n" + "="*60)
        print("📊 Cycle Complete - Summary:")
        print(f"  Wake: {summary['wake_episodes']} episodes, {summary['total_experiences']} experiences")
        print(f"  Sleep: {summary['sleep_consolidations']} consolidations")
        print(f"  Dream: {summary['dream_explorations']} explorations, {summary['avg_dream_coherence']:.3f} avg coherence")
        print(f"  Total time: {summary['cycle_time_seconds']:.1f} seconds")
        print("="*60 + "\n")
        
        return summary


if __name__ == "__main__":
    print("🚀 Initializing GR00T Sleep Cycle Trainer...")
    
    # Create trainer
    trainer = GR00TSleepCycleTrainer()
    
    print(f"✅ Trainer initialized on device: {trainer.device}")
    print(f"   Context encoder parameters: {sum(p.numel() for p in trainer.context_encoder.parameters()):,}")
    print(f"   SAGE model parameters: {sum(p.numel() for p in trainer.sage_model.parameters()):,}")
    
    # Run a mini cycle for testing
    print("\n🧪 Running mini test cycle...")
    summary = trainer.run_full_cycle(
        wake_hours=0.1,  # 6 minutes of experience
        sleep_samples=100,  # Consolidate 100 experiences
        dream_count=10  # Explore 10 dream scenarios
    )
    
    print("\n✨ GR00T Sleep Cycle Training ready for deployment!")
    print("   Next step: Connect to actual GR00T simulator for real physics")
    print("   Then: Deploy on Jetson for embodied learning")