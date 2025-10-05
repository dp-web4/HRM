#!/usr/bin/env python3
"""
Pragmatic SAGE Training - Fixed Version
=======================================
Real webcam + physics data training without consciousness cache bug
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import json
import pickle
from collections import deque
import cv2

@dataclass
class FixedTrainingConfig:
    """Fixed training config that works"""
    
    # Model scaling (fit in 16GB VRAM)
    sage_hidden_dim: int = 256  # Keep original size
    sage_layers: int = 6        # Keep original
    consciousness_enabled: bool = False  # DISABLED to avoid bug
    
    # Training schedule
    total_epochs: int = 20      # Reduced for demo
    steps_per_epoch: int = 100  # Reduced but meaningful
    batch_size: int = 2         # Small batch
    
    # Data strategy
    use_real_camera: bool = True
    use_physics_sim: bool = True
    
    # Hardware optimization
    device: str = "cuda"

class FixedDataGenerator:
    """Generate real training data without bugs"""
    
    def __init__(self, config: FixedTrainingConfig):
        self.config = config
        print("🔧 Setting up fixed data generator...")
        
        # Test camera
        self.has_camera = False
        if config.use_real_camera:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    self.has_camera = True
                    print("✅ Real camera available")
                cap.release()
            except:
                print("❌ No camera, using physics sim only")
    
    def generate_scenario(self) -> Dict:
        """Generate a single training scenario"""
        
        if self.has_camera and np.random.random() < 0.3:  # 30% real camera
            return self._generate_camera_scenario()
        else:
            return self._generate_physics_scenario()
    
    def _generate_camera_scenario(self) -> Dict:
        """Generate scenario from real camera"""
        cap = cv2.VideoCapture(0)
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return self._generate_physics_scenario()
        
        # Process frame
        frame = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Simple object detection via contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to objects
        objects = []
        for cnt in contours[:3]:  # Max 3 objects
            if cv2.contourArea(cnt) > 500:  # Filter noise
                x, y, w, h = cv2.boundingRect(cnt)
                objects.append({
                    'position': [x/224.0, y/224.0, 0.1],
                    'size': [w/224.0, h/224.0, 0.05],
                    'confidence': min(cv2.contourArea(cnt) / 5000.0, 1.0),
                    'graspable': (w/224.0 < 0.2 and h/224.0 < 0.2)
                })
        
        cap.release()
        
        return {
            'visual_data': frame_tensor,
            'objects': objects,
            'task': np.random.choice(['reach', 'grasp', 'track']),
            'source': 'real_camera'
        }
    
    def _generate_physics_scenario(self) -> Dict:
        """Generate physics-based scenario"""
        
        # Create realistic workspace
        num_objects = np.random.randint(1, 4)
        objects = []
        
        for i in range(num_objects):
            # Objects on table with realistic positions
            x = np.random.uniform(-0.3, 0.3)  # Table width
            y = np.random.uniform(0.2, 0.6)   # Table depth
            z = 0.02 + i * 0.05               # Stacked height
            
            size = np.random.uniform(0.03, 0.1, 3)
            
            objects.append({
                'position': [x, y, z],
                'size': size.tolist(),
                'confidence': np.random.uniform(0.7, 0.95),
                'graspable': np.all(size < 0.08)
            })
        
        # Generate structured visual (not random)
        visual = self._create_structured_visual(objects)
        
        # Choose appropriate task
        graspable_count = sum(1 for obj in objects if obj['graspable'])
        if graspable_count > 0:
            task = np.random.choice(['reach', 'grasp', 'place'])
        else:
            task = np.random.choice(['avoid', 'track'])
        
        return {
            'visual_data': visual,
            'objects': objects,
            'task': task,
            'source': 'physics_sim'
        }
    
    def _create_structured_visual(self, objects) -> torch.Tensor:
        """Create structured visual representation"""
        visual = torch.zeros(3, 224, 224)
        
        # Table background
        visual[0, 150:, :] = 0.4  # Brown table
        visual[1, 150:, :] = 0.3
        visual[2, 150:, :] = 0.2
        
        # Add objects as colored rectangles
        colors = {
            0: [0.8, 0.2, 0.2],  # Red
            1: [0.2, 0.8, 0.2],  # Green
            2: [0.2, 0.2, 0.8],  # Blue
        }
        
        for i, obj in enumerate(objects):
            x, y, z = obj['position']
            w, h, d = obj['size']
            
            # Map to image coordinates
            img_x = int((x + 0.5) * 224)  
            img_y = int((1.0 - y) * 224)  
            img_w = max(int(w * 300), 5)  # Minimum size
            img_h = max(int(h * 300), 5)
            
            # Ensure bounds
            img_x = max(0, min(img_x, 224-img_w))
            img_y = max(0, min(img_y, 224-img_h))
            img_w = min(img_w, 224-img_x)
            img_h = min(img_h, 224-img_y)
            
            # Draw object
            color = colors.get(i % 3, [0.5, 0.5, 0.5])
            for c in range(3):
                visual[c, img_y:img_y+img_h, img_x:img_x+img_w] = color[c]
        
        return visual

class FixedSAGETrainer:
    """Fixed trainer that actually works"""
    
    def __init__(self, config: FixedTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.data_generator = FixedDataGenerator(config)
        
        # Build working model
        self.model = self._build_working_sage()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        print(f"🚀 Fixed SAGE trainer ready with {self.model.param_count():,} parameters")
    
    def _build_working_sage(self):
        """Build SAGE model that works"""
        import sys
        sys.path.append('../core')
        from sage_federation_v1 import SAGE, SAGEConfig
        
        config = SAGEConfig(
            hidden_dim=self.config.sage_hidden_dim,
            h_level_dim=self.config.sage_hidden_dim // 2,
            l_level_dim=self.config.sage_hidden_dim // 2,
            num_heads=8,
            num_layers=self.config.sage_layers,
            vocab_size=1000,
            context_window=512
        )
        
        model = SAGE(config).to(self.device)
        
        # Explicitly disable consciousness cache
        try:
            model.consciousness_cache.enabled = False
        except:
            pass
        
        return model
    
    def encode_for_sage(self, scenario: Dict) -> torch.Tensor:
        """Encode scenario for SAGE (fixed version)"""
        features = []
        
        # Visual features from actual data
        visual = scenario['visual_data']
        visual_flat = visual.flatten()[:50]  # First 50 features
        features.extend(visual_flat.cpu().numpy())
        
        # Object features (up to 3 objects)
        for i in range(3):
            if i < len(scenario['objects']):
                obj = scenario['objects'][i]
                features.extend(obj['position'])
                features.append(obj['confidence'])
                features.append(1.0 if obj['graspable'] else 0.0)
            else:
                features.extend([0, 0, 0, 0, 0])  # Padding
        
        # Task encoding
        task_map = {'reach': 0, 'grasp': 1, 'avoid': 2, 'track': 3, 'place': 4}
        task_id = task_map.get(scenario['task'], 0)
        task_one_hot = [0] * 5
        task_one_hot[task_id] = 1
        features.extend(task_one_hot)
        
        # Source indicator
        source_val = 1.0 if scenario['source'] == 'real_camera' else 0.0
        features.append(source_val)
        
        # Pad to fixed size
        while len(features) < 100:
            features.append(0.0)
        features = features[:100]
        
        # Convert to tokens
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        input_ids = (feature_tensor * 100).long().clamp(0, 999)
        
        return input_ids.unsqueeze(0).to(self.device)
    
    def compute_realistic_loss(self, output: Dict, scenario: Dict) -> torch.Tensor:
        """Compute loss based on actual scenario"""
        device = output['h_ratio'].device
        losses = []
        
        # 1. Task complexity should match H-level usage
        task_complexity = {
            'reach': 0.3,
            'track': 0.4, 
            'avoid': 0.5,
            'grasp': 0.7,
            'place': 0.8
        }
        expected_h = task_complexity.get(scenario['task'], 0.5)
        h_loss = F.mse_loss(output['h_ratio'], torch.tensor(expected_h, device=device))
        losses.append(h_loss)
        
        # 2. Real camera data should get bonus (negative loss)
        if scenario['source'] == 'real_camera':
            real_data_bonus = torch.tensor(-0.1, device=device)
            losses.append(real_data_bonus)
        
        # 3. Object count should influence salience
        if 'objects' in scenario and scenario['objects']:
            salience = output['salience'].squeeze()
            target_salience = torch.tensor(
                min(len(scenario['objects']) / 3.0, 1.0), 
                device=device
            )
            if len(salience) > 0:
                salience_loss = F.mse_loss(salience[0], target_salience)
                losses.append(salience_loss)
        
        return sum(losses) / len(losses)
    
    def evaluate_on_scenarios(self, scenarios: List[Dict]) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        
        task_success = []
        h_usage = []
        real_data_perf = []
        
        with torch.no_grad():
            for scenario in scenarios[:20]:  # Sample evaluation
                try:
                    input_ids = self.encode_for_sage(scenario)
                    output = self.model(input_ids, use_consciousness=False)
                    
                    # Evaluate task success (heuristic)
                    h_ratio = output['h_ratio'].item()
                    task = scenario['task']
                    
                    # Success criteria
                    if task == 'grasp' and h_ratio > 0.6:
                        success = 1.0
                    elif task == 'reach' and h_ratio > 0.2:
                        success = 1.0
                    elif task in ['avoid', 'track'] and h_ratio > 0.3:
                        success = 1.0
                    else:
                        success = 0.0
                    
                    task_success.append(success)
                    h_usage.append(h_ratio)
                    
                    if scenario['source'] == 'real_camera':
                        real_data_perf.append(success)
                        
                except Exception as e:
                    print(f"⚠️  Evaluation error: {e}")
                    continue
        
        return {
            'task_success_rate': np.mean(task_success) if task_success else 0.0,
            'avg_h_usage': np.mean(h_usage) if h_usage else 0.0,
            'real_camera_success': np.mean(real_data_perf) if real_data_perf else 0.0,
            'scenarios_evaluated': len(task_success)
        }
    
    def train_fixed_sage(self):
        """Main training loop that works"""
        print("\n🎯 Starting Fixed Pragmatic SAGE Training")
        print("="*50)
        
        # Generate training data
        print("📊 Generating training scenarios...")
        train_scenarios = []
        for i in range(200):  # 200 scenarios
            if i % 50 == 0:
                print(f"  Generated {i}/200 scenarios")
            scenario = self.data_generator.generate_scenario()
            train_scenarios.append(scenario)
        
        print(f"✅ Generated {len(train_scenarios)} training scenarios")
        
        # Count data sources
        camera_count = sum(1 for s in train_scenarios if s['source'] == 'real_camera')
        physics_count = len(train_scenarios) - camera_count
        print(f"📷 Real camera: {camera_count} scenarios")
        print(f"⚗️  Physics sim: {physics_count} scenarios")
        
        # Training loop
        best_score = 0.0
        for epoch in range(self.config.total_epochs):
            self.model.train()
            epoch_losses = []
            
            # Training steps
            for step in range(self.config.steps_per_epoch):
                # Sample batch
                batch_scenarios = np.random.choice(train_scenarios, self.config.batch_size, replace=False)
                
                batch_loss = 0.0
                successful_steps = 0
                
                for scenario in batch_scenarios:
                    try:
                        input_ids = self.encode_for_sage(scenario)
                        output = self.model(input_ids, use_consciousness=False)
                        loss = self.compute_realistic_loss(output, scenario)
                        batch_loss += loss
                        successful_steps += 1
                    except Exception as e:
                        print(f"⚠️  Training step error: {e}")
                        continue
                
                if successful_steps > 0 and batch_loss.requires_grad:
                    batch_loss = batch_loss / successful_steps
                    
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    epoch_losses.append(batch_loss.item())
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            print(f"Epoch {epoch+1:2d}/{self.config.total_epochs}: Loss={avg_loss:.4f}")
            
            # Evaluation every 5 epochs
            if (epoch + 1) % 5 == 0:
                eval_results = self.evaluate_on_scenarios(train_scenarios)
                
                print(f"  📊 Evaluation:")
                print(f"    Task Success: {eval_results['task_success_rate']:.2%}")
                print(f"    H-Level Usage: {eval_results['avg_h_usage']:.3f}")
                if eval_results['real_camera_success'] is not None:
                    print(f"    Real Camera: {eval_results['real_camera_success']:.2%}")
                print(f"    Scenarios: {eval_results['scenarios_evaluated']}")
                
                # Track best model
                score = eval_results['task_success_rate']
                if score > best_score:
                    best_score = score
                    print(f"    🎯 New best score: {score:.2%}")
        
        print(f"\n✅ Fixed Training Complete!")
        print(f"📈 Best task success rate: {best_score:.2%}")
        print(f"🎥 Used real camera data: {self.data_generator.has_camera}")
        print(f"⚗️  Physics simulation scenarios included")
        
        return {
            'best_score': best_score,
            'used_real_camera': self.data_generator.has_camera,
            'total_scenarios': len(train_scenarios)
        }

def main():
    """Run fixed pragmatic training"""
    config = FixedTrainingConfig()
    trainer = FixedSAGETrainer(config)
    results = trainer.train_fixed_sage()
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Best Performance: {results['best_score']:.2%}")
    print(f"  Real Camera Used: {'✅' if results['used_real_camera'] else '❌'}")
    print(f"  Total Scenarios: {results['total_scenarios']}")

if __name__ == "__main__":
    main()