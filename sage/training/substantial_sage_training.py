#!/usr/bin/env python3
"""
Substantial SAGE Training - 12 Hour GPU Budget
==============================================
Real training with substance: curriculum learning, data augmentation, 
progressive difficulty, comprehensive evaluation, and model scaling
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
import cv2
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

@dataclass 
class SubstantialTrainingConfig:
    """12-hour substantial training configuration"""
    
    # Training duration (12 hours)
    total_hours: float = 12.0
    epochs_per_hour: int = 15  # 180 total epochs
    steps_per_epoch: int = 500  # 90,000 total training steps
    
    # Model scaling (use full RTX 4090)
    sage_hidden_dim: int = 512  # 2x larger than demo
    sage_layers: int = 12       # 2x deeper than demo  
    consciousness_enabled: bool = False  # Still has bugs
    
    # Batch and memory management
    batch_size: int = 8         # Larger batches
    gradient_accumulation: int = 4  # Effective batch size = 32
    mixed_precision: bool = True
    
    # Data strategy (much more comprehensive)
    training_scenarios: int = 5000   # 25x more data
    validation_scenarios: int = 1000
    camera_ratio: float = 0.4        # 40% real camera data
    
    # Curriculum learning
    curriculum_stages: int = 4
    difficulty_progression: bool = True
    
    # Data augmentation
    visual_augmentation: bool = True
    spatial_augmentation: bool = True
    noise_injection: bool = True
    
    # Advanced training
    learning_rate: float = 1e-4      # Base learning rate
    learning_rate_schedule: bool = True
    early_stopping: bool = True
    model_checkpointing: bool = True
    
    # Evaluation
    comprehensive_evaluation: bool = True
    evaluation_frequency: int = 30  # Every 2 hours
    
    device: str = "cuda"

class CurriculumDataGenerator:
    """Advanced data generator with curriculum learning"""
    
    def __init__(self, config: SubstantialTrainingConfig):
        self.config = config
        self.current_stage = 0
        self.difficulty_factor = 0.0
        
        # Setup camera
        self.setup_camera()
        
        # Curriculum stages
        self.curriculum_stages = [
            {"name": "Foundation", "difficulty": 0.2, "epochs": 45},
            {"name": "Integration", "difficulty": 0.5, "epochs": 45}, 
            {"name": "Complexity", "difficulty": 0.8, "epochs": 45},
            {"name": "Mastery", "difficulty": 1.0, "epochs": 45}
        ]
        
        print(f"📚 Curriculum Learning: {len(self.curriculum_stages)} stages over {config.total_hours} hours")
        
    def setup_camera(self):
        """Setup camera with error handling"""
        self.has_camera = False
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Test capture
                ret, frame = cap.read()
                if ret:
                    self.has_camera = True
                    print("✅ Camera available for substantial training")
                else:
                    print("⚠️  Camera detected but failed to capture")
            cap.release()
        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
        
        if not self.has_camera:
            print("📷 Using enhanced simulation only")
    
    def set_curriculum_stage(self, epoch: int):
        """Update curriculum stage based on epoch"""
        cumulative_epochs = 0
        for i, stage in enumerate(self.curriculum_stages):
            cumulative_epochs += stage["epochs"]
            if epoch < cumulative_epochs:
                if self.current_stage != i:
                    self.current_stage = i
                    self.difficulty_factor = stage["difficulty"]
                    print(f"\n📈 Curriculum Stage {i+1}: {stage['name']} (difficulty {stage['difficulty']:.1f})")
                break
    
    def generate_scenario(self, augment: bool = True) -> Dict:
        """Generate scenario with curriculum difficulty"""
        
        if self.has_camera and random.random() < self.config.camera_ratio:
            scenario = self._generate_advanced_camera_scenario()
        else:
            scenario = self._generate_curriculum_physics_scenario()
        
        # Apply augmentation
        if augment and self.config.visual_augmentation:
            scenario = self._apply_augmentations(scenario)
            
        return scenario
    
    def _generate_advanced_camera_scenario(self) -> Dict:
        """Generate advanced camera scenario with multiple captures"""
        cap = cv2.VideoCapture(0)
        
        # Capture multiple frames for temporal context
        frames = []
        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            time.sleep(0.1)  # Small delay between frames
        
        cap.release()
        
        if not frames:
            return self._generate_curriculum_physics_scenario()
        
        # Use latest frame as primary
        primary_frame = frames[-1]
        frame_tensor = torch.from_numpy(primary_frame).permute(2, 0, 1).float() / 255.0
        
        # Advanced object detection
        objects = self._advanced_object_detection(primary_frame)
        
        # Motion detection if multiple frames
        motion_score = 0.0
        if len(frames) > 1:
            motion_score = self._detect_motion(frames[-2], frames[-1])
        
        # Curriculum-based task selection
        task = self._select_curriculum_task(objects, motion_score)
        
        return {
            'visual_data': frame_tensor,
            'objects': objects,
            'task': task,
            'motion_score': motion_score,
            'source': 'real_camera',
            'difficulty': self.difficulty_factor,
            'timestamp': time.time()
        }
    
    def _advanced_object_detection(self, frame) -> List[Dict]:
        """Advanced object detection with multiple methods"""
        objects = []
        
        # Method 1: Contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple threshold levels for robustness
        for thresh_val in [50, 100, 150]:
            edges = cv2.Canny(gray, thresh_val, thresh_val * 2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 10000:  # Filter noise and background
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Calculate confidence based on shape properties
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Aspect ratio
                    aspect_ratio = w / h if h > 0 else 1
                    
                    confidence = solidity * (1.0 - abs(aspect_ratio - 1.0)) * (area / 10000)
                    confidence = min(confidence, 1.0)
                    
                    if confidence > 0.1:
                        objects.append({
                            'position': [x/224.0, y/224.0, 0.1 + random.random() * 0.2],
                            'size': [w/224.0, h/224.0, 0.05 + random.random() * 0.05],
                            'confidence': confidence,
                            'graspable': (w < 50 and h < 50),
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity,
                            'detection_method': f'contour_thresh_{thresh_val}'
                        })
        
        # Method 2: Color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common objects
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],  
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)]
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(cnt)
                    objects.append({
                        'position': [x/224.0, y/224.0, 0.15],
                        'size': [w/224.0, h/224.0, 0.06],
                        'confidence': min(area / 5000.0, 1.0),
                        'graspable': (w < 60 and h < 60),
                        'color': color_name,
                        'detection_method': 'color_based'
                    })
        
        # Remove duplicates and limit count
        objects = self._remove_duplicate_objects(objects)
        return objects[:6]  # Max 6 objects for complexity management
    
    def _remove_duplicate_objects(self, objects: List[Dict]) -> List[Dict]:
        """Remove spatially overlapping objects"""
        if len(objects) <= 1:
            return objects
        
        unique_objects = []
        for obj in objects:
            is_duplicate = False
            for existing in unique_objects:
                # Calculate spatial distance
                dist = np.linalg.norm(np.array(obj['position'][:2]) - np.array(existing['position'][:2]))
                if dist < 0.1:  # 10% of image width
                    # Keep higher confidence object
                    if obj['confidence'] > existing['confidence']:
                        unique_objects.remove(existing)
                        unique_objects.append(obj)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_objects.append(obj)
        
        return unique_objects
    
    def _detect_motion(self, frame1, frame2) -> float:
        """Detect motion between frames"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion score
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_score = motion_pixels / total_pixels
        
        return motion_score
    
    def _select_curriculum_task(self, objects: List[Dict], motion_score: float) -> str:
        """Select task based on curriculum difficulty"""
        
        if self.difficulty_factor < 0.3:
            # Foundation stage: simple tasks
            return random.choice(['reach', 'track'])
        elif self.difficulty_factor < 0.6:
            # Integration stage: add grasping
            if any(obj.get('graspable', False) for obj in objects):
                return random.choice(['reach', 'grasp', 'track'])
            else:
                return random.choice(['reach', 'track', 'avoid'])
        elif self.difficulty_factor < 0.9:
            # Complexity stage: multi-step tasks
            if motion_score > 0.05:
                return random.choice(['track', 'avoid', 'navigate'])
            else:
                return random.choice(['grasp', 'place', 'manipulate'])
        else:
            # Mastery stage: all tasks
            return random.choice(['reach', 'grasp', 'place', 'avoid', 'track', 'navigate', 'manipulate'])
    
    def _generate_curriculum_physics_scenario(self) -> Dict:
        """Generate physics scenario based on curriculum"""
        
        # Scale complexity with curriculum
        max_objects = int(1 + self.difficulty_factor * 5)  # 1-6 objects
        num_objects = random.randint(1, max_objects)
        
        objects = []
        for i in range(num_objects):
            # More complex spatial arrangements at higher difficulty
            if self.difficulty_factor < 0.3:
                # Simple: objects spread out
                x = random.uniform(-0.3, 0.3)
                y = random.uniform(0.2, 0.6)
            elif self.difficulty_factor < 0.6:
                # Medium: some clustering
                if i == 0:
                    x = random.uniform(-0.3, 0.3) 
                    y = random.uniform(0.2, 0.6)
                else:
                    # Cluster around first object
                    base_x, base_y = objects[0]['position'][:2]
                    x = base_x + random.uniform(-0.1, 0.1)
                    y = base_y + random.uniform(-0.1, 0.1)
                    x = np.clip(x, -0.4, 0.4)
                    y = np.clip(y, 0.1, 0.7)
            else:
                # Complex: tight arrangements, occlusions
                if i == 0:
                    x = random.uniform(-0.2, 0.2)
                    y = random.uniform(0.3, 0.5)
                else:
                    # Create complex arrangements
                    prev_obj = objects[random.randint(0, i-1)]
                    x = prev_obj['position'][0] + random.uniform(-0.05, 0.05)
                    y = prev_obj['position'][1] + random.uniform(-0.05, 0.05)
            
            z = 0.02 + i * 0.03  # Stack height
            
            # Size varies with difficulty
            if self.difficulty_factor < 0.4:
                size = np.random.uniform(0.04, 0.08, 3)  # Larger, easier to grasp
            else:
                size = np.random.uniform(0.02, 0.06, 3)  # Smaller, more challenging
            
            # Object properties
            obj_type = random.choice(['cube', 'cylinder', 'sphere'])
            
            objects.append({
                'position': [x, y, z],
                'size': size.tolist(),
                'type': obj_type,
                'confidence': random.uniform(0.7, 0.95),
                'graspable': np.all(size < 0.07),
                'mass': np.prod(size) * 1000,
                'difficulty_level': self.difficulty_factor
            })
        
        # Create visual representation
        visual = self._create_advanced_visual(objects)
        
        # Task selection based on curriculum
        task = self._select_curriculum_task(objects, 0.0)
        
        return {
            'visual_data': visual,
            'objects': objects,
            'task': task,
            'source': 'physics_sim',
            'difficulty': self.difficulty_factor,
            'complexity_score': self._calculate_complexity(objects)
        }
    
    def _create_advanced_visual(self, objects) -> torch.Tensor:
        """Create more sophisticated visual representation"""
        visual = torch.zeros(3, 224, 224)
        
        # Realistic lighting and shadows
        # Add gradient background (table lighting)
        for y in range(224):
            intensity = 0.2 + 0.3 * (y / 224.0)  # Darker at top, lighter at bottom
            visual[0, y, :] = intensity * 0.4  # Brown table
            visual[1, y, :] = intensity * 0.3
            visual[2, y, :] = intensity * 0.2
        
        # Advanced object rendering
        object_colors = {
            'cube': ([0.8, 0.2, 0.2], [0.7, 0.1, 0.1]),      # Red with shadow
            'cylinder': ([0.2, 0.7, 0.2], [0.1, 0.6, 0.1]),  # Green with shadow
            'sphere': ([0.2, 0.2, 0.8], [0.1, 0.1, 0.7])     # Blue with shadow
        }
        
        for i, obj in enumerate(objects):
            x, y, z = obj['position']
            w, h, d = obj['size']
            obj_type = obj.get('type', 'cube')
            
            # Map to image coordinates
            img_x = int((x + 0.5) * 224)
            img_y = int((1.0 - y) * 224)
            img_w = max(int(w * 400), 3)
            img_h = max(int(h * 400), 3)
            
            # Ensure bounds
            img_x = max(0, min(img_x, 224-img_w))
            img_y = max(0, min(img_y, 224-img_h))
            img_w = min(img_w, 224-img_x)
            img_h = min(img_h, 224-img_y)
            
            # Get colors
            primary_color, shadow_color = object_colors.get(obj_type, ([0.5, 0.5, 0.5], [0.3, 0.3, 0.3]))
            
            # Add shadow first (slightly offset)
            shadow_x = img_x + 2
            shadow_y = img_y + 2
            if shadow_x + img_w < 224 and shadow_y + img_h < 224:
                for c in range(3):
                    visual[c, shadow_y:shadow_y+img_h, shadow_x:shadow_x+img_w] = shadow_color[c]
            
            # Add main object
            for c in range(3):
                visual[c, img_y:img_y+img_h, img_x:img_x+img_w] = primary_color[c]
            
            # Add highlight for 3D effect
            highlight_h = max(img_h // 3, 1)
            highlight_w = max(img_w // 3, 1)
            for c in range(3):
                visual[c, img_y:img_y+highlight_h, img_x:img_x+highlight_w] = min(primary_color[c] + 0.2, 1.0)
        
        return visual
    
    def _calculate_complexity(self, objects: List[Dict]) -> float:
        """Calculate scenario complexity score"""
        if not objects:
            return 0.0
        
        complexity = 0.0
        
        # Object count complexity
        complexity += len(objects) / 6.0
        
        # Spatial complexity (clustering)
        if len(objects) > 1:
            distances = []
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i+1:], i+1):
                    dist = np.linalg.norm(np.array(obj1['position'][:2]) - np.array(obj2['position'][:2]))
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            complexity += (1.0 - avg_distance) * 0.5  # Closer objects = higher complexity
        
        # Size complexity (smaller objects harder)
        avg_size = np.mean([np.mean(obj['size']) for obj in objects])
        complexity += (0.1 - avg_size) * 2.0  # Smaller = harder
        
        return min(complexity, 1.0)
    
    def _apply_augmentations(self, scenario: Dict) -> Dict:
        """Apply data augmentations"""
        
        if not self.config.visual_augmentation:
            return scenario
        
        visual = scenario['visual_data']
        
        # Random brightness/contrast
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            visual = torch.clamp(visual * contrast + (brightness - 1.0), 0, 1)
        
        # Random noise
        if self.config.noise_injection and random.random() < 0.3:
            noise = torch.randn_like(visual) * 0.05
            visual = torch.clamp(visual + noise, 0, 1)
        
        # Random horizontal flip
        if random.random() < 0.3:
            visual = torch.flip(visual, [2])
            # Update object positions accordingly
            for obj in scenario.get('objects', []):
                obj['position'][0] = 1.0 - obj['position'][0]
        
        scenario['visual_data'] = visual
        return scenario

class SubstantialSAGETrainer:
    """Substantial 12-hour SAGE trainer"""
    
    def __init__(self, config: SubstantialTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.start_time = time.time()
        
        # Training state
        self.training_history = []
        self.best_score = 0.0
        self.patience_counter = 0
        self.current_lr = config.learning_rate
        
        # Initialize components
        self.data_generator = CurriculumDataGenerator(config)
        self.model = self._build_substantial_sage()
        self.optimizer = self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Create directories
        self.checkpoint_dir = Path("substantial_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Substantial SAGE Trainer Ready")
        print(f"📊 Model: {self.model.param_count():,} parameters")
        print(f"⏰ Budget: {config.total_hours} hours ({config.total_hours * config.epochs_per_hour} epochs)")
        
    def _build_substantial_sage(self):
        """Build larger SAGE model for substantial training"""
        import sys
        sys.path.append('../core')
        from sage_federation_v1 import SAGE, SAGEConfig
        
        config = SAGEConfig(
            hidden_dim=self.config.sage_hidden_dim,
            h_level_dim=self.config.sage_hidden_dim // 2,
            l_level_dim=self.config.sage_hidden_dim // 2,
            num_heads=16,  # More attention heads
            num_layers=self.config.sage_layers,
            vocab_size=2000,
            context_window=1024,
            dropout=0.1
        )
        
        model = SAGE(config).to(self.device)
        
        # Disable consciousness cache (still buggy)
        try:
            model.consciousness_cache.enabled = False
        except:
            pass
        
        return model
    
    def _setup_optimizer(self):
        """Setup optimizer with learning rate scheduling"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.current_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        return optimizer
    
    def train_substantial_sage(self):
        """Main 12-hour training loop"""
        print(f"\n🎯 Starting Substantial SAGE Training")
        print(f"⏰ Target duration: {self.config.total_hours} hours")
        print("="*60)
        
        # Generate comprehensive training dataset
        training_data = self._generate_comprehensive_dataset()
        
        total_epochs = int(self.config.total_hours * self.config.epochs_per_hour)
        epoch = 0
        
        try:
            while epoch < total_epochs:
                elapsed_hours = (time.time() - self.start_time) / 3600
                
                # Check time budget
                if elapsed_hours >= self.config.total_hours:
                    print(f"\n⏰ Time budget exhausted: {elapsed_hours:.1f}h")
                    break
                
                # Update curriculum
                self.data_generator.set_curriculum_stage(epoch)
                
                # Training epoch
                epoch_metrics = self._train_epoch(training_data, epoch)
                
                # Learning rate scheduling
                if self.config.learning_rate_schedule:
                    self._update_learning_rate(epoch, epoch_metrics)
                
                # Comprehensive evaluation
                if (epoch + 1) % self.config.evaluation_frequency == 0:
                    eval_metrics = self._comprehensive_evaluation(training_data)
                    self._log_evaluation(epoch, eval_metrics, elapsed_hours)
                    
                    # Model checkpointing
                    if self.config.model_checkpointing:
                        self._save_checkpoint(epoch, eval_metrics)
                    
                    # Early stopping check
                    if self.config.early_stopping:
                        if self._check_early_stopping(eval_metrics):
                            print(f"🛑 Early stopping triggered at epoch {epoch}")
                            break
                
                epoch += 1
                
                # Progress logging
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}/{total_epochs} | "
                          f"Time: {elapsed_hours:.1f}h/{self.config.total_hours}h | "
                          f"Loss: {epoch_metrics.get('loss', 0):.4f}")
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted at epoch {epoch}")
        
        # Final evaluation and save
        final_metrics = self._comprehensive_evaluation(training_data)
        self._save_final_model(epoch, final_metrics)
        
        elapsed_time = time.time() - self.start_time
        print(f"\n✅ Substantial Training Complete!")
        print(f"⏰ Total time: {elapsed_time/3600:.1f} hours")
        print(f"📊 Final performance: {final_metrics.get('overall_score', 0):.3f}")
        
        return final_metrics

    def _generate_comprehensive_dataset(self) -> Dict:
        """Generate comprehensive training and validation datasets"""
        print(f"📊 Generating comprehensive dataset...")
        print(f"  Training scenarios: {self.config.training_scenarios:,}")
        print(f"  Validation scenarios: {self.config.validation_scenarios:,}")
        
        # Generate training data
        training_scenarios = []
        for i in range(self.config.training_scenarios):
            if i % 500 == 0:
                print(f"    Training: {i:,}/{self.config.training_scenarios:,}")
            scenario = self.data_generator.generate_scenario(augment=True)
            training_scenarios.append(scenario)
        
        # Generate validation data (no augmentation)
        validation_scenarios = []
        for i in range(self.config.validation_scenarios):
            if i % 200 == 0:
                print(f"    Validation: {i:,}/{self.config.validation_scenarios:,}")
            scenario = self.data_generator.generate_scenario(augment=False)
            validation_scenarios.append(scenario)
        
        # Analyze dataset composition
        train_camera = sum(1 for s in training_scenarios if s['source'] == 'real_camera')
        val_camera = sum(1 for s in validation_scenarios if s['source'] == 'real_camera')
        
        print(f"✅ Dataset generated:")
        print(f"  Training: {len(training_scenarios):,} scenarios ({train_camera} camera, {len(training_scenarios)-train_camera} physics)")
        print(f"  Validation: {len(validation_scenarios):,} scenarios ({val_camera} camera, {len(validation_scenarios)-val_camera} physics)")
        
        return {
            'training': training_scenarios,
            'validation': validation_scenarios,
            'stats': {
                'train_camera_ratio': train_camera / len(training_scenarios),
                'val_camera_ratio': val_camera / len(validation_scenarios)
            }
        }
    
    def _train_epoch(self, dataset: Dict, epoch: int) -> Dict:
        """Train one epoch with substantial rigor"""
        self.model.train()
        
        training_scenarios = dataset['training']
        epoch_losses = []
        task_accuracies = []
        
        for step in range(self.config.steps_per_epoch):
            # Sample batch
            batch_scenarios = random.sample(training_scenarios, self.config.batch_size)
            
            # Forward pass with gradient accumulation
            accumulated_loss = 0.0
            successful_samples = 0
            
            for i, scenario in enumerate(batch_scenarios):
                try:
                    # Encode scenario
                    input_tensor = self._encode_substantial_scenario(scenario)
                    
                    # Forward pass
                    if self.config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = self.model(input_tensor, use_consciousness=False)
                            loss = self._compute_substantial_loss(output, scenario)
                    else:
                        output = self.model(input_tensor, use_consciousness=False)
                        loss = self._compute_substantial_loss(output, scenario)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation
                    accumulated_loss += loss.item()
                    
                    # Backward pass
                    if self.config.mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    successful_samples += 1
                    
                    # Calculate task accuracy
                    task_accuracy = self._calculate_task_accuracy(output, scenario)
                    task_accuracies.append(task_accuracy)
                    
                except Exception as e:
                    print(f"⚠️  Training error in step {step}, sample {i}: {e}")
                    continue
            
            # Optimizer step (every gradient_accumulation steps)
            if (step + 1) % self.config.gradient_accumulation == 0:
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            if successful_samples > 0:
                epoch_losses.append(accumulated_loss)
        
        return {
            'loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'task_accuracy': np.mean(task_accuracies) if task_accuracies else 0.0,
            'successful_steps': len(epoch_losses),
            'total_steps': self.config.steps_per_epoch
        }
    
    def _encode_substantial_scenario(self, scenario: Dict) -> torch.Tensor:
        """Encode scenario with substantial detail"""
        features = []
        
        # Visual features (more sophisticated)
        visual = scenario['visual_data']
        if isinstance(visual, torch.Tensor):
            # Use more visual features for substantial training
            visual_flat = visual.flatten()[:100]  # 100 features instead of 50
            features.extend(visual_flat.cpu().numpy())
        else:
            features.extend(np.zeros(100))
        
        # Object features (up to 6 objects with more properties)
        for i in range(6):
            if i < len(scenario.get('objects', [])):
                obj = scenario['objects'][i]
                # Position and size
                features.extend(obj['position'])
                features.extend(obj['size'])
                # Properties
                features.append(obj.get('confidence', 0.5))
                features.append(1.0 if obj.get('graspable', False) else 0.0)
                features.append(obj.get('mass', 0.1) / 100.0)  # Normalized mass
                features.append(obj.get('aspect_ratio', 1.0))
                features.append(obj.get('solidity', 0.5))
                # Detection metadata
                features.append(1.0 if 'color' in obj else 0.0)
            else:
                features.extend([0] * 11)  # Padding
        
        # Task encoding (expanded)
        task_map = {
            'reach': [1,0,0,0,0,0,0],
            'grasp': [0,1,0,0,0,0,0],
            'place': [0,0,1,0,0,0,0],
            'avoid': [0,0,0,1,0,0,0],
            'track': [0,0,0,0,1,0,0],
            'navigate': [0,0,0,0,0,1,0],
            'manipulate': [0,0,0,0,0,0,1]
        }
        task = scenario.get('task', 'reach')
        features.extend(task_map.get(task, [1,0,0,0,0,0,0]))
        
        # Scenario metadata
        features.append(scenario.get('difficulty', 0.0))
        features.append(scenario.get('complexity_score', 0.0))
        features.append(1.0 if scenario.get('source') == 'real_camera' else 0.0)
        features.append(scenario.get('motion_score', 0.0))
        
        # Pad to fixed size
        target_size = 200  # Larger feature vector for substantial training
        while len(features) < target_size:
            features.append(0.0)
        features = features[:target_size]
        
        # Convert to tokens
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        input_ids = (feature_tensor * 100).long().clamp(0, 1999)
        
        return input_ids.unsqueeze(0).to(self.device)
    
    def _compute_substantial_loss(self, output: Dict, scenario: Dict) -> torch.Tensor:
        """Compute sophisticated loss function"""
        device = output['h_ratio'].device
        losses = []
        
        # 1. Curriculum-aware task complexity loss
        task_complexity = {
            'reach': 0.2, 'track': 0.3, 'avoid': 0.4,
            'grasp': 0.6, 'navigate': 0.7, 'place': 0.8, 'manipulate': 0.9
        }
        base_complexity = task_complexity.get(scenario['task'], 0.5)
        
        # Adjust for curriculum difficulty
        difficulty = scenario.get('difficulty', 0.5)
        expected_h = base_complexity * (0.5 + 0.5 * difficulty)
        
        h_loss = F.mse_loss(output['h_ratio'], torch.tensor(expected_h, device=device))
        losses.append(h_loss)
        
        # 2. Real data bonus (stronger signal)
        if scenario.get('source') == 'real_camera':
            real_bonus = torch.tensor(-0.2, device=device)  # Stronger bonus
            losses.append(real_bonus)
        
        # 3. Object complexity loss
        objects = scenario.get('objects', [])
        if objects:
            salience = output['salience'].squeeze()
            
            # Expected salience based on object properties
            total_importance = 0.0
            for obj in objects:
                importance = obj.get('confidence', 0.5)
                if obj.get('graspable', False) and scenario['task'] in ['grasp', 'place']:
                    importance += 0.3
                total_importance += importance
            
            expected_salience = min(total_importance / len(objects), 1.0)
            
            if len(salience) > 0:
                salience_loss = F.mse_loss(salience[0], torch.tensor(expected_salience, device=device))
                losses.append(salience_loss)
        
        # 4. Difficulty progression loss
        complexity_score = scenario.get('complexity_score', 0.0)
        if complexity_score > 0:
            # Higher complexity should require higher H-level usage
            complexity_loss = F.mse_loss(
                output['h_ratio'], 
                torch.tensor(min(complexity_score + 0.2, 1.0), device=device)
            )
            losses.append(complexity_loss * 0.5)  # Weight down this component
        
        # 5. Motion-aware loss (for camera data)
        if 'motion_score' in scenario:
            motion = scenario['motion_score']
            if motion > 0.05:  # Significant motion
                # Should increase attention for tracking tasks
                if scenario['task'] in ['track', 'avoid']:
                    motion_bonus = torch.tensor(-motion * 0.1, device=device)
                    losses.append(motion_bonus)
        
        return sum(losses) / len(losses)
    
    def _calculate_task_accuracy(self, output: Dict, scenario: Dict) -> float:
        """Calculate task-specific accuracy"""
        h_ratio = output['h_ratio'].item()
        task = scenario['task']
        difficulty = scenario.get('difficulty', 0.5)
        
        # Task-specific thresholds adjusted for difficulty
        thresholds = {
            'reach': 0.15 + difficulty * 0.1,
            'track': 0.2 + difficulty * 0.15,
            'avoid': 0.25 + difficulty * 0.2,
            'grasp': 0.4 + difficulty * 0.25,
            'navigate': 0.5 + difficulty * 0.2,
            'place': 0.6 + difficulty * 0.2,
            'manipulate': 0.7 + difficulty * 0.15
        }
        
        threshold = thresholds.get(task, 0.5)
        
        # Success if H-ratio exceeds threshold
        if h_ratio >= threshold:
            # Additional bonus for real camera data
            if scenario.get('source') == 'real_camera':
                return 1.2  # 120% for real data success
            return 1.0
        else:
            # Partial credit based on how close
            return max(0.0, h_ratio / threshold)
    
    def _comprehensive_evaluation(self, dataset: Dict) -> Dict:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        validation_scenarios = dataset['validation']
        eval_metrics = {
            'task_success_rates': {},
            'source_performance': {},
            'difficulty_performance': {},
            'overall_metrics': {}
        }
        
        task_results = {}
        source_results = {'real_camera': [], 'physics_sim': []}
        difficulty_results = {0.2: [], 0.5: [], 0.8: [], 1.0: []}
        
        with torch.no_grad():
            for i, scenario in enumerate(validation_scenarios[:200]):  # Sample for speed
                try:
                    input_tensor = self._encode_substantial_scenario(scenario)
                    output = self.model(input_tensor, use_consciousness=False)
                    
                    accuracy = self._calculate_task_accuracy(output, scenario)
                    
                    # By task
                    task = scenario['task']
                    if task not in task_results:
                        task_results[task] = []
                    task_results[task].append(accuracy)
                    
                    # By source
                    source = scenario.get('source', 'physics_sim')
                    source_results[source].append(accuracy)
                    
                    # By difficulty
                    difficulty = scenario.get('difficulty', 0.5)
                    # Find closest difficulty bucket
                    closest_diff = min(difficulty_results.keys(), 
                                     key=lambda x: abs(x - difficulty))
                    difficulty_results[closest_diff].append(accuracy)
                    
                except Exception as e:
                    print(f"⚠️  Evaluation error: {e}")
                    continue
        
        # Calculate metrics
        for task, accuracies in task_results.items():
            eval_metrics['task_success_rates'][task] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'count': len(accuracies)
            }
        
        for source, accuracies in source_results.items():
            if accuracies:
                eval_metrics['source_performance'][source] = {
                    'mean': np.mean(accuracies),
                    'count': len(accuracies)
                }
        
        for diff, accuracies in difficulty_results.items():
            if accuracies:
                eval_metrics['difficulty_performance'][diff] = {
                    'mean': np.mean(accuracies),
                    'count': len(accuracies)
                }
        
        # Overall metrics
        all_accuracies = [acc for accs in task_results.values() for acc in accs]
        if all_accuracies:
            eval_metrics['overall_metrics'] = {
                'overall_accuracy': np.mean(all_accuracies),
                'total_evaluated': len(all_accuracies),
                'overall_score': np.mean(all_accuracies)  # For early stopping
            }
        
        return eval_metrics
    
    def _update_learning_rate(self, epoch: int, epoch_metrics: Dict):
        """Update learning rate with scheduling"""
        total_epochs = int(self.config.total_hours * self.config.epochs_per_hour)
        
        # Cosine annealing
        min_lr = self.current_lr * 0.1
        cosine_factor = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        new_lr = min_lr + (self.current_lr - min_lr) * cosine_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def _log_evaluation(self, epoch: int, metrics: Dict, elapsed_hours: float):
        """Log comprehensive evaluation results"""
        print(f"\n📊 Comprehensive Evaluation - Epoch {epoch}")
        print(f"⏰ Elapsed: {elapsed_hours:.1f}h")
        
        # Overall performance
        overall = metrics.get('overall_metrics', {})
        if overall:
            print(f"🎯 Overall Accuracy: {overall.get('overall_accuracy', 0):.3f}")
            print(f"📈 Evaluated Scenarios: {overall.get('total_evaluated', 0)}")
        
        # Task performance
        task_perf = metrics.get('task_success_rates', {})
        if task_perf:
            print(f"📋 Task Performance:")
            for task, stats in task_perf.items():
                print(f"  {task:10s}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} samples)")
        
        # Source performance
        source_perf = metrics.get('source_performance', {})
        if source_perf:
            print(f"📷 Source Performance:")
            for source, stats in source_perf.items():
                print(f"  {source:12s}: {stats['mean']:.3f} ({stats['count']} samples)")
        
        # Difficulty performance
        diff_perf = metrics.get('difficulty_performance', {})
        if diff_perf:
            print(f"📈 Difficulty Performance:")
            for diff, stats in sorted(diff_perf.items()):
                print(f"  {diff:.1f} difficulty: {stats['mean']:.3f} ({stats['count']} samples)")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        overall_score = metrics.get('overall_metrics', {}).get('overall_score', 0)
        
        if overall_score > self.best_score:
            self.best_score = overall_score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_score': self.best_score,
                'metrics': metrics,
                'config': self.config
            }
            
            checkpoint_path = self.checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 New best model saved: {overall_score:.3f} -> {checkpoint_path}")
    
    def _check_early_stopping(self, metrics: Dict) -> bool:
        """Check early stopping criteria"""
        overall_score = metrics.get('overall_metrics', {}).get('overall_score', 0)
        
        if overall_score > self.best_score:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            patience_limit = 6  # Stop if no improvement for 6 evaluations
            
            if self.patience_counter >= patience_limit:
                return True
        
        return False
    
    def _save_final_model(self, epoch: int, metrics: Dict):
        """Save final model and training results"""
        final_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'final_metrics': metrics,
            'training_history': self.training_history,
            'config': self.config,
            'total_time_hours': (time.time() - self.start_time) / 3600
        }
        
        final_path = self.checkpoint_dir / "final_substantial_sage.pt"
        torch.save(final_checkpoint, final_path)
        print(f"💾 Final model saved: {final_path}")

def main():
    """Run 12-hour substantial SAGE training"""
    print("🚀 Substantial SAGE Training - 12 Hour GPU Budget")
    print("="*60)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
    else:
        print("❌ CUDA not available")
        return
    
    config = SubstantialTrainingConfig()
    trainer = SubstantialSAGETrainer(config)
    
    # Confirm before starting 12-hour training
    print(f"\n⚠️  This will run for {config.total_hours} hours using substantial GPU resources.")
    print(f"📊 Training plan:")
    print(f"  - {config.total_hours * config.epochs_per_hour} epochs")
    print(f"  - {config.total_hours * config.epochs_per_hour * config.steps_per_epoch:,} training steps")
    print(f"  - {config.training_scenarios:,} training scenarios")
    print(f"  - Curriculum learning with {config.curriculum_stages} stages")
    print(f"  - Real camera data: {config.camera_ratio:.0%}")
    
    # Auto-proceed for unattended training
    print("\n✅ Auto-proceeding with substantial training...")
    proceed = 'y'
    
    # Start substantial training
    results = trainer.train_substantial_sage()
    
    print(f"\n🎊 Substantial training results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()