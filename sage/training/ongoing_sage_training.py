#!/usr/bin/env python3
"""
Ongoing SAGE Training - Continuous Learning During Waking Hours
===============================================================
Evolution-driven continuous training with adaptive curriculum
"""

import torch
import time
import signal
import sys
from substantial_sage_training import SubstantialSAGETrainer, SubstantialTrainingConfig
from pathlib import Path
import json
from datetime import datetime, timedelta

class OngoingTrainingConfig(SubstantialTrainingConfig):
    """Ongoing training configuration"""
    
    # Continuous training parameters
    continuous_training: bool = True
    save_interval_hours: float = 2.0  # Save every 2 hours
    evolution_cycles: int = 1000  # Keep evolving
    adaptive_curriculum: bool = True
    
    # Performance-based evolution
    performance_threshold: float = 1.2  # When to increase difficulty
    stagnation_threshold: int = 5  # Cycles before curriculum adaptation
    
    # Waking hours (can be customized)
    waking_start_hour: int = 6  # 6 AM
    waking_end_hour: int = 22    # 10 PM
    
def is_waking_hours(config):
    """Check if currently in waking hours"""
    current_hour = datetime.now().hour
    return config.waking_start_hour <= current_hour < config.waking_end_hour

class OngoingSAGETrainer(SubstantialSAGETrainer):
    """Ongoing SAGE trainer with continuous evolution"""
    
    def __init__(self, config):
        super().__init__(config)
        self.cycle_count = 0
        self.performance_history = []
        self.stagnation_counter = 0
        self.setup_signal_handlers()
        
        print(f"🔄 Ongoing SAGE Training Initialized")
        print(f"⏰ Waking hours: {config.waking_start_hour}:00 - {config.waking_end_hour}:00")
        print(f"💾 Auto-save every {config.save_interval_hours} hours")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Graceful shutdown requested (signal {signum})")
            self.save_current_state()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def save_current_state(self):
        """Save current training state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = self.checkpoint_dir / f"ongoing_state_{timestamp}.pt"
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cycle_count': self.cycle_count,
            'performance_history': self.performance_history,
            'stagnation_counter': self.stagnation_counter,
            'curriculum_stage': getattr(self.data_generator, 'current_stage', 0),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(state, state_path)
        print(f"💾 State saved: {state_path}")
        return state_path
    
    def load_latest_state(self):
        """Load latest training state if available"""
        state_files = list(self.checkpoint_dir.glob("ongoing_state_*.pt"))
        if not state_files:
            return False
        
        latest_state = max(state_files, key=lambda x: x.stat().st_mtime)
        print(f"📂 Loading state: {latest_state}")
        
        try:
            state = torch.load(latest_state, map_location=self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.cycle_count = state['cycle_count']
            self.performance_history = state['performance_history']
            self.stagnation_counter = state['stagnation_counter']
            
            print(f"✅ Resumed from cycle {self.cycle_count}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load state: {e}")
            return False
    
    def evolve_curriculum(self, performance_metrics):
        """Evolve curriculum based on performance"""
        current_performance = performance_metrics.get('overall_score', 0)
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Check for stagnation
        if len(self.performance_history) >= 3:
            recent_avg = sum(self.performance_history[-3:]) / 3
            if recent_avg < self.config.performance_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Adapt curriculum if stagnating
        if self.stagnation_counter >= self.config.stagnation_threshold:
            print(f"📈 Curriculum adaptation triggered (stagnation: {self.stagnation_counter})")
            self._adapt_curriculum()
            self.stagnation_counter = 0
    
    def _adapt_curriculum(self):
        """Adapt curriculum difficulty"""
        # Increase difficulty slightly
        if hasattr(self.data_generator, 'difficulty_factor'):
            old_difficulty = self.data_generator.difficulty_factor
            new_difficulty = min(old_difficulty + 0.1, 1.0)
            self.data_generator.difficulty_factor = new_difficulty
            print(f"🎯 Difficulty adapted: {old_difficulty:.2f} → {new_difficulty:.2f}")
        
        # Add more real camera data if performance is good
        if len(self.performance_history) > 0 and sum(self.performance_history[-3:]) / 3 > 1.1:
            self.config.camera_ratio = min(self.config.camera_ratio + 0.05, 0.6)
            print(f"📷 Real camera ratio increased to {self.config.camera_ratio:.0%}")
    
    def ongoing_training_cycle(self, dataset):
        """Single training cycle with evolution"""
        print(f"\n🔄 Training Cycle {self.cycle_count + 1}")
        
        # Training phase (shorter cycles for continuous learning)
        cycle_epochs = 20  # Shorter cycles
        for epoch in range(cycle_epochs):
            if not is_waking_hours(self.config):
                print(f"😴 Outside waking hours, pausing...")
                return False
            
            epoch_metrics = self._train_epoch(dataset, epoch)
            
            if epoch % 10 == 0:
                print(f"  Cycle {self.cycle_count + 1}, Epoch {epoch}: Loss={epoch_metrics.get('loss', 0):.4f}")
        
        # Evaluation phase
        eval_metrics = self._comprehensive_evaluation(dataset)
        
        # Evolution phase
        self.evolve_curriculum(eval_metrics)
        
        # Log cycle results
        overall_score = eval_metrics.get('overall_metrics', {}).get('overall_score', 0)
        print(f"🎯 Cycle {self.cycle_count + 1} complete: Score={overall_score:.3f}")
        
        self.cycle_count += 1
        return True
    
    def run_ongoing_training(self):
        """Main ongoing training loop"""
        print(f"\n🚀 Starting Ongoing SAGE Training")
        print("="*60)
        
        # Try to resume from previous state
        resumed = self.load_latest_state()
        
        # Generate or load dataset
        if resumed:
            print("📊 Using existing dataset...")
            # For simplicity, regenerate dataset (could be optimized to save/load)
            dataset = self._generate_comprehensive_dataset()
        else:
            print("📊 Generating fresh dataset...")
            dataset = self._generate_comprehensive_dataset()
        
        last_save_time = time.time()
        
        try:
            while self.cycle_count < self.config.evolution_cycles:
                # Check if in waking hours
                if not is_waking_hours(self.config):
                    print(f"😴 Outside waking hours ({datetime.now().strftime('%H:%M')}), sleeping for 1 hour...")
                    time.sleep(3600)  # Sleep for 1 hour
                    continue
                
                # Run training cycle
                cycle_success = self.ongoing_training_cycle(dataset)
                
                if not cycle_success:
                    continue
                
                # Periodic saves
                if time.time() - last_save_time >= self.config.save_interval_hours * 3600:
                    self.save_current_state()
                    last_save_time = time.time()
                
                # Short rest between cycles
                time.sleep(60)  # 1 minute between cycles
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
        finally:
            # Final save
            final_path = self.save_current_state()
            print(f"🎊 Ongoing training completed: {self.cycle_count} cycles")
            print(f"💾 Final state: {final_path}")

def main():
    """Run ongoing SAGE training"""
    print("🔄 Ongoing SAGE Training - Continuous Evolution")
    print("="*60)
    
    config = OngoingTrainingConfig()
    
    # Check if in waking hours
    if not is_waking_hours(config):
        print(f"😴 Currently outside waking hours ({datetime.now().strftime('%H:%M')})")
        print(f"⏰ Training will start at {config.waking_start_hour}:00")
        print("💤 Sleeping until waking hours...")
        
        # Calculate sleep time
        now = datetime.now()
        if now.hour >= config.waking_end_hour:
            # Sleep until next day's waking time
            next_wake = now.replace(hour=config.waking_start_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            # Sleep until today's waking time
            next_wake = now.replace(hour=config.waking_start_hour, minute=0, second=0, microsecond=0)
        
        sleep_seconds = (next_wake - now).total_seconds()
        time.sleep(sleep_seconds)
    
    # Start training
    trainer = OngoingSAGETrainer(config)
    trainer.run_ongoing_training()

if __name__ == "__main__":
    main()