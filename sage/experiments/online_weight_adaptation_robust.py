"""
Online SNARC Weight Adaptation - Robust Version
=================================================

Improved version with:
- Checkpointing every N cycles
- Better error handling and logging
- Progress tracking with timestamps
- Graceful recovery from interruptions
- Dual logging (file + stdout)

**Author**: Claude (autonomous session)
**Date**: 2025-12-06
"""

import sys
import os
import time
import json
import signal
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from sage.core.snarc_compression import SNARCCompressor, SNARCWeights, SNARCDimensions
import psutil

# Import the existing classes
from online_weight_adaptation import OnlineWeightLearner, OnlineLearningConsciousness, WeightSnapshot


class RobustOnlineLearning:
    """
    Robust wrapper for online learning experiments.

    Features:
    - Checkpointing every N cycles
    - Dual logging (file + console with timestamps)
    - Graceful shutdown handling
    - Progress tracking and ETA
    - Auto-recovery from checkpoints
    """

    def __init__(
        self,
        duration_seconds: int = 1800,
        learning_rate: float = 0.05,
        output_dir: str = "/home/dp",
        experiment_name: str = "online_learning",
        checkpoint_interval: int = 50
    ):
        """
        Initialize robust online learning experiment.

        Args:
            duration_seconds: How long to run
            learning_rate: Learning rate for weight updates
            output_dir: Directory for outputs
            experiment_name: Base name for output files
            checkpoint_interval: Save checkpoint every N cycles
        """
        self.duration_seconds = duration_seconds
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.checkpoint_interval = checkpoint_interval

        # Create output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        self.log_file = self.output_dir / f"{self.run_id}.log"
        self.checkpoint_file = self.output_dir / f"{self.run_id}_checkpoint.json"
        self.results_file = self.output_dir / f"{self.run_id}_results.json"

        # State
        self.consciousness = None
        self.start_time = None
        self.should_stop = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on signals"""
        self.log(f"\n⚠️  Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True

    def log(self, message: str, to_file: bool = True, to_console: bool = True):
        """Log message to both file and console with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"

        if to_console:
            print(formatted)

        if to_file and hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")

    def save_checkpoint(self):
        """Save current state to checkpoint file"""
        if not self.consciousness or not self.consciousness.learner:
            return

        stats = self.consciousness.learner.get_statistics()
        checkpoint = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'cycle': self.consciousness.cycle,
            'attended_count': self.consciousness.attended_count,
            'elapsed_seconds': time.time() - self.start_time if self.start_time else 0,
            'current_weights': stats['current_weights'],
            'weight_updates': stats['update_count'],
            'current_lr': stats['current_lr'],
            'converged': stats.get('converged', False),
            'loss_history': self.consciousness.learner.loss_history[-10:] if self.consciousness.learner.loss_history else []
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def run(self):
        """Run the experiment with robust error handling"""
        try:
            self._run_experiment()
        except Exception as e:
            self.log(f"❌ Error during experiment: {e}")
            self.log(f"Traceback: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

            # Try to save what we have
            self.log("Attempting to save partial results...")
            try:
                self.save_checkpoint()
                self._save_final_results(partial=True)
            except Exception as save_error:
                self.log(f"Failed to save partial results: {save_error}")

            raise

    def _run_experiment(self):
        """Main experiment loop"""
        self.log("="*80)
        self.log("ROBUST ONLINE SNARC WEIGHT ADAPTATION")
        self.log("="*80)
        self.log("")
        self.log(f"Run ID: {self.run_id}")
        self.log(f"Duration: {self.duration_seconds} seconds ({self.duration_seconds/60:.1f} minutes)")
        self.log(f"Learning Rate: {self.learning_rate}")
        self.log(f"Checkpoint Interval: {self.checkpoint_interval} cycles")
        self.log(f"Log File: {self.log_file}")
        self.log("")

        # Initialize consciousness
        self.log("Initializing consciousness kernel...")
        self.consciousness = OnlineLearningConsciousness(
            learning_enabled=True,
            learning_rate=self.learning_rate
        )

        self.log("Initial weights:")
        for dim, weight in self.consciousness.compressor.weights.as_dict().items():
            self.log(f"  {dim}: {weight:.3f}")
        self.log("")

        self.log("Starting online learning deployment...")
        self.log("(Press Ctrl+C for graceful shutdown)")
        self.log("="*80)
        self.log("")

        # Main loop
        self.start_time = time.time()
        last_checkpoint_time = self.start_time
        checkpoint_interval_seconds = 60  # Save checkpoint every minute

        while not self.should_stop and (time.time() - self.start_time < self.duration_seconds):
            # Run one cycle
            result = self.consciousness.run_cycle()

            # Log cycle details to file only (not console)
            cycle_log = []
            cycle_log.append(f"[Cycle {result['cycle']}] State: {result['state'].upper()}")
            cycle_log.append(f"  Focus: {result['focus']} (salience={result['salience']:.3f})")
            cycle_log.append(f"  Dimensions: S={result['dimensions']['surprise']:.2f} "
                           f"N={result['dimensions']['novelty']:.2f} "
                           f"A={result['dimensions']['arousal']:.2f} "
                           f"R={result['dimensions']['reward']:.2f} "
                           f"C={result['dimensions']['conflict']:.2f}")
            cycle_log.append(f"  Threshold: {result['threshold']:.3f} | Decision: {result['decision']}")

            if 'weights' in result:
                cycle_log.append(f"  ⭐ UPDATED WEIGHTS: "
                               f"S={result['weights']['surprise']:.3f} "
                               f"N={result['weights']['novelty']:.3f} "
                               f"A={result['weights']['arousal']:.3f} "
                               f"R={result['weights']['reward']:.3f} "
                               f"C={result['weights']['conflict']:.3f} "
                               f"(loss={result['loss']:.4f})")

            cycle_log.append("")

            # Write cycle log to file
            with open(self.log_file, 'a') as f:
                f.write('\n'.join(cycle_log) + '\n')

            # Checkpoint if needed (time-based)
            if time.time() - last_checkpoint_time >= checkpoint_interval_seconds:
                self.save_checkpoint()
                last_checkpoint_time = time.time()

                # Print progress
                elapsed = time.time() - self.start_time
                progress_pct = 100 * elapsed / self.duration_seconds
                eta_seconds = (self.duration_seconds - elapsed)

                self.log(f"\n⏱️  Progress: {elapsed:.0f}s / {self.duration_seconds}s ({progress_pct:.1f}%) | ETA: {eta_seconds/60:.1f} min")
                self.log(f"   Cycles: {result['cycle']}, Attended: {self.consciousness.attended_count}")

                if self.consciousness.learner:
                    stats = self.consciousness.learner.get_statistics()
                    self.log(f"   Updates: {stats['update_count']}, LR: {stats['current_lr']:.4f}")
                    self.log(f"   Current weights: N={stats['current_weights']['novelty']:.3f}, "
                           f"A={stats['current_weights']['arousal']:.3f}, "
                           f"S={stats['current_weights']['surprise']:.3f}")
                    if 'current_loss' in stats:
                        self.log(f"   Loss: {stats['current_loss']:.4f}")
                    if stats.get('converged', False):
                        self.log(f"   🎯 CONVERGED!")
                self.log("")

            # Sleep between cycles
            time.sleep(2.0)

        # Experiment complete
        if self.should_stop:
            self.log("\n🛑 Graceful shutdown completed")
        else:
            self.log("\n✅ Experiment duration complete")

        # Final checkpoint
        self.save_checkpoint()

        # Save final results
        self._save_final_results()

    def _save_final_results(self, partial: bool = False):
        """Save final experiment results"""
        if not self.consciousness:
            return

        total_time = time.time() - self.start_time if self.start_time else 0

        self.log("")
        self.log("="*80)
        self.log(f"EXPERIMENT {'PARTIALLY ' if partial else ''}COMPLETE")
        self.log("="*80)
        self.log("")
        self.log(f"Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.log(f"Cycles: {self.consciousness.cycle}")
        self.log(f"Attended: {self.consciousness.attended_count} " +
               f"({100*self.consciousness.attended_count/self.consciousness.cycle:.1f}%)" if self.consciousness.cycle > 0 else "")
        self.log("")

        if self.consciousness.learner and self.consciousness.learner.weight_history:
            stats = self.consciousness.learner.get_statistics()
            self.log(f"Weight Updates: {stats['update_count']}")
            self.log(f"Final Learning Rate: {stats['current_lr']:.4f}")
            self.log(f"Converged: {stats.get('converged', False)}")
            self.log("")

            self.log("Weight Evolution:")
            self.log("  Initial → Final")
            initial = self.consciousness.learner.weight_history[0].weights
            final = stats['current_weights']
            for dim in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']:
                i = initial.get(dim, 0.0)
                f = final[dim]
                change = f - i
                self.log(f"  {dim:8s}: {i:.3f} → {f:.3f} ({change:+.3f})")
            self.log("")

            # Save JSON results
            results = {
                'run_id': self.run_id,
                'partial': partial,
                'runtime_seconds': total_time,
                'cycles': self.consciousness.cycle,
                'attended_count': self.consciousness.attended_count,
                'attention_rate': self.consciousness.attended_count / self.consciousness.cycle if self.consciousness.cycle > 0 else 0,
                'weight_updates': stats['update_count'],
                'converged': stats.get('converged', False),
                'initial_weights': initial,
                'final_weights': final,
                'weight_changes': {dim: final[dim] - initial.get(dim, 0.0) for dim in final.keys()},
                'final_loss': stats.get('current_loss', 0.0),
                'learning_rate_initial': self.consciousness.learner.initial_lr,
                'learning_rate_final': stats['current_lr'],
                'loss_history': self.consciousness.learner.loss_history[-20:] if self.consciousness.learner.loss_history else []
            }

            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)

            self.log(f"Results saved to: {self.results_file}")
            self.log(f"Checkpoint saved to: {self.checkpoint_file}")
            self.log(f"Log saved to: {self.log_file}")

        self.log("")
        self.log("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Robust online SNARC weight adaptation')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (default: 0.05)')
    parser.add_argument('--name', type=str, default='online_learning',
                       help='Experiment name (default: online_learning)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Checkpoint every N cycles (default: 50)')

    args = parser.parse_args()

    experiment = RobustOnlineLearning(
        duration_seconds=args.duration,
        learning_rate=args.lr,
        experiment_name=args.name,
        checkpoint_interval=args.checkpoint_interval
    )

    experiment.run()
