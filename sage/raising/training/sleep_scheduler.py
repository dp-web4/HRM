"""
Sleep Training Scheduler

Integrates Phase 3 sleep training with raising session rhythm.

Design Philosophy:
- Sleep training triggered after each session (not continuous)
- Checks for minimum experience threshold before training
- Respects minimum time between sleep cycles
- Logs all training decisions for analysis

Usage:
    scheduler = SleepScheduler()

    # After each raising session
    if scheduler.should_run_sleep_cycle():
        scheduler.run_sleep_cycle()
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, Optional
import logging

from sleep_training import SleepTrainingLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleepScheduler:
    """
    Manages automatic sleep training based on raising session rhythm.

    Decides when to run sleep cycles based on:
    1. Experience buffer size (minimum threshold)
    2. Time since last sleep cycle
    3. New high-salience experiences available
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        experience_buffer_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        min_experiences: int = 5,
        min_new_experiences: int = 2,
        min_hours_between_sleep: float = 6.0,
        min_salience: float = 0.6,
        device: Optional[str] = None
    ):
        """
        Initialize sleep scheduler.

        Args:
            model_path: Path to base model
            experience_buffer_path: Path to experience buffer
            checkpoint_dir: Path to checkpoint directory
            min_experiences: Minimum total experiences needed
            min_new_experiences: Minimum new experiences since last sleep
            min_hours_between_sleep: Minimum hours between sleep cycles
            min_salience: Minimum salience threshold for training
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Paths
        if model_path is None:
            model_path = str(Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism")
        if experience_buffer_path is None:
            experience_buffer_path = str(Path.home() / "ai-workspace/HRM/sage/raising/state/experience_buffer.json")
        if checkpoint_dir is None:
            checkpoint_dir = str(Path.home() / "ai-workspace/HRM/sage/checkpoints/sleep")

        self.model_path = Path(model_path)
        self.experience_buffer_path = Path(experience_buffer_path)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Thresholds
        self.min_experiences = min_experiences
        self.min_new_experiences = min_new_experiences
        self.min_hours_between_sleep = min_hours_between_sleep
        self.min_salience = min_salience
        self.device = device

        # State file
        self.state_file = self.checkpoint_dir / "scheduler_state.json"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load scheduler state from file."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        else:
            # Check if there are existing checkpoints to sync with
            return self._sync_with_existing_checkpoints()

    def _sync_with_existing_checkpoints(self) -> Dict:
        """
        Sync scheduler state with existing checkpoints.

        If checkpoints exist but scheduler_state.json doesn't, this
        recovers the state from the checkpoint training history.
        """
        # Find latest checkpoint
        checkpoints = sorted([d for d in self.checkpoint_dir.iterdir()
                            if d.is_dir() and d.name.startswith("cycle_")])

        if not checkpoints:
            # No checkpoints, initialize fresh
            return {
                'last_sleep_time': None,
                'last_sleep_cycle': 0,
                'experiences_at_last_sleep': 0,
                'total_sleep_cycles': 0,
                'sleep_history': []
            }

        # Load latest checkpoint state
        latest_checkpoint = checkpoints[-1]
        training_state_file = latest_checkpoint / "training_state.json"

        if training_state_file.exists():
            with open(training_state_file) as f:
                training_state = json.load(f)

            # Reconstruct scheduler state from training state
            logger.info(f"Syncing with existing checkpoint: {latest_checkpoint.name}")

            return {
                'last_sleep_time': training_state['last_updated'],
                'last_sleep_cycle': training_state['sleep_cycle_count'],
                'experiences_at_last_sleep': training_state['total_experiences_trained'],
                'total_sleep_cycles': training_state['sleep_cycle_count'],
                'sleep_history': training_state.get('training_history', [])
            }
        else:
            # Checkpoint exists but no training state
            return {
                'last_sleep_time': None,
                'last_sleep_cycle': 0,
                'experiences_at_last_sleep': 0,
                'total_sleep_cycles': 0,
                'sleep_history': []
            }

    def _save_state(self):
        """Save scheduler state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _load_experiences(self):
        """Load experience buffer."""
        if not self.experience_buffer_path.exists():
            return []
        with open(self.experience_buffer_path) as f:
            return json.load(f)

    def should_run_sleep_cycle(self) -> tuple[bool, str]:
        """
        Check if sleep cycle should run.

        Returns:
            Tuple of (should_run: bool, reason: str)
        """
        experiences = self._load_experiences()
        num_experiences = len(experiences)

        # Check 1: Minimum total experiences
        if num_experiences < self.min_experiences:
            return False, f"Insufficient experiences ({num_experiences} < {self.min_experiences})"

        # Check 2: New experiences since last sleep
        experiences_at_last_sleep = self.state['experiences_at_last_sleep']
        new_experiences = num_experiences - experiences_at_last_sleep

        if new_experiences < self.min_new_experiences:
            return False, f"Insufficient new experiences ({new_experiences} < {self.min_new_experiences})"

        # Check 3: Time since last sleep
        last_sleep_time = self.state['last_sleep_time']
        if last_sleep_time:
            last_sleep_dt = datetime.fromisoformat(last_sleep_time)
            time_since_sleep = datetime.now() - last_sleep_dt
            min_time = timedelta(hours=self.min_hours_between_sleep)

            if time_since_sleep < min_time:
                hours_since = time_since_sleep.total_seconds() / 3600
                return False, f"Too soon since last sleep ({hours_since:.1f}h < {self.min_hours_between_sleep}h)"

        # All checks passed
        return True, f"Ready: {num_experiences} experiences ({new_experiences} new), sufficient time elapsed"

    def run_sleep_cycle(
        self,
        epochs: int = 3,
        learning_rate: float = 1e-5,
        max_experiences: Optional[int] = None
    ) -> Dict:
        """
        Run a sleep training cycle.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            max_experiences: Maximum experiences to train on

        Returns:
            Training results dictionary
        """
        logger.info("="*60)
        logger.info("INITIATING SLEEP CYCLE")
        logger.info("="*60)

        # Check if should run
        should_run, reason = self.should_run_sleep_cycle()
        if not should_run:
            logger.warning(f"Sleep cycle skipped: {reason}")
            return {
                'status': 'skipped',
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }

        logger.info(f"Sleep cycle approved: {reason}")

        # Initialize trainer
        trainer = SleepTrainingLoop(
            model_path=str(self.model_path),
            experience_buffer_path=str(self.experience_buffer_path),
            checkpoint_dir=str(self.checkpoint_dir),
            device=self.device
        )

        # Run sleep cycle
        results = trainer.run_sleep_cycle(
            min_salience=self.min_salience,
            max_experiences=max_experiences,
            epochs=epochs,
            learning_rate=learning_rate,
            save_checkpoint=True
        )

        # Update state
        experiences = self._load_experiences()
        self.state['last_sleep_time'] = datetime.now().isoformat()
        self.state['last_sleep_cycle'] = results['sleep_cycle']
        self.state['experiences_at_last_sleep'] = len(experiences)
        self.state['total_sleep_cycles'] += 1
        self.state['sleep_history'].append({
            'cycle': results['sleep_cycle'],
            'timestamp': results['timestamp'],
            'num_experiences': results['num_experiences'],
            'final_loss': results['final_loss']
        })

        self._save_state()

        logger.info("="*60)
        logger.info(f"SLEEP CYCLE {results['sleep_cycle']} COMPLETE")
        logger.info("="*60)

        return results

    def get_status(self) -> Dict:
        """Get current scheduler status."""
        experiences = self._load_experiences()
        should_run, reason = self.should_run_sleep_cycle()

        status = {
            'total_experiences': len(experiences),
            'experiences_since_last_sleep': len(experiences) - self.state['experiences_at_last_sleep'],
            'total_sleep_cycles': self.state['total_sleep_cycles'],
            'last_sleep_time': self.state['last_sleep_time'],
            'should_run_sleep': should_run,
            'reason': reason,
            'sleep_history': self.state['sleep_history'][-5:]  # Last 5 cycles
        }

        if self.state['last_sleep_time']:
            last_sleep_dt = datetime.fromisoformat(self.state['last_sleep_time'])
            status['hours_since_last_sleep'] = (datetime.now() - last_sleep_dt).total_seconds() / 3600
        else:
            status['hours_since_last_sleep'] = None

        return status


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run sleep training scheduler")
    parser.add_argument("--check", action="store_true", help="Check status without running")
    parser.add_argument("--force", action="store_true", help="Force run even if checks fail")

    args = parser.parse_args()

    # Initialize scheduler
    scheduler = SleepScheduler()

    if args.check:
        # Just check status
        status = scheduler.get_status()
        print("\n" + "="*60)
        print("SLEEP SCHEDULER STATUS")
        print("="*60)
        print(json.dumps(status, indent=2))
        print("="*60 + "\n")
    else:
        # Run sleep cycle (if approved or forced)
        if args.force:
            logger.info("FORCING sleep cycle (bypassing checks)")
            # Temporarily disable checks
            scheduler.min_experiences = 0
            scheduler.min_new_experiences = 0
            scheduler.min_hours_between_sleep = 0.0

        results = scheduler.run_sleep_cycle()

        print("\n" + "="*60)
        print("SLEEP CYCLE RESULTS")
        print("="*60)
        print(json.dumps(results, indent=2))
        print("="*60 + "\n")
