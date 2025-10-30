#!/usr/bin/env python3
"""
Monitor ongoing training progress and auto-start next training when complete
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime
import sys


def is_training_running(size: int) -> bool:
    """Check if training for a given size is running"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        return f"train_threshold_models.py {size}" in result.stdout
    except Exception as e:
        print(f"Error checking process: {e}")
        return False


def get_training_log_tail(size: int, lines: int = 20) -> str:
    """Get the last N lines from a training log"""
    log_file = Path(f"training_logs/{size}examples_training.log")
    if not log_file.exists():
        return f"Log file not found: {log_file}"

    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return ''.join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading log: {e}"


def start_training(size: int):
    """Start training for a given dataset size"""
    log_file = f"training_logs/{size}examples_training.log"
    print(f"\n{'='*80}")
    print(f"Starting training for {size}-example model")
    print(f"{'='*80}")
    print(f"Log file: {log_file}")
    print(f"Start time: {datetime.now()}")

    try:
        subprocess.Popen(
            [
                'nohup',
                'python3',
                'train_threshold_models.py',
                str(size)
            ],
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        print(f"✓ Training started for {size} examples")
        return True
    except Exception as e:
        print(f"❌ Failed to start training: {e}")
        return False


def monitor_and_autostart():
    """Monitor training progress and auto-start next when complete"""
    sizes = [40, 60, 80, 100]
    current_size_idx = 0

    print("="*80)
    print("Training Monitor & Auto-Starter")
    print("="*80)
    print("Will train models in sequence:")
    for size in sizes:
        print(f"  - {size} examples")
    print()

    # Check if first training is already running
    if not is_training_running(sizes[0]):
        print(f"Starting first training: {sizes[0]} examples")
        start_training(sizes[0])
    else:
        print(f"Training already running: {sizes[0]} examples")

    while current_size_idx < len(sizes):
        current_size = sizes[current_size_idx]

        # Wait for current training to complete
        while is_training_running(current_size):
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Training {current_size} examples... ", end='', flush=True)
            time.sleep(30)  # Check every 30 seconds

        print(f"\n✓ Training complete for {current_size} examples!")
        print("\nLast 20 lines of log:")
        print("-" * 80)
        print(get_training_log_tail(current_size))
        print("-" * 80)

        # Move to next size
        current_size_idx += 1

        if current_size_idx < len(sizes):
            next_size = sizes[current_size_idx]
            print(f"\nStarting next training: {next_size} examples")
            start_training(next_size)
            time.sleep(5)  # Give it time to start
        else:
            print("\n" + "="*80)
            print("All training complete!")
            print("="*80)
            print("\nTrained models:")
            for size in sizes:
                model_dir = Path(f"threshold_models/{size}examples_model/final_model")
                if model_dir.exists():
                    print(f"  ✓ {size} examples: {model_dir}")
                else:
                    print(f"  ✗ {size} examples: NOT FOUND")


def check_status():
    """Just check current status without auto-starting"""
    sizes = [40, 60, 80, 100]

    print("="*80)
    print("Training Status Check")
    print("="*80)
    print()

    for size in sizes:
        running = is_training_running(size)
        model_dir = Path(f"threshold_models/{size}examples_model/final_model")
        completed = model_dir.exists()

        status = "🟢 RUNNING" if running else ("✓ COMPLETE" if completed else "⏸ PENDING")
        print(f"{size:3d} examples: {status}")

        if running:
            print(f"\n  Last 10 lines of log:")
            print("  " + "-" * 76)
            log_lines = get_training_log_tail(size, 10)
            for line in log_lines.split('\n'):
                print(f"  {line}")
            print()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--status':
        check_status()
    else:
        try:
            monitor_and_autostart()
        except KeyboardInterrupt:
            print("\n\nMonitoring interrupted by user")
            print("Training processes will continue in background")


if __name__ == "__main__":
    main()
