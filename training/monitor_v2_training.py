#!/usr/bin/env python3
"""
Monitor V2 training progress
"""
import time
import json
from pathlib import Path
import subprocess

def get_latest_metrics():
    """Get latest training metrics from logs or wandb"""
    # Try to get from wandb offline run
    wandb_dir = Path('/home/dp/ai-workspace/HRM/training/wandb')
    latest_run = None
    
    # Find latest run directory
    for run_dir in wandb_dir.glob('offline-run-*'):
        if latest_run is None or run_dir.stat().st_mtime > latest_run.stat().st_mtime:
            latest_run = run_dir
    
    if latest_run:
        summary_file = latest_run / 'files' / 'wandb-summary.json'
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    data = json.load(f)
                    return data
            except:
                pass
    
    return None

def check_training_status():
    """Check if training is still running"""
    result = subprocess.run(['pgrep', '-f', 'finetune_reasoning_fixed_v2.py'], 
                          capture_output=True, text=True)
    return len(result.stdout.strip()) > 0

def main():
    print("Monitoring V2 Training Progress")
    print("=" * 60)
    
    while True:
        if not check_training_status():
            print("\n✅ Training completed or stopped")
            break
            
        metrics = get_latest_metrics()
        if metrics:
            step = metrics.get('step', 0)
            cycles_mean = metrics.get('cycles_mean', 0)
            cycles_min = metrics.get('cycles_min', 0)
            cycles_max = metrics.get('cycles_max', 0)
            loss = metrics.get('loss', 0)
            halt_prob = metrics.get('final_halt_prob', 0)
            
            print(f"\rStep {step:5d}/10000 | Loss: {loss:.4f} | "
                  f"Cycles: {cycles_mean:.1f} ({cycles_min}-{cycles_max}) | "
                  f"Halt P: {halt_prob:.3f}", end='', flush=True)
        else:
            print("\rWaiting for metrics...", end='', flush=True)
        
        time.sleep(5)
    
    # Final summary
    print("\n" + "=" * 60)
    metrics = get_latest_metrics()
    if metrics:
        print("Final Training Metrics:")
        print(f"  Steps completed: {metrics.get('step', 0)}")
        print(f"  Final loss: {metrics.get('loss', 0):.4f}")
        print(f"  Average cycles: {metrics.get('cycles_mean', 0):.1f}")
        print(f"  Cycle range: {metrics.get('cycles_min', 0)}-{metrics.get('cycles_max', 0)}")
        print(f"  Final halt prob: {metrics.get('final_halt_prob', 0):.3f}")

if __name__ == "__main__":
    main()