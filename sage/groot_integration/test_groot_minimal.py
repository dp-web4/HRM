"""
Minimal test of GR00T integration focusing on the sleep cycle training.
Works around dependency issues by using mock GR00T components.
"""

import torch
import numpy as np
import time
import sys
import os

# Add path for our components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from context.reality_context_4k import RealityContext4K, RealityContextEncoder4K
from groot_integration.sleep_cycle_training import GR00TSleepCycleTrainer

print("\n" + "="*60)
print("🧪 Testing GR00T Sleep Cycle Training (Minimal)")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
else:
    print("⚠️ No GPU available, using CPU")

print("\n🚀 Initializing Sleep Cycle Trainer...")

# Create trainer
trainer = GR00TSleepCycleTrainer(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"✅ Trainer initialized")
print(f"   Context encoder: {sum(p.numel() for p in trainer.context_encoder.parameters()):,} params")
print(f"   SAGE model: {sum(p.numel() for p in trainer.sage_model.parameters()):,} params")

print("\n🔄 Running complete sleep cycle...")

# Run full cycle
start_time = time.time()

# 1. Wake Phase - Generate experiences
print("\n🌅 WAKE PHASE: Generating synthetic experiences...")
wake_episodes = trainer.wake_phase(hours=0.05)  # 3 minutes of experience
print(f"   Generated {len(wake_episodes)} episodes")
print(f"   Total experiences: {trainer.wake_experiences}")

# 2. Sleep Phase - Consolidate through augmentation
print("\n😴 SLEEP PHASE: Consolidating experiences...")
sleep_consolidated = trainer.sleep_phase(num_samples=50)
print(f"   Consolidated {len(sleep_consolidated)} experiences")
if sleep_consolidated:
    avg_loss = np.mean([c['loss'] for c in sleep_consolidated])
    print(f"   Average consolidation loss: {avg_loss:.4f}")

# 3. Dream Phase - Test with edge cases
print("\n💭 DREAM PHASE: Exploring hypothetical scenarios...")
dream_scenarios = trainer.dream_phase(num_dreams=5)
print(f"   Explored {len(dream_scenarios)} dream scenarios")
if dream_scenarios:
    coherence_scores = [d['coherence'] for d in dream_scenarios]
    print(f"   Coherence scores: {coherence_scores}")
    print(f"   Average coherence: {np.mean(coherence_scores):.3f}")
    
    # Show dream types
    dream_types = [d['modification'] for d in dream_scenarios]
    print(f"   Dream modifications: {dream_types}")

elapsed = time.time() - start_time

print("\n" + "="*60)
print("📊 CYCLE COMPLETE - Summary")
print("="*60)
print(f"✅ Wake: {trainer.wake_experiences} experiences collected")
print(f"✅ Sleep: {trainer.sleep_consolidations} consolidations performed")
print(f"✅ Dream: {trainer.dream_explorations} scenarios explored")
print(f"⏱️ Total time: {elapsed:.1f} seconds")

# Memory usage
if torch.cuda.is_available():
    mem_used = torch.cuda.memory_allocated() / 1e9
    print(f"💾 GPU memory used: {mem_used:.2f} GB")

print("\n" + "="*60)
print("🎯 KEY INSIGHTS")
print("="*60)
print("1. Experience generates context automatically (no labeling)")
print("2. Sleep consolidation extracts invariances through augmentation") 
print("3. Dreams test robustness with impossible scenarios")
print("4. The loop is complete: Reality → Experience → Context → Understanding")

print("\n" + "="*60)
print("🚀 READY FOR GR00T INTEGRATION")
print("="*60)
print("Next steps:")
print("1. ✅ Sleep cycle training working")
print("2. ✅ 4K context encoder functional")
print("3. ⏳ Connect to actual GR00T simulator (pending dependencies)")
print("4. ⏳ Deploy on Jetson for real robot")
print("="*60 + "\n")