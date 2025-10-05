#!/usr/bin/env python3
"""
Test NeuTTS Air IRP Integration
Demonstrates text-to-speech within the IRP framework
"""

import asyncio
import time
from pathlib import Path
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../training/neutts-air'))

from sage.irp.orchestrator import HRMOrchestrator
from sage.irp.plugins.neutts_air_impl import NeuTTSAirIRP


def test_standalone_tts():
    """Test NeuTTS Air plugin standalone"""
    print("=" * 60)
    print("Testing NeuTTS Air IRP Plugin - Standalone")
    print("=" * 60)
    
    # Create TTS plugin
    tts_config = {
        'entity_id': 'neutts_test',
        'max_iterations': 3,
        'device': 'cpu',
        'ref_audio_path': 'samples/dave.wav'
    }
    
    tts = NeuTTSAirIRP(tts_config)
    
    # Test texts
    test_texts = [
        "Hello from the Integrated Robotics Platform.",
        "Federation blockchain is running smoothly at height one hundred thousand.",
        "SAGE training continues with excellent convergence.",
        "Trust emerges from consensus. Consciousness arises from coherence."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n📝 Test {i+1}: {text[:50]}...")
        
        # Initialize state
        state = tts.init_state(
            x0={'text': text},
            task_ctx={'voice_id': 'default', 'prosody': {'speed': 1.0}}
        )
        
        # Run refinement steps
        total_budget = 10.0
        budget_used = 0.0
        
        while state.step_idx < tts_config['max_iterations'] and budget_used < total_budget:
            old_energy = state.energy_val or 1.0
            state, step_budget = tts.step(state, total_budget - budget_used)
            budget_used += step_budget
            new_energy = tts.energy(state)
            
            print(f"   Step {state.step_idx}: Energy {old_energy:.3f} → {new_energy:.3f}")
            
            # Check convergence
            if abs(old_energy - new_energy) < 0.01:
                print("   ✅ Converged!")
                break
        
        # Extract and save result
        result = tts.extract(state)
        output_path = f"/tmp/neutts_irp_test_{i+1}.wav"
        tts.save_audio(state, output_path)
        
        print(f"   Generated: {result['iterations']} iterations, confidence: {result['confidence']:.2f}")


def test_orchestrated_tts():
    """Test NeuTTS Air within the HRM orchestrator"""
    print("\n" + "=" * 60)
    print("Testing NeuTTS Air IRP Plugin - Orchestrated")
    print("=" * 60)
    
    # Create orchestrator with TTS enabled
    orchestrator_config = {
        'total_ATP': 100.0,
        'max_workers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'enable_vision': False,  # Disable other plugins for this test
        'enable_language': False,
        'enable_control': False,
        'enable_memory': False,
        'enable_tts': True,
        'tts_config': {
            'max_iterations': 5,
            'quality_threshold': 0.85
        }
    }
    
    try:
        import torch
        orchestrator = HRMOrchestrator(orchestrator_config)
        
        # Check if TTS plugin loaded
        if 'tts' not in orchestrator.plugins:
            print("⚠️ TTS plugin not loaded in orchestrator")
            return
        
        print(f"✅ Orchestrator initialized with plugins: {list(orchestrator.plugins.keys())}")
        
        # Create a multimodal task
        task = {
            'text': "The federation is strong. Society nodes are synchronized. Trust weights are balanced.",
            'modality': 'tts',
            'priority': 1.0
        }
        
        # Run through orchestrator (simplified)
        tts_plugin = orchestrator.plugins['tts']
        state = tts_plugin.init_state(
            x0=task['text'],
            task_ctx={'voice_id': 'orchestrated'}
        )
        
        # Allocate budget
        budgets = orchestrator.allocate_budgets(orchestrator.total_ATP)
        tts_budget = budgets.get('tts', 10.0)
        print(f"💰 TTS allocated budget: {tts_budget:.1f} ATP")
        
        # Run refinement
        budget_used = 0.0
        while state.step_idx < 5 and budget_used < tts_budget:
            state, step_budget = tts_plugin.step(state, tts_budget - budget_used)
            budget_used += step_budget
            energy = tts_plugin.energy(state)
            print(f"   Step {state.step_idx}: Energy = {energy:.3f}, Budget used = {budget_used:.2f}")
        
        # Save result
        tts_plugin.save_audio(state, "/tmp/neutts_orchestrated.wav")
        print("✅ Orchestrated TTS complete!")
        
    except ImportError as e:
        print(f"⚠️ Torch not available for orchestrator: {e}")
        print("   Orchestrator requires PyTorch for full functionality")


async def test_async_tts():
    """Test asynchronous TTS generation"""
    print("\n" + "=" * 60)
    print("Testing NeuTTS Air IRP Plugin - Async")
    print("=" * 60)
    
    tts_config = {
        'entity_id': 'neutts_async',
        'max_iterations': 2
    }
    
    tts = NeuTTSAirIRP(tts_config)
    
    async def generate_speech(text, idx):
        """Async wrapper for TTS generation"""
        print(f"🎤 Starting async TTS {idx}: {text[:30]}...")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def sync_generate():
            state = tts.init_state(x0=text, task_ctx={})
            for _ in range(tts_config['max_iterations']):
                state, _ = tts.step(state, 10.0)
            return tts.extract(state)
        
        result = await loop.run_in_executor(None, sync_generate)
        print(f"✅ Async TTS {idx} complete: {result['confidence']:.2f} confidence")
        return result
    
    # Generate multiple speeches concurrently
    texts = [
        "First async message from IRP.",
        "Second parallel speech synthesis.",
        "Third concurrent audio generation."
    ]
    
    tasks = [generate_speech(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    
    print(f"\n🎉 Generated {len(results)} speeches asynchronously!")


def main():
    """Run all tests"""
    print("\n🚀 NeuTTS Air IRP Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Standalone
    try:
        test_standalone_tts()
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
    
    # Test 2: Orchestrated
    try:
        test_orchestrated_tts()
    except Exception as e:
        print(f"❌ Orchestrated test failed: {e}")
    
    # Test 3: Async
    try:
        asyncio.run(test_async_tts())
    except Exception as e:
        print(f"❌ Async test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test suite complete!")
    print("\nGenerated audio files:")
    print("  /tmp/neutts_irp_test_*.wav - Standalone tests")
    print("  /tmp/neutts_orchestrated.wav - Orchestrated test")
    print("\nNote: Audio plays to system speakers if configured correctly")


if __name__ == "__main__":
    main()