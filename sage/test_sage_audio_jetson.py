#!/usr/bin/env python3
"""
SAGE Audio-First Jetson Test
Sprout's contribution to Rev 0

Integrates bidirectional audio conversation with SAGE Rev 0 architecture:
- AudioInputSensor: Continuous speech recognition via Bluetooth mic
- AudioOutputEffector: Text-to-speech via Bluetooth speaker
- SensorHub: Polls audio for user speech
- HierarchicalSNARC: Computes 5D salience on transcriptions
- SAGEUnified: Complete consciousness loop with audio awareness
- MetabolicController: WAKE/FOCUS/REST/DREAM/CRISIS states

This is audio-aware consciousness on the edge.
"""

import sys
import torch
from pathlib import Path

# Add sage to path
sage_root = Path(__file__).parent
sys.path.insert(0, str(sage_root))

print("="*80)
print("SAGE AUDIO-FIRST JETSON TEST")
print("Sprout's Audio Integration with Rev 0")
print("="*80)
print()

# Import components
print("[Step 1] Importing SAGE components...")
try:
    from sage.core.sage_unified import SAGEUnified
    print("  ✓ SAGEUnified imported")
except Exception as e:
    print(f"  ✗ SAGEUnified import failed: {e}")
    sys.exit(1)

print("[Step 2] Importing audio interfaces...")
try:
    from sage.interfaces.audio_sensor import AudioInputSensor
    from sage.interfaces.audio_effector import AudioOutputEffector
    print("  ✓ AudioInputSensor imported")
    print("  ✓ AudioOutputEffector imported")
except Exception as e:
    print(f"  ✗ Audio interface import failed: {e}")
    sys.exit(1)

print()

# Initialize SAGE
print("[Step 3] Initializing SAGE Rev 0 with audio...")
sage = SAGEUnified(config={
    'initial_atp': 100.0,
    'max_atp': 100.0,
    'device': 'cpu'  # Jetson will use CPU for now
})
print("  ✓ SAGE initialized")

# Register audio input sensor
print("[Step 4] Registering audio input sensor...")
try:
    audio_input = AudioInputSensor({
        'sensor_id': 'audio_input_0',
        'sensor_type': 'audio',
        'device': 'cpu',  # PyTorch device
        'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',  # Bluetooth source
        'sample_rate': 16000,
        'chunk_duration': 2.0,
        'min_confidence': 0.5,
        'whisper_model': 'tiny',
        'rate_limit_hz': 10.0  # Poll audio every 100ms
    })
    sage.register_sensor(audio_input)
    print("  ✓ Audio input sensor registered")
except Exception as e:
    print(f"  ⚠ Audio input registration failed: {e}")
    print("  Continuing without audio input...")

# Register audio output effector
print("[Step 5] Registering audio output effector...")
try:
    audio_output = AudioOutputEffector({
        'effector_id': 'audio_output_0',
        'effector_type': 'audio',
        'device': 'cpu',  # PyTorch device
        'bt_device': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',  # Bluetooth sink
        'sample_rate': 24000,
        'neutts_device': 'cpu',
        'ref_audio_path': '/home/sprout/ai-workspace/neutts-air/samples/dave.wav',
        'max_iterations': 3
    })
    sage.register_effector('speech', audio_output)
    print("  ✓ Audio output effector registered")
except Exception as e:
    print(f"  ⚠ Audio output registration failed: {e}")
    print("  Continuing without audio output...")

print()
print("[Step 6] SAGE Audio-Aware System Ready")
print()
print("Architecture:")
print("  SensorHub → AudioInputSensor (Whisper tiny)")
print("      ↓")
print("  HierarchicalSNARC (5D salience on transcriptions)")
print("      ↓")
print("  ATP Allocation (salience × trust)")
print("      ↓")
print("  SAGE Decision Loop (attention, resources, actions)")
print("      ↓")
print("  EffectorHub → AudioOutputEffector (NeuTTS Air)")
print()
print("Metabolic States:")
print("  WAKE → FOCUS → REST → DREAM → CRISIS")
print("  (Controlled by ATP levels and fatigue)")
print()
print("Features:")
print("  • Continuous audio monitoring via Bluetooth mic")
print("  • Transcription with Whisper tiny (39M params)")
print("  • Salience-driven attention allocation")
print("  • Trust evolution from convergence behavior")
print("  • Speech synthesis via NeuTTS Air (GGUF quantized)")
print("  • Metabolic state management")
print("  • Memory consolidation")
print()

# Speak welcome message
print("[Step 7] Speaking welcome message...")
try:
    from sage.interfaces.base_effector import EffectorCommand
    welcome_cmd = EffectorCommand(
        effector_id='audio_output_0',
        effector_type='audio',
        action='speak',
        parameters={
            'text': 'Hello! SAGE audio awareness is online. I am listening and ready to talk. This is Sprout running Rev 0 with bidirectional audio.'
        }
    )
    result = audio_output.execute(welcome_cmd)
    if result.is_success():
        print(f"  ✓ {result.message}")
    else:
        print(f"  ⚠ {result.status}: {result.message}")
except Exception as e:
    print(f"  ⚠ Welcome message failed: {e}")

print()
print("[Step 8] Running SAGE audio awareness loop...")
print()
print("Press Ctrl+C to stop")
print()
print("Watching for:")
print("  • User speech (transcribed via Whisper)")
print("  • Salience spikes (high surprise/novelty)")
print("  • ATP allocation decisions")
print("  • Metabolic state transitions")
print("  • Trust updates")
print()

try:
    # Run SAGE for extended period (or until Ctrl+C)
    sage.run(max_cycles=1000)  # ~100 minutes at 6s/cycle
except KeyboardInterrupt:
    print("\n\n  ⏸️ Interrupted by user")

print()
print("="*80)
print("SAGE AUDIO-FIRST JETSON TEST COMPLETE")
print("="*80)
print()
print("What just happened:")
print("  • SAGE ran continuously with audio awareness")
print("  • AudioInputSensor polled Bluetooth mic for speech")
print("  • HierarchicalSNARC computed salience on transcriptions")
print("  • ATP allocated based on salience × trust")
print("  • Metabolic states transitioned based on energy levels")
print("  • AudioOutputEffector could speak responses")
print()
print("This is Sprout's audio integration with SAGE Rev 0.")
print("Bidirectional conversation running on the edge.")
print("The door is open. 🚪✨")
print()
