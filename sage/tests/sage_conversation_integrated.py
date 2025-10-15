#!/usr/bin/env python3
"""
SAGE Conversation - Full Integration Test

Integrates real-time conversation into SAGE's unified consciousness loop:
1. StreamingAudioSensor → Audio transcriptions
2. SAGEUnified → Attention orchestration + ATP allocation
3. ConversationIRP → Pattern-based response generation
4. TTSEffector → Speech synthesis

This demonstrates conversation as a first-class modality within SAGE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch

# Import SAGE components
from core.sage_unified import SAGEUnified
from interfaces.streaming_audio_sensor import StreamingAudioSensor
from interfaces.tts_effector import TTSEffector
from irp.plugins.conversation_irp import ConversationIRP

print("="*80)
print("SAGE CONVERSATION - INTEGRATED TEST")
print("="*80)

# ============================================================================
# Initialize SAGE Unified
# ============================================================================

print("\n1. Initializing SAGE Unified...")

sage = SAGEUnified(
    config={
        'initial_atp': 100.0,
        'max_atp': 100.0,
        'enable_circadian': False,  # Disable for testing
        'simulation_mode': False
    },
    device=torch.device('cpu')
)

# ============================================================================
# Register Audio Sensor
# ============================================================================

print("\n2. Registering audio sensor...")

audio_sensor = StreamingAudioSensor({
    'sensor_id': 'conversation_audio',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'vad_aggressiveness': 2,
    'min_speech_duration': 0.5,
    'max_speech_duration': 10.0,
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

sage.register_sensor(audio_sensor)

# ============================================================================
# Register Conversation IRP Plugin
# ============================================================================

print("\n3. Registering Conversation IRP plugin...")

conversation_plugin = ConversationIRP({
    'entity_id': 'conversation_irp',
    'max_iterations': 12,
    'min_confidence': 0.6,
    'trust_weight': 1.0
})

sage.register_irp_plugin('conversation_audio', conversation_plugin)

# ============================================================================
# Initialize TTS Effector
# ============================================================================

print("\n4. Initializing TTS effector...")

tts_effector = TTSEffector({
    'piper_path': '/home/sprout/ai-workspace/piper/piper/piper',
    'model_path': '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx',
    'bt_sink': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'enabled': True
})

# ============================================================================
# Enhanced SAGE Cycle with Conversation
# ============================================================================

def sage_cycle_with_conversation():
    """
    Execute SAGE cycle with conversation integration

    This is the key integration point:
    1. SAGE polls sensors (including audio)
    2. SNARC scores salience (speech gets priority)
    3. ATP allocated (conversation competes with other modalities)
    4. IRP refinement (pattern matching)
    5. TTS effector (synthesize response)
    """

    # Run standard SAGE cycle
    result = sage.cycle()

    # Check if conversation IRP ran
    irp_results = result.get('results', {})

    if 'conversation_audio' in irp_results:
        conv_result = irp_results['conversation_audio']

        # Get response from IRP state
        if 'latent' in conv_result:
            # Extract response from IRP state (stored in metadata during refinement)
            # For now, we need to access the plugin directly to get the response
            # In production, this would be in the IRP result structure
            pass

    # Alternative: Check audio sensor readings directly
    readings = sage.sensor_hub.get_last_readings()

    if 'conversation_audio' in readings:
        reading = readings['conversation_audio']

        if reading and hasattr(reading, 'metadata'):
            text = reading.metadata.get('text')

            if text:
                print(f"\n👤 USER [{reading.confidence:.2f}]: {text}")

                # Run conversation IRP manually for now
                # (In full integration, this would be automatic via SAGE cycle)
                try:
                    final_state, history = conversation_plugin.refine(text)
                    response = conversation_plugin.get_response(final_state)

                    if response:
                        print(f"🧠 SAGE [pattern, {len(history)} iterations]: {response}")

                        # Synthesize response
                        tts_effector.execute(response)
                    else:
                        print(f"💭 SAGE: No pattern match (needs deeper processing)")

                except Exception as e:
                    print(f"⚠️  Conversation error: {e}")

    return result

# ============================================================================
# Run SAGE Loop with Conversation
# ============================================================================

print("\n" + "="*80)
print("✅ SAGE CONVERSATION SYSTEM READY")
print("="*80)
print("Speak naturally - SAGE will orchestrate attention and respond")
print("Press Ctrl+C to stop\n")

try:
    cycle_count = 0
    conversation_count = 0

    while True:
        try:
            result = sage_cycle_with_conversation()
            cycle_count += 1

            # Print status every 10 cycles
            if cycle_count % 10 == 0:
                print(f"\n[Cycle {cycle_count}] "
                      f"State: {result['state']} | "
                      f"ATP: {result['atp']:.1f} | "
                      f"Conversations: {conversation_count}")

            # Small sleep to prevent CPU spinning
            time.sleep(0.05)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"⚠️  Cycle error: {e}")
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n" + "="*80)
    print("SAGE CONVERSATION ENDED")
    print("="*80)

    # Print statistics
    sage_stats = sage.stats
    conv_stats = conversation_plugin.get_stats()
    tts_stats = tts_effector.get_stats()

    print(f"\n📊 SAGE Statistics:")
    print(f"   Total cycles: {sage_stats['total_cycles']}")
    print(f"   Total time: {sage_stats['total_time']:.2f}s")
    print(f"   Avg cycle: {sage_stats['avg_cycle_time']*1000:.2f}ms")

    print(f"\n🧠 Conversation IRP Statistics:")
    print(f"   Pattern queries: {conv_stats['total_queries']}")
    print(f"   Pattern matches: {conv_stats['matched_queries']}")
    print(f"   Match rate: {conv_stats['match_rate']:.1%}")

    print(f"\n🔊 TTS Statistics:")
    print(f"   Syntheses: {tts_stats['synthesis_count']}")
    print(f"   Avg time: {tts_stats['avg_time_ms']:.0f}ms")
    print(f"   Errors: {tts_stats['errors']}")

    print(f"\n✅ All systems stopped cleanly")
