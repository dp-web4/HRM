"""
Audio Input Sensor for SAGE
Wraps AudioInputIRP as a BaseSensor for integration with SensorHub

Uses Whisper tiny for continuous speech recognition via Bluetooth microphone.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.util

# Add paths for imports
sage_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_root))
sys.path.insert(0, str(sage_root / 'irp'))

from interfaces.base_sensor import BaseSensor, SensorReading

# Import AudioInputIRP directly to avoid __init__.py conflicts
spec_audio = importlib.util.spec_from_file_location(
    "audio_input_impl",
    sage_root / "irp" / "plugins" / "audio_input_impl.py"
)
audio_module = importlib.util.module_from_spec(spec_audio)
spec_audio.loader.exec_module(audio_module)
AudioInputIRP = audio_module.AudioInputIRP


class AudioInputSensor(BaseSensor):
    """
    Audio input sensor using Whisper for continuous speech recognition.

    Wraps AudioInputIRP to conform to SAGE sensor interface.

    Configuration:
        sensor_id: Unique identifier (default: 'audio_input')
        sensor_type: Always 'audio'
        device: Bluetooth source device
        sample_rate: Audio sample rate (default: 16000)
        chunk_duration: Recording chunk size in seconds (default: 2.0)
        min_confidence: Minimum transcription confidence (default: 0.5)
        whisper_model: Whisper model size (default: 'tiny')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize audio input sensor with Whisper."""
        # Set device to 'cpu' before calling super().__init__
        if 'device' in config and 'bluez' in str(config['device']):
            # Save Bluetooth device separately
            self.bt_device = config['device']
            config = config.copy()
            config['device'] = 'cpu'  # PyTorch device
        else:
            self.bt_device = config.get('bt_device',
                                         'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit')
            if 'device' not in config:
                config = config.copy()
                config['device'] = 'cpu'

        super().__init__(config)
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 2.0)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.whisper_model = config.get('whisper_model', 'tiny')

        # Initialize AudioInputIRP plugin
        irp_config = {
            'entity_id': self.sensor_id,
            'device': self.bt_device,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'min_confidence': self.min_confidence,
            'whisper_model': self.whisper_model,
            'max_duration': 10.0
        }

        self.audio_irp = AudioInputIRP(irp_config)

        print(f"✅ AudioInputSensor initialized")
        print(f"   Device: {self.bt_device}")
        print(f"   Whisper: {self.whisper_model}")
        print(f"   Sample rate: {self.sample_rate} Hz")

    def poll(self) -> Optional[SensorReading]:
        """
        Poll for audio input (blocking until speech detected).

        Returns:
            SensorReading with transcription data or None if no speech/low confidence
        """
        if not self._should_poll():
            return None

        try:
            # Initialize listening state
            state = self.audio_irp.init_state(
                x0=None,
                task_ctx={'prompt': 'Listen for user speech'}
            )

            history = [state]

            # Run refinement loop until confident transcription
            while not self.audio_irp.halt(history):
                state = self.audio_irp.step(state)
                state.energy_val = self.audio_irp.energy(state)
                history.append(state)

            # Extract final transcription
            result = self.audio_irp.extract(state)

            # Filter out silence or low confidence
            if not result['text'] or len(result['text']) < 5:
                return None
            if result['confidence'] < self.min_confidence:
                return None

            # Convert transcription to tensor (embedding or token IDs could go here)
            # For now, return confidence as scalar and store text in metadata
            data_tensor = torch.tensor([result['confidence']], dtype=torch.float32)

            return SensorReading(
                sensor_id=self.sensor_id,
                sensor_type='audio',
                data=data_tensor,
                confidence=result['confidence'],
                metadata={
                    'text': result['text'],
                    'chunks': result['chunks'],
                    'duration': result['duration'],
                    'halt_reason': self.audio_irp.get_halt_reason(history)
                },
                device=str(self.device)
            )

        except Exception as e:
            print(f"⚠️ AudioInputSensor poll error: {e}")
            return None

    async def poll_async(self) -> Optional[SensorReading]:
        """Async version of poll (currently just wraps sync version)."""
        import asyncio
        return await asyncio.to_thread(self.poll)

    def is_available(self) -> bool:
        """Check if audio input hardware is available."""
        import subprocess
        try:
            # Check if Bluetooth device is available
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return self.bt_device in result.stdout
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Return sensor capabilities and configuration."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': 'audio',
            'bluetooth_device': self.bt_device,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'min_confidence': self.min_confidence,
            'whisper_model': self.whisper_model,
            'available': self.is_available(),
            'enabled': self.enabled
        }
