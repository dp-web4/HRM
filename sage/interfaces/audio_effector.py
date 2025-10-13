"""
Audio Output Effector for SAGE
Wraps NeuTTSAirIRP as a BaseEffector for integration with EffectorHub

Uses NeuTTS Air for text-to-speech via Bluetooth speaker.
"""

import torch
import numpy as np
import subprocess
import tempfile
import os
import soundfile as sf
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.util

# Add paths for imports
sage_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_root))
sys.path.insert(0, str(sage_root / 'irp'))
sys.path.insert(0, str(Path.home() / 'ai-workspace' / 'neutts-air'))

from interfaces.base_effector import BaseEffector, EffectorCommand, EffectorResult, EffectorStatus

# Import NeuTTSAirIRP directly to avoid conflicts
spec_tts = importlib.util.spec_from_file_location(
    "neutts_air_impl",
    sage_root / "irp" / "plugins" / "neutts_air_impl.py"
)
tts_module = importlib.util.module_from_spec(spec_tts)
spec_tts.loader.exec_module(tts_module)
NeuTTSAirIRP = tts_module.NeuTTSAirIRP


class AudioOutputEffector(BaseEffector):
    """
    Audio output effector using NeuTTS Air for text-to-speech.

    Wraps NeuTTSAirIRP to conform to SAGE effector interface.

    Configuration:
        effector_id: Unique identifier (default: 'audio_output')
        effector_type: Always 'audio'
        device: Bluetooth sink device for playback
        sample_rate: Audio sample rate (default: 24000)
        neutts_device: Device for TTS inference (default: 'cpu')
        ref_audio_path: Reference voice for cloning
        max_iterations: Max refinement iterations (default: 3)

    Commands:
        action='speak':
            parameters={'text': 'Text to speak'}
        action='play':
            data=torch.Tensor (audio waveform)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize audio output effector with NeuTTS."""
        # Set device to 'cpu' before calling super().__init__
        if 'device' in config and 'bluez' in str(config['device']):
            # Save Bluetooth device separately
            self.bt_device = config['device']
            config = config.copy()
            config['device'] = 'cpu'  # PyTorch device
        else:
            self.bt_device = config.get('bt_device',
                                         'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit')
            if 'device' not in config:
                config = config.copy()
                config['device'] = 'cpu'

        super().__init__(config)
        self.sample_rate = config.get('sample_rate', 24000)
        self.neutts_device = config.get('neutts_device', 'cpu')
        self.ref_audio_path = config.get('ref_audio_path',
                                          '/home/sprout/ai-workspace/neutts-air/samples/dave.wav')
        self.max_iterations = config.get('max_iterations', 3)

        # Initialize NeuTTSAirIRP plugin
        irp_config = {
            'entity_id': self.effector_id,
            'backbone_repo': 'neuphonic/neutts-air-q4-gguf',
            'codec_repo': 'neuphonic/neucodec',
            'device': self.neutts_device,
            'sample_rate': self.sample_rate,
            'ref_audio_path': self.ref_audio_path,
            'max_iterations': self.max_iterations
        }

        self.tts_irp = NeuTTSAirIRP(irp_config)

        print(f"✅ AudioOutputEffector initialized")
        print(f"   Device: {self.bt_device}")
        print(f"   TTS device: {self.neutts_device}")
        print(f"   Sample rate: {self.sample_rate} Hz")

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """
        Execute audio output command.

        Supported actions:
            'speak': Synthesize text and play via Bluetooth
            'play': Play provided audio waveform

        Args:
            command: EffectorCommand with action and parameters

        Returns:
            EffectorResult with execution status
        """
        import time
        start_time = time.time()

        try:
            if command.action == 'speak':
                # Text-to-speech synthesis
                text = command.parameters.get('text', '')
                if not text:
                    return EffectorResult(
                        effector_id=self.effector_id,
                        status=EffectorStatus.INVALID_COMMAND,
                        message="No text provided for speech synthesis",
                        execution_time=time.time() - start_time
                    )

                # Initialize TTS state
                tts_input = {
                    'text': text,
                    'ref_audio': self.ref_audio_path,
                    'ref_text': "So I'm live on radio."
                }

                state = self.tts_irp.init_state(
                    x0=tts_input,
                    task_ctx={'voice_id': 'dave', 'prosody': {'speed': 1.0}}
                )

                # Run refinement loop
                for i in range(self.max_iterations):
                    state, budget_used = self.tts_irp.step(state, budget=10.0)
                    energy = self.tts_irp.energy(state)

                    # Early stopping if quality is good
                    if energy < 0.3:
                        break

                # Extract final audio
                result = self.tts_irp.extract(state)
                audio = result['audio']
                sample_rate = result['sample_rate']

                # Play audio
                status = self._play_audio(audio, sample_rate)

                return EffectorResult(
                    effector_id=self.effector_id,
                    status=status,
                    message=f"Spoke: {text[:50]}..." if len(text) > 50 else f"Spoke: {text}",
                    execution_time=time.time() - start_time,
                    metadata={
                        'text': text,
                        'confidence': result['confidence'],
                        'iterations': result['iterations']
                    }
                )

            elif command.action == 'play':
                # Direct audio playback
                if command.data is None:
                    return EffectorResult(
                        effector_id=self.effector_id,
                        status=EffectorStatus.INVALID_COMMAND,
                        message="No audio data provided for playback",
                        execution_time=time.time() - start_time
                    )

                # Convert tensor to numpy
                audio = command.data.cpu().numpy()
                sample_rate = command.parameters.get('sample_rate', self.sample_rate)

                # Play audio
                status = self._play_audio(audio, sample_rate)

                return EffectorResult(
                    effector_id=self.effector_id,
                    status=status,
                    message=f"Played audio ({len(audio)} samples)",
                    execution_time=time.time() - start_time
                )

            else:
                return EffectorResult(
                    effector_id=self.effector_id,
                    status=EffectorStatus.INVALID_COMMAND,
                    message=f"Unknown action: {command.action}",
                    execution_time=time.time() - start_time
                )

        except Exception as e:
            return EffectorResult(
                effector_id=self.effector_id,
                status=EffectorStatus.FAILED,
                message=f"Execution error: {e}",
                execution_time=time.time() - start_time
            )

    def _play_audio(self, audio: np.ndarray, sample_rate: int) -> EffectorStatus:
        """
        Play audio waveform through Bluetooth speaker.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            EffectorStatus indicating success or failure
        """
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name
                sf.write(temp_wav, audio, sample_rate)

            # Play through Bluetooth speaker
            result = subprocess.run(
                ["paplay", "--device", self.bt_device, temp_wav],
                capture_output=True,
                timeout=30,
                check=False
            )

            # Cleanup
            os.unlink(temp_wav)

            if result.returncode == 0:
                return EffectorStatus.SUCCESS
            else:
                print(f"⚠️ paplay error: {result.stderr.decode()}")
                return EffectorStatus.HARDWARE_UNAVAILABLE

        except subprocess.TimeoutExpired:
            return EffectorStatus.TIMEOUT
        except Exception as e:
            print(f"⚠️ Audio playback error: {e}")
            return EffectorStatus.FAILED

    async def execute_async(self, command: EffectorCommand) -> EffectorResult:
        """Async version of execute (currently just wraps sync version)."""
        import asyncio
        return await asyncio.to_thread(self.execute, command)

    def validate_command(self, command: EffectorCommand) -> bool:
        """Validate command before execution."""
        if command.action == 'speak':
            return 'text' in command.parameters and len(command.parameters['text']) > 0
        elif command.action == 'play':
            return command.data is not None
        return False

    def is_available(self) -> bool:
        """Check if audio output hardware is available."""
        try:
            # Check if Bluetooth device is available
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return self.bt_device in result.stdout
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Return effector capabilities (required by BaseEffector)."""
        return {
            'effector_id': self.effector_id,
            'effector_type': 'audio',
            'actions': ['speak', 'play'],
            'bluetooth_device': self.bt_device,
            'sample_rate': self.sample_rate,
            'tts_device': self.neutts_device,
            'available': self.is_available(),
            'enabled': self.enabled
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Alias for get_info()."""
        return self.get_info()
