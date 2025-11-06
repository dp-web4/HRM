#!/usr/bin/env python3
"""
Real-Time SAGE Consciousness Demo
==================================

Complete demonstration of SAGE with REAL sensor integration:
- Camera (auto-detects: OpenCV → GR00T → synthetic)
- Audio (auto-detects: PyAudio → synthetic)
- Vision Puzzle VAE encoding
- Audio Puzzle VAE encoding
- SNARC salience assessment
- Memory storage
- Continuous consciousness loop

This is Phase 1 of Nano deployment - proving real sensor integration works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import time
from typing import Dict, Any
from dataclasses import dataclass

from sage.sensors.camera_sensor import CameraSensor
from sage.sensors.audio_sensor import AudioSensor
from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.compression.audio_puzzle_vae import AudioPuzzleVAE
from sage.services.snarc import SNARCService, SalienceReport
from sage.core.unified_sage_system import SensorOutput


@dataclass
class RealtimeStats:
    """Performance statistics for realtime demo"""
    cycle_count: int = 0
    total_time: float = 0.0
    camera_latency: float = 0.0
    audio_latency: float = 0.0
    vision_encode_latency: float = 0.0
    audio_encode_latency: float = 0.0
    snarc_latency: float = 0.0


class RealtimeSAGELoop:
    """
    Real-time SAGE consciousness loop with real sensors.

    This demonstrates the complete pipeline:
    1. Capture from real sensors (camera + audio)
    2. Encode to puzzle space via VAEs
    3. Assess salience via SNARC
    4. Store salient experiences in memory
    5. Track performance metrics
    """

    def __init__(
        self,
        device: str = "cuda",
        camera_backend: str = "auto",
        audio_backend: str = "auto",
        target_fps: int = 10,
    ):
        """
        Initialize real-time SAGE loop.

        Args:
            device: Device for tensor operations
            camera_backend: Camera backend ('auto', 'opencv', 'groot', 'synthetic')
            audio_backend: Audio backend ('auto', 'pyaudio', 'synthetic')
            target_fps: Target frames per second for main loop
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.target_fps = target_fps
        self.cycle_interval = 1.0 / target_fps

        print("="*70)
        print("SAGE Real-Time Consciousness Loop")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Target FPS: {target_fps}")
        print()

        # Initialize sensors
        print("Initializing sensors...")
        self.camera = CameraSensor(
            backend=camera_backend,
            device=self.device,
            fps=target_fps
        )
        self.audio = AudioSensor(
            backend=audio_backend,
            device=self.device
        )
        print()

        # Initialize VAEs
        print("Loading VAE models...")
        self.vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10).to(self.device)
        self.vision_vae.eval()
        self.audio_vae = AudioPuzzleVAE(latent_dim=64, num_codes=10).to(self.device)
        self.audio_vae.eval()
        print("✓ VAEs loaded (untrained, using random weights)")
        print()

        # Initialize SNARC
        self.snarc = SNARCService()
        print("✓ SNARC service ready")
        print()

        # Statistics
        self.stats = RealtimeStats()
        self.experiences = []

    def process_cycle(self) -> Dict[str, Any]:
        """
        Process single consciousness cycle.

        Returns:
            Cycle results including observations, puzzles, salience
        """
        cycle_start = time.time()

        # 1. Capture from sensors
        t_start = time.time()
        camera_frame = self.camera.capture()
        camera_latency = (time.time() - t_start) * 1000
        self.stats.camera_latency = camera_latency

        t_start = time.time()
        audio_chunk = self.audio.capture()
        audio_latency = (time.time() - t_start) * 1000
        self.stats.audio_latency = audio_latency

        if camera_frame is None or audio_chunk is None:
            return None

        # 2. Encode to puzzle space
        with torch.no_grad():
            # Vision encoding
            t_start = time.time()
            vision_puzzle = self.vision_vae.encode_to_puzzle(
                camera_frame.image.unsqueeze(0)  # Add batch dim
            )
            vision_encode_latency = (time.time() - t_start) * 1000
            self.stats.vision_encode_latency = vision_encode_latency

            # Audio encoding
            t_start = time.time()
            audio_puzzle = self.audio_vae.encode_to_puzzle(
                audio_chunk.waveform.unsqueeze(0)  # Add batch dim
            )
            audio_encode_latency = (time.time() - t_start) * 1000
            self.stats.audio_encode_latency = audio_encode_latency

        # 3. Create sensor outputs
        vision_obs = SensorOutput(
            data=vision_puzzle,
            timestamp=camera_frame.timestamp,
            quality=1.0,
            sensor_type='vision',
            metadata=camera_frame.metadata,
        )

        audio_obs = SensorOutput(
            data=audio_puzzle,
            timestamp=audio_chunk.timestamp,
            quality=1.0,
            sensor_type='audio',
            metadata=audio_chunk.metadata,
        )

        observations = {
            'vision': vision_obs,
            'audio': audio_obs,
        }

        # 4. SNARC salience assessment
        t_start = time.time()
        snarc_output = self.snarc.assess_salience(observations)
        snarc_latency = (time.time() - t_start) * 1000
        self.stats.snarc_latency = snarc_latency

        # 5. Store salient experiences
        if snarc_output.salience_score > 0.5:
            experience = {
                'cycle': self.stats.cycle_count,
                'timestamp': time.time(),
                'observations': observations,
                'snarc': snarc_output,
                'latencies': {
                    'camera_ms': camera_latency,
                    'audio_ms': audio_latency,
                    'vision_encode_ms': vision_encode_latency,
                    'audio_encode_ms': audio_encode_latency,
                    'snarc_ms': snarc_latency,
                }
            }
            self.experiences.append(experience)

        # 6. Update statistics
        cycle_time = time.time() - cycle_start
        self.stats.cycle_count += 1
        self.stats.total_time += cycle_time

        # Rate limiting
        if cycle_time < self.cycle_interval:
            time.sleep(self.cycle_interval - cycle_time)

        return {
            'cycle': self.stats.cycle_count,
            'observations': observations,
            'snarc': snarc_output,
            'cycle_time_ms': cycle_time * 1000,
        }

    def run(self, max_cycles: int = 100, print_interval: int = 10):
        """
        Run continuous consciousness loop.

        Args:
            max_cycles: Maximum cycles to run (0 = infinite)
            print_interval: Print stats every N cycles
        """
        print("="*70)
        print(f"Starting SAGE consciousness loop (max {max_cycles} cycles)")
        print("="*70)
        print()

        try:
            cycle = 0
            while max_cycles == 0 or cycle < max_cycles:
                result = self.process_cycle()

                if result is not None and (cycle + 1) % print_interval == 0:
                    self.print_status(result)

                cycle += 1

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        finally:
            self.print_final_stats()
            self.cleanup()

    def print_status(self, result: Dict[str, Any]):
        """Print current cycle status"""
        cycle = result['cycle']
        snarc = result['snarc']
        breakdown = snarc.salience_breakdown

        print(f"Cycle {cycle:3d} | "
              f"Time: {result['cycle_time_ms']:5.1f}ms | "
              f"Salience: {snarc.salience_score:.3f} | "
              f"S:{breakdown.surprise:.2f} N:{breakdown.novelty:.2f} "
              f"A:{breakdown.arousal:.2f} R:{breakdown.reward:.2f} C:{breakdown.conflict:.2f}")

    def print_final_stats(self):
        """Print final performance statistics"""
        print("\n" + "="*70)
        print("SAGE Real-Time Performance Report")
        print("="*70)

        if self.stats.cycle_count == 0:
            print("No cycles completed")
            return

        avg_cycle_time = (self.stats.total_time / self.stats.cycle_count) * 1000
        actual_fps = 1000.0 / avg_cycle_time if avg_cycle_time > 0 else 0

        print(f"\n📊 Overall Performance:")
        print(f"   Cycles completed: {self.stats.cycle_count}")
        print(f"   Total time: {self.stats.total_time:.2f}s")
        print(f"   Avg cycle time: {avg_cycle_time:.2f}ms")
        print(f"   Actual FPS: {actual_fps:.1f} (target: {self.target_fps})")

        print(f"\n⏱️  Component Latencies (last cycle):")
        print(f"   Camera capture: {self.stats.camera_latency:.2f}ms")
        print(f"   Audio capture: {self.stats.audio_latency:.2f}ms")
        print(f"   Vision encoding: {self.stats.vision_encode_latency:.2f}ms")
        print(f"   Audio encoding: {self.stats.audio_encode_latency:.2f}ms")
        print(f"   SNARC assessment: {self.stats.snarc_latency:.2f}ms")

        total_latency = (
            self.stats.camera_latency +
            self.stats.audio_latency +
            self.stats.vision_encode_latency +
            self.stats.audio_encode_latency +
            self.stats.snarc_latency
        )
        print(f"   Total pipeline: {total_latency:.2f}ms")

        print(f"\n💾 Memory:")
        print(f"   Salient experiences: {len(self.experiences)}")
        if len(self.experiences) > 0:
            avg_salience = sum(e['snarc'].salience_score for e in self.experiences) / len(self.experiences)
            print(f"   Avg salience: {avg_salience:.3f}")

        # Sensor stats
        camera_stats = self.camera.get_stats()
        audio_stats = self.audio.get_stats()

        print(f"\n📷 Camera:")
        print(f"   Backend: {camera_stats['backend']}")
        print(f"   Frames: {camera_stats['frames_captured']}")
        print(f"   Avg FPS: {camera_stats.get('avg_fps', 0):.1f}")

        print(f"\n🎤 Audio:")
        print(f"   Backend: {audio_stats['backend']}")
        print(f"   Chunks: {audio_stats['chunks_captured']}")
        print(f"   Sample rate: {audio_stats['sample_rate']}Hz")

        # Real-time capability assessment
        print(f"\n✓ Real-Time Assessment:")
        if actual_fps >= self.target_fps * 0.9:
            status = "EXCELLENT - Hitting target FPS"
        elif actual_fps >= self.target_fps * 0.7:
            status = "GOOD - Near target FPS"
        elif actual_fps >= self.target_fps * 0.5:
            status = "ACCEPTABLE - Usable performance"
        else:
            status = "NEEDS OPTIMIZATION - Below target"
        print(f"   Status: {status}")

        print("\n" + "="*70)

    def cleanup(self):
        """Cleanup resources"""
        self.camera.close()
        self.audio.close()
        print("\n✓ Resources released")


def main():
    """Run real-time SAGE demo"""
    import argparse

    parser = argparse.ArgumentParser(description="SAGE Real-Time Demo")
    parser.add_argument('--camera', default='auto', choices=['auto', 'opencv', 'groot', 'synthetic'],
                        help='Camera backend')
    parser.add_argument('--audio', default='auto', choices=['auto', 'pyaudio', 'synthetic'],
                        help='Audio backend')
    parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    parser.add_argument('--cycles', type=int, default=100, help='Max cycles (0=infinite)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    # Run demo
    sage = RealtimeSAGELoop(
        device=args.device,
        camera_backend=args.camera,
        audio_backend=args.audio,
        target_fps=args.fps,
    )

    sage.run(max_cycles=args.cycles, print_interval=10)


if __name__ == "__main__":
    main()
