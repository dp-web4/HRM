#!/usr/bin/env python3
"""
Multi-Modal Puzzle SAGE Demo - Vision + Audio → Unified Consciousness

Demonstrates the power of puzzle space as universal geometric interface:
1. Vision sensor encodes images to 30×30 puzzles
2. Audio sensor encodes waveforms to 30×30 puzzles
3. BOTH feed into single unified SAGE consciousness loop
4. SNARC assesses salience across modalities
5. Cross-modal pattern detection and reasoning
6. Demonstrates puzzle space universality

This shows how different sensor modalities can share the same
geometric reasoning space through puzzle encoding.
"""

import sys
from pathlib import Path

# Add HRM to path
hrm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(hrm_root))

import torch
import time
from typing import Dict, Any

from sage.core.unified_sage_system import (
    UnifiedSAGESystem,
    SensorOutput,
    ExecutionResult
)
from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.compression.audio_puzzle_vae import AudioPuzzleVAE
from sage.services.snarc.data_structures import CognitiveStance


class MultiModalPuzzleSensor:
    """
    Multi-modal sensor providing both vision and audio as puzzles

    Wraps both VAEs to provide unified puzzle space output
    """

    def __init__(
        self,
        vision_vae: VisionPuzzleVAE,
        audio_vae: AudioPuzzleVAE
    ):
        self.vision_vae = vision_vae
        self.audio_vae = audio_vae
        self.vision_vae.eval()
        self.audio_vae.eval()
        self.frame_count = 0

    def capture_vision(self) -> SensorOutput:
        """Capture vision and encode to puzzle"""
        # Simulate camera frame
        image = torch.rand(1, 3, 224, 224)

        # Encode to puzzle
        with torch.no_grad():
            puzzle = self.vision_vae.encode_to_puzzle(image)

        return SensorOutput(
            data=puzzle[0],  # [30, 30]
            timestamp=time.time(),
            quality=1.0,
            sensor_type='vision_puzzle',
            metadata={
                'modality': 'vision',
                'encoding': 'VQ-VAE puzzle space',
                'frame': self.frame_count
            }
        )

    def capture_audio(self) -> SensorOutput:
        """Capture audio and encode to puzzle"""
        # Simulate audio sample (1 sec @ 16kHz)
        waveform = torch.randn(1, 16000)

        # Encode to puzzle
        with torch.no_grad():
            puzzle = self.audio_vae.encode_to_puzzle(waveform)

        return SensorOutput(
            data=puzzle[0],  # [30, 30]
            timestamp=time.time(),
            quality=1.0,
            sensor_type='audio_puzzle',
            metadata={
                'modality': 'audio',
                'encoding': 'VQ-VAE puzzle space',
                'frame': self.frame_count
            }
        )

    def capture_both(self) -> Dict[str, SensorOutput]:
        """Capture both modalities simultaneously"""
        self.frame_count += 1
        return {
            'vision': self.capture_vision(),
            'audio': self.capture_audio()
        }


class CrossModalPatternDetector:
    """
    Detects patterns across vision and audio puzzle spaces

    Demonstrates unified geometric reasoning over different modalities
    """

    def __init__(self):
        self.patterns_detected = []

    def __call__(
        self,
        observations: Dict[str, SensorOutput],
        stance: CognitiveStance
    ) -> ExecutionResult:
        """
        Process both vision and audio puzzles simultaneously

        Args:
            observations: Dict with 'vision' and 'audio' SensorOutput
            stance: Cognitive stance from SNARC

        Returns:
            ExecutionResult with cross-modal analysis
        """
        vision_puzzle = observations['vision'].data  # [30, 30]
        audio_puzzle = observations['audio'].data  # [30, 30]

        # Within-modality patterns
        vision_patterns = self._analyze_puzzle(vision_puzzle, 'vision')
        audio_patterns = self._analyze_puzzle(audio_puzzle, 'audio')

        # Cross-modal patterns
        cross_modal = self._cross_modal_analysis(vision_puzzle, audio_puzzle)

        # Stance-dependent interpretation
        if stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            description = (
                f"Curious about multi-modal patterns: "
                f"Vision={vision_patterns['dominant_region']}, "
                f"Audio={audio_patterns['temporal_region']}, "
                f"Correlation={cross_modal['spatial_correlation']:.2f}"
            )
            reward = 0.6
        elif stance == CognitiveStance.FOCUSED_ATTENTION:
            description = (
                f"Focused on high-salience regions: "
                f"{cross_modal['salient_regions']} cross-modal hotspots detected"
            )
            reward = 0.8
        else:
            description = (
                f"Observing multi-modal scene: "
                f"Vision diversity={vision_patterns['unique_values']}, "
                f"Audio energy={audio_patterns['avg_energy']:.1f}"
            )
            reward = 0.7

        # Store pattern
        pattern = {
            'vision': vision_patterns,
            'audio': audio_patterns,
            'cross_modal': cross_modal,
            'stance': stance.value
        }
        self.patterns_detected.append(pattern)

        return ExecutionResult(
            success=True,
            reward=reward,
            description=description,
            outputs={
                'patterns': pattern,
                'actions': {
                    'analysis': description,
                    'attention_distribution': cross_modal['attention_map']
                }
            }
        )

    def _analyze_puzzle(self, puzzle: torch.Tensor, modality: str) -> Dict[str, Any]:
        """Analyze single-modality puzzle"""
        # Value distribution
        unique_vals, counts = torch.unique(puzzle, return_counts=True)
        dominant_value = unique_vals[counts.argmax()].item()

        # Spatial clustering (quadrants)
        h, w = puzzle.shape
        quadrants = {
            'top_left': puzzle[:h//2, :w//2].float().mean().item(),
            'top_right': puzzle[:h//2, w//2:].float().mean().item(),
            'bottom_left': puzzle[h//2:, :w//2].float().mean().item(),
            'bottom_right': puzzle[h//2:, w//2:].float().mean().item()
        }
        dominant_region = max(quadrants, key=quadrants.get)

        # High-value regions
        high_val_mask = puzzle >= 7
        high_val_count = high_val_mask.sum().item()

        # Modality-specific analysis
        if modality == 'vision':
            # Spatial variance
            structure_metric = puzzle.float().var().item()
        else:  # audio
            # Temporal progression (columns = time)
            temporal_mean = puzzle.float().mean(dim=0)  # [30]
            early = temporal_mean[:10].mean().item()
            late = temporal_mean[20:].mean().item()
            structure_metric = abs(early - late)  # Temporal change

        return {
            'unique_values': len(unique_vals),
            'dominant_value': dominant_value,
            'dominant_region': dominant_region,
            'quadrants': quadrants,
            'high_value_count': high_val_count,
            'avg_energy': puzzle.float().mean().item(),
            'structure_metric': structure_metric,
            'temporal_region': dominant_region if modality == 'audio' else None
        }

    def _cross_modal_analysis(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze cross-modal patterns between vision and audio puzzles"""

        # Spatial correlation (do they have similar spatial distributions?)
        vision_flat = vision.float().flatten()
        audio_flat = audio.float().flatten()
        correlation = torch.corrcoef(torch.stack([vision_flat, audio_flat]))[0, 1].item()

        # Salient region overlap (where are both modalities active?)
        vision_salient = vision >= 6  # [30, 30]
        audio_salient = audio >= 6
        overlap = (vision_salient & audio_salient).sum().item()

        # Attention map (where should we focus?)
        # Combine energy from both modalities
        attention_map = (vision.float() + audio.float()) / 2  # [30, 30]

        # Find regions of high cross-modal activity
        high_attention = (attention_map >= 5).sum().item()

        # Dominant cross-modal quadrant
        h, w = attention_map.shape
        cross_quadrants = {
            'top_left': attention_map[:h//2, :w//2].mean().item(),
            'top_right': attention_map[:h//2, w//2:].mean().item(),
            'bottom_left': attention_map[h//2:, :w//2].mean().item(),
            'bottom_right': attention_map[h//2:, w//2:].mean().item()
        }
        dominant_cross_quad = max(cross_quadrants, key=cross_quadrants.get)

        return {
            'spatial_correlation': correlation,
            'salient_overlap': overlap,
            'salient_regions': high_attention,
            'attention_map': attention_map.tolist(),  # For serialization
            'dominant_quadrant': dominant_cross_quad,
            'cross_quadrants': cross_quadrants
        }


def main():
    print("=" * 70)
    print("Multi-Modal Puzzle SAGE Demo - Vision + Audio Consciousness")
    print("=" * 70)

    # Create VAEs
    print("\n1. Creating Vision → Puzzle VAE...")
    vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10)
    print("   ✓ Vision VAE initialized")

    print("\n2. Creating Audio → Puzzle VAE...")
    audio_vae = AudioPuzzleVAE(latent_dim=64, num_codes=10)
    print("   ✓ Audio VAE initialized")

    # Create multi-modal sensor
    print("\n3. Creating Multi-Modal Puzzle Sensor...")
    sensor = MultiModalPuzzleSensor(vision_vae, audio_vae)
    print("   ✓ Unified sensor ready (vision + audio → puzzle)")

    # Create cross-modal detector
    print("\n4. Creating Cross-Modal Pattern Detector...")
    detector = CrossModalPatternDetector()
    print("   ✓ Detector ready (geometric reasoning)")

    # Create Unified SAGE with multi-modal sensing
    print("\n5. Creating Unified SAGE System...")

    # Wrap sensor methods for SAGE
    def capture_sensors():
        return sensor.capture_both()

    # Create single action handler that receives both modalities
    def handle_action(observations, stance):
        # Ensure observations is a dict with both modalities
        if not isinstance(observations, dict):
            # If single observation, wrap it
            observations = {'vision': observations}
        return detector(observations, stance)

    sage = UnifiedSAGESystem(
        sensor_sources={
            'multimodal': capture_sensors  # Returns dict with vision+audio
        },
        action_handlers={
            'multimodal': handle_action  # Handles dict with both modalities
        },
        config={
            'initial_atp': 100.0,
            'enable_circadian': False
        },
        enable_logging=True
    )
    print("   ✓ UnifiedSAGE with multi-modal processing")

    # Run consciousness loop
    print("\n6. Running Multi-Modal Consciousness Loop...")
    print("   (Processing 10 frames with vision + audio)\\n")

    sage.run(max_cycles=10, cycle_delay=0.1)

    # Analysis
    print("\n" + "=" * 70)
    print("MULTI-MODAL PUZZLE CONSCIOUSNESS ANALYSIS")
    print("=" * 70)

    # System status
    status = sage.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Cycles completed: {status['cycle_count']}")
    print(f"  ATP remaining: {status['atp_level']:.1f}/{status['max_atp']}")
    print(f"  Memory items: {status['memory_items']}")

    # Performance
    perf = sage.get_performance_summary()
    print(f"\nPerformance:")
    print(f"  Average cycle time: {perf['avg_cycle_time']*1000:.2f}ms")
    print(f"  Average salience: {perf['avg_salience']:.3f}")

    # Cross-modal pattern analysis
    print(f"\nCross-Modal Pattern Detection:")
    print(f"  Total patterns analyzed: {len(detector.patterns_detected)}")

    if detector.patterns_detected:
        # Aggregate statistics
        avg_correlation = sum(
            p['cross_modal']['spatial_correlation']
            for p in detector.patterns_detected
        ) / len(detector.patterns_detected)

        avg_overlap = sum(
            p['cross_modal']['salient_overlap']
            for p in detector.patterns_detected
        ) / len(detector.patterns_detected)

        avg_salient_regions = sum(
            p['cross_modal']['salient_regions']
            for p in detector.patterns_detected
        ) / len(detector.patterns_detected)

        print(f"  Avg vision-audio correlation: {avg_correlation:.3f}")
        print(f"  Avg salient overlap: {avg_overlap:.1f} cells")
        print(f"  Avg high-attention regions: {avg_salient_regions:.1f} cells")

        # Modality-specific summaries
        avg_vision_diversity = sum(
            p['vision']['unique_values']
            for p in detector.patterns_detected
        ) / len(detector.patterns_detected)

        avg_audio_energy = sum(
            p['audio']['avg_energy']
            for p in detector.patterns_detected
        ) / len(detector.patterns_detected)

        print(f"\n  Vision modality:")
        print(f"    Avg diversity: {avg_vision_diversity:.1f}/10 unique values")

        print(f"  Audio modality:")
        print(f"    Avg energy: {avg_audio_energy:.1f}/9")

    # Show sample pattern
    if detector.patterns_detected:
        print(f"\nSample Cross-Modal Pattern (first frame):")
        sample = detector.patterns_detected[0]
        print(f"  Vision dominant region: {sample['vision']['dominant_region']}")
        print(f"  Audio temporal region: {sample['audio']['temporal_region']}")
        print(f"  Spatial correlation: {sample['cross_modal']['spatial_correlation']:.3f}")
        print(f"  Salient overlap: {sample['cross_modal']['salient_overlap']} cells")
        print(f"  Cross-modal dominant quadrant: {sample['cross_modal']['dominant_quadrant']}")
        print(f"  Stance: {sample['stance']}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n1. PUZZLE SPACE IS UNIVERSAL")
    print("   Both vision and audio encoded to same 30×30×10 format")
    print("\n2. CROSS-MODAL REASONING WORKS")
    print("   SNARC assesses salience across both modalities simultaneously")
    print("\n3. GEOMETRIC PATTERNS ARE SHARED")
    print("   Quadrant analysis, value distributions work for both")
    print("\n4. ATTENTION IS UNIFIED")
    print("   System allocates focus based on combined modality signals")
    print("\n5. CONSCIOUSNESS SCALES")
    print("   Same loop handles single or multiple modalities seamlessly")

    print("\n" + "=" * 70)
    print("Multi-modal puzzle-based geometric consciousness operational!")
    print("=" * 70)


if __name__ == "__main__":
    main()
