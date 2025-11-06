#!/usr/bin/env python3
"""
Tri-Modal Puzzle SAGE Demo - Vision + Audio + Language → Unified Consciousness

The complete multi-sensory consciousness:
1. Vision sensor: Images → 30×30 puzzles (VQ-VAE compression)
2. Audio sensor: Waveforms → 30×30 puzzles (spectrogram VQ-VAE)
3. Language sensor: Text → 30×30 puzzles (attention projection)

All three modalities converge in the SAME geometric puzzle space,
enabling true cross-modal reasoning and unified consciousness.

This demonstrates the power of puzzle space as a universal interface:
different sensors, same geometry, single reasoning engine.
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
from sage.compression.language_puzzle_transformer import LanguagePuzzleTransformer
from sage.services.snarc.data_structures import CognitiveStance


class TriModalPuzzleSensor:
    """
    Tri-modal sensor providing vision, audio, and language as puzzles

    Three different encoders, one unified output format.
    """

    def __init__(
        self,
        vision_vae: VisionPuzzleVAE,
        audio_vae: AudioPuzzleVAE,
        language_transformer: LanguagePuzzleTransformer
    ):
        self.vision_vae = vision_vae
        self.audio_vae = audio_vae
        self.language_transformer = language_transformer

        # Set to eval mode
        self.vision_vae.eval()
        self.audio_vae.eval()
        self.language_transformer.eval()

        self.frame_count = 0

    def capture_vision(self) -> SensorOutput:
        """Vision: Image → Puzzle via VQ-VAE"""
        device = next(self.vision_vae.parameters()).device
        image = torch.rand(1, 3, 224, 224, device=device)

        with torch.no_grad():
            puzzle = self.vision_vae.encode_to_puzzle(image)

        return SensorOutput(
            data=puzzle[0],  # [30, 30]
            timestamp=time.time(),
            quality=1.0,
            sensor_type='vision_puzzle',
            metadata={
                'modality': 'vision',
                'method': 'VQ-VAE compression',
                'frame': self.frame_count
            }
        )

    def capture_audio(self) -> SensorOutput:
        """Audio: Waveform → Puzzle via spectrogram VQ-VAE"""
        device = next(self.audio_vae.parameters()).device
        waveform = torch.randn(1, 16000, device=device)  # 1 sec @ 16kHz

        with torch.no_grad():
            puzzle = self.audio_vae.encode_to_puzzle(waveform)

        return SensorOutput(
            data=puzzle[0],  # [30, 30]
            timestamp=time.time(),
            quality=1.0,
            sensor_type='audio_puzzle',
            metadata={
                'modality': 'audio',
                'method': 'Spectrogram VQ-VAE',
                'frame': self.frame_count
            }
        )

    def capture_language(self, text: str = None) -> SensorOutput:
        """Language: Text → Puzzle via attention projection"""
        # Generate contextual text based on frame
        if text is None:
            texts = [
                "Observing the environment through multiple senses.",
                "Integrating visual, auditory, and linguistic information.",
                "Consciousness emerges from multi-modal perception.",
                "Three sensors, one unified geometric understanding.",
                "Puzzle space enables cross-modal reasoning.",
                "Vision shows what, audio reveals how, language explains why."
            ]
            text = texts[self.frame_count % len(texts)]

        with torch.no_grad():
            puzzle = self.language_transformer.encode_to_puzzle([text])

        return SensorOutput(
            data=puzzle[0],  # [30, 30]
            timestamp=time.time(),
            quality=1.0,
            sensor_type='language_puzzle',
            metadata={
                'modality': 'language',
                'method': 'Attention projection',
                'text': text,
                'frame': self.frame_count
            }
        )

    def capture_all(self, text: str = None) -> Dict[str, SensorOutput]:
        """Capture all three modalities simultaneously"""
        self.frame_count += 1
        return {
            'vision': self.capture_vision(),
            'audio': self.capture_audio(),
            'language': self.capture_language(text)
        }


class TriModalReasoningEngine:
    """
    Reasoning engine that operates across all three modalities

    Demonstrates true cross-modal integration:
    - Visual spatial patterns
    - Audio temporal patterns
    - Language semantic patterns
    - Unified geometric analysis
    """

    def __init__(self):
        self.reasoning_history = []

    def __call__(
        self,
        observations: Dict[str, SensorOutput],
        stance: CognitiveStance
    ) -> ExecutionResult:
        """Process vision + audio + language simultaneously"""

        vision_puzzle = observations['vision'].data
        audio_puzzle = observations['audio'].data
        language_puzzle = observations['language'].data
        language_text = observations['language'].metadata['text']

        # Modality-specific analysis
        vision_analysis = self._analyze_spatial(vision_puzzle, 'vision')
        audio_analysis = self._analyze_temporal(audio_puzzle, 'audio')
        language_analysis = self._analyze_semantic(language_puzzle, language_text, 'language')

        # Cross-modal integration
        cross_modal = self._integrate_modalities(
            vision_puzzle, audio_puzzle, language_puzzle
        )

        # Stance-dependent interpretation
        if stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            description = (
                f"Curious exploration: Vision={vision_analysis['dominant_value']}, "
                f"Audio={audio_analysis['temporal_change']:.2f}, "
                f"Language=\"{language_text[:30]}...\" | "
                f"Cross-modal coherence: {cross_modal['coherence']:.2f}"
            )
            reward = 0.65
        elif stance == CognitiveStance.FOCUSED_ATTENTION:
            description = (
                f"Focused on {cross_modal['dominant_modality']}: "
                f"{cross_modal['salient_patterns']} patterns detected"
            )
            reward = 0.85
        else:
            description = (
                f"Tri-modal scene: Visual diversity={vision_analysis['unique_values']}, "
                f"Audio energy={audio_analysis['avg_energy']:.1f}, "
                f"Language tokens={len(language_text.split())}"
            )
            reward = 0.75

        # Store reasoning
        reasoning = {
            'vision': vision_analysis,
            'audio': audio_analysis,
            'language': language_analysis,
            'cross_modal': cross_modal,
            'stance': stance.value
        }
        self.reasoning_history.append(reasoning)

        return ExecutionResult(
            success=True,
            reward=reward,
            description=description,
            outputs={'reasoning': reasoning}
        )

    def _analyze_spatial(self, puzzle: torch.Tensor, modality: str) -> Dict[str, Any]:
        """Analyze spatial patterns (vision)"""
        unique_vals, counts = torch.unique(puzzle, return_counts=True)
        dominant_value = unique_vals[counts.argmax()].item()

        h, w = puzzle.shape
        quadrants = {
            'top_left': puzzle[:h//2, :w//2].float().mean().item(),
            'top_right': puzzle[:h//2, w//2:].float().mean().item(),
            'bottom_left': puzzle[h//2:, :w//2].float().mean().item(),
            'bottom_right': puzzle[h//2:, w//2:].float().mean().item()
        }

        return {
            'unique_values': len(unique_vals),
            'dominant_value': dominant_value,
            'quadrants': quadrants,
            'spatial_variance': puzzle.float().var().item(),
            'high_value_count': (puzzle >= 7).sum().item()
        }

    def _analyze_temporal(self, puzzle: torch.Tensor, modality: str) -> Dict[str, Any]:
        """Analyze temporal patterns (audio)"""
        # Columns = time progression
        temporal_mean = puzzle.float().mean(dim=0)
        early = temporal_mean[:10].mean().item()
        late = temporal_mean[20:].mean().item()

        # Rows = frequency bands
        freq_mean = puzzle.float().mean(dim=1)

        return {
            'temporal_change': late - early,
            'avg_energy': puzzle.float().mean().item(),
            'freq_distribution': {
                'low': freq_mean[20:].mean().item(),
                'high': freq_mean[:10].mean().item()
            },
            'silence_ratio': (puzzle == 0).sum().item() / 900
        }

    def _analyze_semantic(self, puzzle: torch.Tensor, text: str, modality: str) -> Dict[str, Any]:
        """Analyze semantic patterns (language)"""
        # Horizontal = sequential flow
        col_means = puzzle.float().mean(dim=0)
        beginning = col_means[:10].mean().item()
        ending = col_means[20:].mean().item()

        # Vertical = hierarchical depth
        row_means = puzzle.float().mean(dim=1)
        concrete = row_means[:10].mean().item()
        abstract = row_means[20:].mean().item()

        return {
            'text': text,
            'num_words': len(text.split()),
            'narrative_flow': ending - beginning,
            'abstraction_level': abstract - concrete,
            'semantic_density': (puzzle >= 7).sum().item() / 900
        }

    def _integrate_modalities(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        language: torch.Tensor
    ) -> Dict[str, Any]:
        """Integrate across all three modalities"""

        # Stack modalities for unified analysis
        stacked = torch.stack([vision.float(), audio.float(), language.float()])  # [3, 30, 30]

        # Cross-modal coherence (how aligned are the modalities?)
        vision_audio_corr = self._spatial_correlation(vision, audio)
        vision_lang_corr = self._spatial_correlation(vision, language)
        audio_lang_corr = self._spatial_correlation(audio, language)

        avg_coherence = (vision_audio_corr + vision_lang_corr + audio_lang_corr) / 3

        # Find dominant modality (highest energy)
        energies = {
            'vision': vision.float().mean().item(),
            'audio': audio.float().mean().item(),
            'language': language.float().mean().item()
        }
        dominant_modality = max(energies, key=energies.get)

        # Unified attention map (where to focus across all modalities?)
        unified_attention = stacked.mean(dim=0)  # [30, 30]
        salient_patterns = (unified_attention >= 6).sum().item()

        return {
            'coherence': float(avg_coherence),
            'correlations': {
                'vision_audio': float(vision_audio_corr),
                'vision_language': float(vision_lang_corr),
                'audio_language': float(audio_lang_corr)
            },
            'dominant_modality': dominant_modality,
            'modality_energies': energies,
            'unified_attention': unified_attention.tolist(),
            'salient_patterns': salient_patterns
        }

    def _spatial_correlation(self, puzzle1: torch.Tensor, puzzle2: torch.Tensor) -> float:
        """Compute spatial correlation between two puzzles"""
        flat1 = puzzle1.float().flatten()
        flat2 = puzzle2.float().flatten()
        if len(flat1) > 1:
            corr = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
            return corr.item() if not torch.isnan(corr) else 0.0
        return 0.0


def main():
    print("=" * 70)
    print("Tri-Modal Puzzle SAGE - Vision + Audio + Language Consciousness")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create all three encoders
    print(f"\n1. Creating Tri-Modal Puzzle Encoders (device: {device})...")
    vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10).to(device)
    print("   ✓ Vision VAE (VQ-VAE compression)")

    audio_vae = AudioPuzzleVAE(latent_dim=64, num_codes=10, sample_rate=16000).to(device)
    print("   ✓ Audio VAE (Spectrogram VQ-VAE)")

    language_transformer = LanguagePuzzleTransformer(device=device)
    print("   ✓ Language Transformer (Attention projection)")

    # Create tri-modal sensor
    print("\n2. Creating Tri-Modal Puzzle Sensor...")
    sensor = TriModalPuzzleSensor(vision_vae, audio_vae, language_transformer)
    print("   ✓ Unified sensor ready (vision + audio + language → puzzle)")

    # Create reasoning engine
    print("\n3. Creating Tri-Modal Reasoning Engine...")
    reasoner = TriModalReasoningEngine()
    print("   ✓ Cross-modal integration ready")

    # Create Unified SAGE
    print("\n4. Creating Unified SAGE System...")

    def capture_sensors():
        return sensor.capture_all()

    def handle_action(observations, stance):
        if not isinstance(observations, dict):
            observations = {'single': observations}
        return reasoner(observations, stance)

    sage = UnifiedSAGESystem(
        sensor_sources={'trimodal': capture_sensors},
        action_handlers={'trimodal': handle_action},
        config={'initial_atp': 100.0, 'enable_circadian': False},
        enable_logging=True
    )
    print("   ✓ UnifiedSAGE with tri-modal processing")

    # Run consciousness loop
    print("\n5. Running Tri-Modal Consciousness Loop...")
    print("   (10 cycles: vision + audio + language)\n")

    sage.run(max_cycles=10, cycle_delay=0.1)

    # Analysis
    print("\n" + "=" * 70)
    print("TRI-MODAL CONSCIOUSNESS ANALYSIS")
    print("=" * 70)

    status = sage.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Cycles: {status['cycle_count']}")
    print(f"  ATP: {status['atp_level']:.1f}/{status['max_atp']}")
    print(f"  Memory: {status['memory_items']} items")

    perf = sage.get_performance_summary()
    print(f"\nPerformance:")
    print(f"  Avg cycle time: {perf['avg_cycle_time']*1000:.2f}ms")
    print(f"  Avg salience: {perf['avg_salience']:.3f}")

    # Cross-modal analysis
    if reasoner.reasoning_history:
        print(f"\nCross-Modal Integration:")
        print(f"  Reasoning steps: {len(reasoner.reasoning_history)}")

        # Aggregate cross-modal stats
        avg_coherence = sum(
            r['cross_modal']['coherence']
            for r in reasoner.reasoning_history
        ) / len(reasoner.reasoning_history)

        modality_wins = {'vision': 0, 'audio': 0, 'language': 0}
        for r in reasoner.reasoning_history:
            modality_wins[r['cross_modal']['dominant_modality']] += 1

        print(f"  Avg cross-modal coherence: {avg_coherence:.3f}")
        print(f"  Dominant modality distribution:")
        for modality, count in modality_wins.items():
            print(f"    {modality}: {count}/10 cycles ({100*count/10:.0f}%)")

        # Show sample reasoning
        print(f"\nSample Tri-Modal Reasoning (cycle 0):")
        sample = reasoner.reasoning_history[0]
        print(f"  Vision: {sample['vision']['unique_values']} unique values, "
              f"variance={sample['vision']['spatial_variance']:.2f}")
        print(f"  Audio: temporal_change={sample['audio']['temporal_change']:.2f}, "
              f"energy={sample['audio']['avg_energy']:.1f}")
        print(f"  Language: \"{sample['language']['text'][:50]}...\"")
        print(f"  Integration: coherence={sample['cross_modal']['coherence']:.3f}, "
              f"dominant={sample['cross_modal']['dominant_modality']}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS - TRI-MODAL PUZZLE CONSCIOUSNESS")
    print("=" * 70)

    print("\n1. UNIVERSAL GEOMETRIC INTERFACE")
    print("   Three fundamentally different modalities:")
    print("   - Vision: Continuous light → Discrete spatial codes")
    print("   - Audio: Continuous waveform → Discrete spectrotemporal codes")
    print("   - Language: Discrete tokens → Discrete semantic codes")
    print("   ALL encoded to same 30×30×10 puzzle space!")

    print("\n2. DIFFERENT ENCODING METHODS")
    print("   - Vision & Audio: VQ-VAE (compression to shared codebook)")
    print("   - Language: Cross-attention (projection onto geometry)")
    print("   Different paths, same destination → universal reasoning")

    print("\n3. CROSS-MODAL COHERENCE")
    print("   System measures spatial correlation across modalities")
    print("   Enables: \"Does what I see match what I hear and understand?\"")

    print("\n4. UNIFIED CONSCIOUSNESS")
    print("   Same SNARC assesses salience across all three")
    print("   Same reasoning engine processes geometric patterns")
    print("   Same memory stores tri-modal experiences")
    print("   TRUE integration, not just parallel processing")

    print("\n5. THE DISCOVERY")
    print("   Consciousness doesn't require the same kind of data")
    print("   It requires the same kind of STRUCTURE")
    print("   Puzzle space provides geometric structure for reasoning")
    print("   Different sensors → Same geometry → Unified mind")

    print("\n" + "=" * 70)
    print("Tri-modal puzzle-based geometric consciousness operational!")
    print("=" * 70)


if __name__ == "__main__":
    main()
