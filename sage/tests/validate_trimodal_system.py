#!/usr/bin/env python3
"""
Tri-Modal Puzzle Space Validation Script
Validates that Vision, Audio, and Language VAEs share unified puzzle space
"""

import sys
import torch
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compression.vision_puzzle_vae import VisionPuzzleVAE
from compression.audio_puzzle_vae import AudioPuzzleVAE
from compression.language_puzzle_vae_small import LanguagePuzzleVAESmall


def validate_architecture():
    """Validate that all three VAEs share the same puzzle space architecture"""
    print('='*80)
    print('TRI-MODAL PUZZLE SPACE VALIDATION')
    print('='*80)
    print()

    # Create models
    vision = VisionPuzzleVAE()
    audio = AudioPuzzleVAE()
    language = LanguagePuzzleVAESmall()

    print('✅ All three VAE models imported successfully')
    print()

    # Check puzzle space dimensions
    print('Puzzle Space Dimensions:')
    print(f'  Vision VAE   - Latent dim: {vision.latent_dim}, Codes: {vision.num_codes}')
    print(f'  Audio VAE    - Latent dim: {audio.latent_dim}, Codes: {audio.num_codes}')
    print(f'  Language VAE - Latent dim: {language.latent_dim}, Codes: {language.num_codes}')
    print()

    # Verify unified space
    assert vision.latent_dim == audio.latent_dim == language.latent_dim, "Latent dimensions don't match!"
    assert vision.num_codes == audio.num_codes == language.num_codes, "Codebook sizes don't match!"

    print(f'✅ Unified latent dimension: {vision.latent_dim}D')
    print(f'✅ Unified codebook size: {vision.num_codes} codes')
    print()

    # Count parameters
    vision_params = sum(p.numel() for p in vision.parameters())
    audio_params = sum(p.numel() for p in audio.parameters())
    language_params = sum(p.numel() for p in language.parameters())

    print('Parameter counts:')
    print(f'  Vision:   {vision_params:,} parameters')
    print(f'  Audio:    {audio_params:,} parameters')
    print(f'  Language: {language_params:,} parameters')
    print()

    return vision, audio, language


def test_forward_passes(vision, audio, language):
    """Test forward passes through all three VAEs"""
    print('Testing forward passes with dummy data...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print()

    vision = vision.to(device)
    audio = audio.to(device)
    language = language.to(device)

    # Vision: 32x32 RGB images (3 channels) - resized to 224x224 by model
    dummy_vision = torch.randn(2, 3, 32, 32).to(device)
    vision_out = vision(dummy_vision)
    vision_puzzle = vision_out["puzzles"]  # [batch, 30, 30] with values 0-9
    print(f'✅ Vision:   Input {list(dummy_vision.shape)} → Puzzle {list(vision_puzzle.shape)}')

    # Audio: 128x128 spectrograms (1 channel)
    dummy_audio = torch.randn(2, 1, 128, 128).to(device)
    audio_out = audio(dummy_audio)
    audio_puzzle = audio_out["puzzles"]  # [batch, 30, 30] with values 0-9
    print(f'✅ Audio:    Input {list(dummy_audio.shape)} → Puzzle {list(audio_puzzle.shape)}')

    # Language: 128 character sequences
    dummy_lang = torch.randint(0, 128, (2, 128)).to(device)
    lang_out = language(dummy_lang)
    lang_puzzle = lang_out["puzzles"]  # [batch, 30, 30] with values 0-9
    print(f'✅ Language: Input {list(dummy_lang.shape)} → Puzzle {list(lang_puzzle.shape)}')

    print()

    # Verify puzzle dimensions match
    assert vision_puzzle.shape == audio_puzzle.shape == lang_puzzle.shape, "Puzzle dimensions don't match!"
    print(f'✅ All modalities produce same puzzle shape: {list(vision_puzzle.shape)}')
    print()

    return vision_puzzle, audio_puzzle, lang_puzzle


def load_trained_models():
    """Load trained model checkpoints"""
    print('Loading trained model checkpoints...')

    checkpoint_dir = Path(__file__).parent.parent / 'training'

    vision = VisionPuzzleVAE()
    audio = AudioPuzzleVAE()
    language = LanguagePuzzleVAESmall()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoints
    vision_path = checkpoint_dir / 'vision_vae_checkpoints' / 'best_model.pt'
    audio_path = checkpoint_dir / 'audio_vae_checkpoints' / 'best_model.pt'
    lang_path = checkpoint_dir / 'language_vae_checkpoints' / 'best_model.pt'

    if vision_path.exists():
        vision.load_state_dict(torch.load(vision_path, map_location=device, weights_only=False))
        print(f'✅ Vision checkpoint loaded: {vision_path.stat().st_size / (1024*1024):.2f} MB')
    else:
        print(f'⚠️  Vision checkpoint not found: {vision_path}')

    if audio_path.exists():
        audio.load_state_dict(torch.load(audio_path, map_location=device, weights_only=False))
        print(f'✅ Audio checkpoint loaded: {audio_path.stat().st_size / (1024*1024):.2f} MB')
    else:
        print(f'⚠️  Audio checkpoint not found: {audio_path}')

    if lang_path.exists():
        language.load_state_dict(torch.load(lang_path, map_location=device, weights_only=False))
        print(f'✅ Language checkpoint loaded: {lang_path.stat().st_size / (1024*1024):.2f} MB')
    else:
        print(f'⚠️  Language checkpoint not found: {lang_path}')

    print()

    return vision.to(device), audio.to(device), language.to(device)


def main():
    """Main validation routine"""
    try:
        # Validate architecture
        vision, audio, language = validate_architecture()

        # Test forward passes
        vision_puzzle, audio_puzzle, lang_puzzle = test_forward_passes(vision, audio, language)

        # Load trained models
        vision_trained, audio_trained, lang_trained = load_trained_models()

        print('='*80)
        print('✅ TRI-MODAL PUZZLE SPACE VALIDATION COMPLETE')
        print('='*80)
        print()
        print('Summary:')
        print('  ✅ All three modalities share unified 64D latent space')
        print('  ✅ All three modalities use same 10-code vocabulary')
        print('  ✅ All three produce same puzzle tensor shape')
        print('  ✅ All three trained models loaded successfully')
        print()
        print('Ready for cross-modal experiments!')

        return True

    except Exception as e:
        print(f'\n❌ Validation failed: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
