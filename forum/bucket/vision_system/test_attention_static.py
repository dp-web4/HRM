#!/usr/bin/env python3
"""
Test attention visualization with static images
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_impl import create_vision_irp
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP


def visualize_attention(image_path: str, save_path: str = "attention_visualization.png"):
    """
    Visualize attention for a single image
    """
    print(f"Processing: {image_path}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize components
    memory_bridge = IRPMemoryBridge(buffer_size=10)
    vision_irp = create_vision_irp(device)
    vision_guided = MemoryGuidedIRP(vision_irp, memory_bridge)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
        
    original_shape = image.shape[:2]
    
    # Prepare for IRP
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device) / 255.0
    
    print("Running IRP refinement...")
    
    # Process through IRP
    refined, telemetry = vision_guided.refine(image_tensor, early_stop=True)
    
    print(f"  Iterations: {telemetry['iterations']}")
    print(f"  Compute saved: {telemetry['compute_saved']*100:.1f}%")
    
    # Extract attention from VAE latent
    attention_map = None
    if hasattr(vision_irp, 'vae'):
        vae = vision_irp.vae
        with torch.no_grad():
            # Get latent representation
            mu, log_var = vae.encode(image_tensor)
            
            # Create attention from latent activation
            attention = torch.mean(torch.abs(mu), dim=1, keepdim=True)
            
            # Upsample to image size
            attention = torch.nn.functional.interpolate(
                attention,
                size=(original_shape[0], original_shape[1]),
                mode='bilinear',
                align_corners=False
            )
            
            # Normalize
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            attention_map = attention.squeeze().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Refined image
    refined_img = refined.squeeze().permute(1, 2, 0).cpu().numpy()
    refined_img = np.clip(refined_img, 0, 1)
    axes[0, 1].imshow(refined_img)
    axes[0, 1].set_title(f'Refined (Iterations: {telemetry["iterations"]})')
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(image_resized/255.0 - refined_img)
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title('Refinement Difference')
    axes[0, 2].axis('off')
    
    # Attention map
    if attention_map is not None:
        im = axes[1, 0].imshow(attention_map, cmap='hot')
        axes[1, 0].set_title('Attention Map')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Overlay attention on image
        heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_rgb, 0.7, heatmap_rgb, 0.3, 0)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Attention Overlay')
        axes[1, 1].axis('off')
    
    # Energy trajectory
    if 'energy_trajectory' in telemetry and telemetry['energy_trajectory']:
        axes[1, 2].plot(telemetry['energy_trajectory'], 'b-', linewidth=2)
        axes[1, 2].set_title('Energy Convergence')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Energy')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No energy data', ha='center', va='center')
        axes[1, 2].set_title('Energy Convergence')
    
    # Add telemetry text
    telemetry_text = f"""
    Device: {device}
    Iterations: {telemetry.get('iterations', 'N/A')}
    Compute Saved: {telemetry.get('compute_saved', 0)*100:.1f}%
    Trust Score: {telemetry.get('trust', 0):.3f}
    Converged: {telemetry.get('converged', False)}
    """
    
    fig.text(0.02, 0.02, telemetry_text, fontsize=10, family='monospace')
    
    plt.suptitle('SAGE Vision IRP - Attention Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    
    # Also save individual attention map
    if attention_map is not None:
        attention_path = save_path.replace('.png', '_attention.png')
        cv2.imwrite(attention_path, (attention_map * 255).astype(np.uint8))
        print(f"Saved attention map to: {attention_path}")
    
    plt.show()
    
    return attention_map, telemetry


def test_with_sample_images():
    """Test with sample images"""
    
    # Create test directory
    os.makedirs("visual_monitor/test_outputs", exist_ok=True)
    
    # Test with different image sources
    test_images = [
        # Try to use existing test images
        "/home/sprout/ai-workspace/HRM/data/test_image.jpg",
        "/home/sprout/ai-workspace/HRM/assets/test.png",
        # Or create a simple test image
        None
    ]
    
    # If no test images exist, create one
    for img_path in test_images:
        if img_path and os.path.exists(img_path):
            output_path = f"visual_monitor/test_outputs/{Path(img_path).stem}_attention.png"
            visualize_attention(img_path, output_path)
            break
    else:
        # Create a simple test image
        print("Creating synthetic test image...")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some features
        cv2.circle(test_img, (320, 240), 100, (255, 0, 0), -1)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.circle(test_img, (500, 150), 50, (0, 0, 255), -1)
        
        # Save and process
        test_path = "visual_monitor/test_outputs/synthetic_test.jpg"
        cv2.imwrite(test_path, test_img)
        
        output_path = "visual_monitor/test_outputs/synthetic_attention.png"
        visualize_attention(test_path, output_path)


def main():
    """Main entry point"""
    print("=" * 60)
    print("SAGE Attention Visualization Test")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Visualize attention for images')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, default='attention_visualization.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    if args.image:
        visualize_attention(args.image, args.output)
    else:
        test_with_sample_images()
        

if __name__ == "__main__":
    main()