#!/usr/bin/env python3
"""
Test HRM model with vision attention system on Jetson
Combines trained HRM model with real-time camera input
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from pathlib import Path
import sys

# Add HRM to path
sys.path.append(str(Path(__file__).parent))

from sage.irp.plugins.vision_attention_plugin import VisionAttentionPlugin, AttentionStrategy
from models.hrm import HRM

class HRMVisionSystem:
    def __init__(self, checkpoint_path, camera_id=0, device=None):
        """
        Initialize HRM with vision attention
        
        Args:
            checkpoint_path: Path to trained HRM checkpoint
            camera_id: Camera index
            device: torch device
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load HRM model
        self.load_hrm(checkpoint_path)
        
        # Initialize vision plugin
        self.vision_plugin = VisionAttentionPlugin(
            sensor_id=camera_id,
            strategy=AttentionStrategy.DYNAMIC_BOX,  # Use dynamic tracking
            enable_vae=False,  # Disable VAE for now to test HRM alone
            device=self.device
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Performance tracking
        self.frame_times = []
        self.hrm_times = []
        
    def load_hrm(self, checkpoint_path):
        """Load HRM model from checkpoint"""
        print(f"📦 Loading HRM from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config based on checkpoint inspection
            config = {
                'num_classes': 10,
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 4,
                'dropout': 0.1,
                'max_seq_len': 784,
                'vocab_size': 12
            }
        
        # Create model
        self.hrm = HRM(
            num_classes=config.get('num_classes', 10),
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 784),
            vocab_size=config.get('vocab_size', 12)
        ).to(self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.hrm.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.hrm.load_state_dict(checkpoint)
        
        self.hrm.eval()
        print(f"✅ HRM loaded with {sum(p.numel() for p in self.hrm.parameters()):,} parameters")
        
    def preprocess_for_hrm(self, image_crop):
        """
        Convert image crop to HRM input format
        HRM expects tokenized sequences
        """
        # Resize to 28x28 (like MNIST/ARC tasks)
        resized = cv2.resize(image_crop, (28, 28))
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Quantize to tokens (0-9 for simplicity)
        tokens = (gray / 255.0 * 9).astype(np.int32)
        
        # Flatten to sequence
        sequence = tokens.flatten()
        
        # Convert to tensor
        tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.device)
        
        return tensor
    
    def run_inference(self, frame):
        """Run HRM inference on focused region"""
        start_time = time.time()
        
        # Get focus region from vision attention
        focus_region, telemetry = self.vision_plugin.refine(frame)
        
        if focus_region and focus_region.confidence > 0.3:
            # Extract crop from focus region
            x, y, w, h = focus_region.x, focus_region.y, focus_region.width, focus_region.height
            crop = frame[y:y+h, x:x+w]
            
            # Preprocess for HRM
            hrm_input = self.preprocess_for_hrm(crop)
            
            # Run HRM inference
            hrm_start = time.time()
            with torch.no_grad():
                h_output, l_output = self.hrm(hrm_input, hrm_input)
                
                # Get predictions
                h_pred = torch.argmax(h_output, dim=-1)
                l_pred = torch.argmax(l_output, dim=-1) if l_output is not None else None
                
            hrm_time = time.time() - hrm_start
            self.hrm_times.append(hrm_time)
            
            return {
                'focus_region': focus_region,
                'h_prediction': h_pred.cpu().numpy(),
                'l_prediction': l_pred.cpu().numpy() if l_pred is not None else None,
                'hrm_time': hrm_time,
                'confidence': focus_region.confidence
            }
        
        return None
    
    def draw_results(self, frame, results):
        """Draw inference results on frame"""
        if results:
            region = results['focus_region']
            
            # Draw focus box
            color = (0, 255, 0) if region.confidence > 0.5 else (0, 255, 255)
            cv2.rectangle(frame, 
                         (region.x, region.y), 
                         (region.x + region.width, region.y + region.height),
                         color, 2)
            
            # Draw HRM predictions
            text = f"H: {results['h_prediction'][0][0]}"
            if results['l_prediction'] is not None:
                text += f" L: {results['l_prediction'][0][0]}"
            text += f" ({results['hrm_time']*1000:.1f}ms)"
            
            cv2.putText(frame, text, 
                       (region.x, region.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            conf_text = f"Conf: {region.confidence:.2f}"
            cv2.putText(frame, conf_text,
                       (region.x, region.y + region.height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw FPS
        if len(self.frame_times) > 0:
            fps = 1.0 / np.mean(self.frame_times[-30:])
            cv2.putText(frame, f"FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw HRM inference time
        if len(self.hrm_times) > 0:
            avg_hrm_time = np.mean(self.hrm_times[-30:]) * 1000
            cv2.putText(frame, f"HRM: {avg_hrm_time:.1f}ms",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("🎥 Starting HRM+Vision system...")
        print("Press 'q' to quit, 's' to switch strategy")
        
        frame_count = 0
        last_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Run inference
                results = self.run_inference(frame)
                
                # Draw results
                display_frame = self.draw_results(frame, results)
                
                # Show frame
                cv2.imshow('HRM Vision System', display_frame)
                
                # Track timing
                current_time = time.time()
                self.frame_times.append(current_time - last_time)
                last_time = current_time
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Switch strategy
                    current = self.vision_plugin.current_strategy
                    new_strategy = (AttentionStrategy.FIXED_TILES 
                                  if current == AttentionStrategy.DYNAMIC_BOX 
                                  else AttentionStrategy.DYNAMIC_BOX)
                    self.vision_plugin.switch_strategy(new_strategy)
                    print(f"Switched to {new_strategy.name}")
                
                frame_count += 1
                if frame_count % 100 == 0:
                    if len(self.hrm_times) > 0:
                        print(f"Frame {frame_count}: "
                              f"FPS={1.0/np.mean(self.frame_times[-30:]):.1f}, "
                              f"HRM={np.mean(self.hrm_times[-30:])*1000:.1f}ms")
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            if len(self.hrm_times) > 0:
                print(f"\n📊 Performance Summary:")
                print(f"  Average FPS: {1.0/np.mean(self.frame_times):.1f}")
                print(f"  Average HRM inference: {np.mean(self.hrm_times)*1000:.1f}ms")
                print(f"  Min HRM time: {min(self.hrm_times)*1000:.1f}ms")
                print(f"  Max HRM time: {max(self.hrm_times)*1000:.1f}ms")

def main():
    # Use best checkpoint
    checkpoint_path = Path("checkpoints/hrm_arc_best.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nDownload it with:")
        print("  python3 dropbox/rclone_sync.py download checkpoints/hrm_arc_best.pt")
        return
    
    # Create and run system
    system = HRMVisionSystem(checkpoint_path, camera_id=0)
    system.run()

if __name__ == "__main__":
    main()