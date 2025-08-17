#!/usr/bin/env python3
"""
GR00T Video Capture and Real-Time Processing
============================================
Demonstrates GR00T's vision capabilities with live video input.
Based on NVIDIA's video capture examples for embodied AI.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Tuple, Optional
import threading
import queue
from dataclasses import dataclass
from load_groot_full import GR00TModel, load_groot_model

@dataclass
class VideoFrame:
    """Container for video frame data"""
    image: np.ndarray
    timestamp: float
    frame_id: int
    
class GR00TVideoProcessor:
    """Real-time video processing with GR00T model"""
    
    def __init__(self, model: Optional[GR00TModel] = None, device: str = "cuda"):
        """Initialize video processor
        
        Args:
            model: Pre-loaded GR00T model (will load if None)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model if not provided
        if model is None:
            print("📦 Loading GR00T model...")
            self.model = load_groot_model(self.device)
        else:
            self.model = model.to(self.device)
        
        self.model.eval()
        
        # Video capture
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        
        # Processing stats
        self.fps_tracker = []
        self.processing_times = []
        
        # Control flags
        self.running = False
        self.recording = False
        self.recorded_frames = []
        
    def start_capture(self, source: int = 0):
        """Start video capture from camera or file
        
        Args:
            source: Camera index or video file path
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        print(f"📹 Video capture started from source: {source}")
        self.running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_loop)
        process_thread.daemon = True
        process_thread.start()
        
    def _capture_loop(self):
        """Continuous capture loop"""
        frame_id = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            timestamp = time.time()
            video_frame = VideoFrame(frame, timestamp, frame_id)
            
            # Non-blocking put
            try:
                self.frame_queue.put_nowait(video_frame)
            except queue.Full:
                # Drop old frames if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(video_frame)
                except queue.Empty:
                    pass
                    
            frame_id += 1
            
    def _process_loop(self):
        """Continuous processing loop"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                result = self._process_frame(frame)
                
                # Non-blocking put
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Drop old results if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
                
    def _process_frame(self, frame: VideoFrame) -> Dict:
        """Process single frame through GR00T
        
        Args:
            frame: Video frame to process
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Preprocess image for GR00T
        image_tensor = self._preprocess_image(frame.image)
        
        # Generate mock inputs for demo (in real deployment, these would come from robot/task)
        batch_size = 1
        language_ids = torch.randint(0, 50000, (batch_size, 50)).to(self.device)
        proprio_state = torch.randn(batch_size, 64).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor, language_ids, proprio_state)
            
        # Extract features
        actions = outputs['actions'].cpu().numpy()
        vision_features = outputs['vision_features'].cpu().numpy()
        
        # Calculate processing time
        process_time = time.time() - start_time
        self.processing_times.append(process_time)
        
        # Track FPS
        current_fps = 1.0 / process_time if process_time > 0 else 0
        self.fps_tracker.append(current_fps)
        if len(self.fps_tracker) > 30:
            self.fps_tracker.pop(0)
            
        result = {
            'frame': frame,
            'actions': actions,
            'vision_features': vision_features,
            'process_time': process_time,
            'fps': np.mean(self.fps_tracker)
        }
        
        # Record if enabled
        if self.recording:
            self.recorded_frames.append(result)
            
        return result
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for GR00T vision encoder
        
        Args:
            image: Raw image from camera
            
        Returns:
            Preprocessed tensor
        """
        # Resize to expected input size
        image = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
        
    def visualize_stream(self):
        """Display live video with GR00T processing overlay"""
        print("\n🎮 Video Stream Controls:")
        print("  'q' - Quit")
        print("  'r' - Toggle recording")
        print("  's' - Save snapshot")
        print("  'f' - Toggle FPS display")
        print("  'a' - Toggle action visualization")
        print("  'v' - Toggle vision features")
        
        show_fps = True
        show_actions = True
        show_features = False
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            frame = result['frame'].image.copy()
            
            # Draw overlays
            if show_fps:
                fps_text = f"FPS: {result['fps']:.1f}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
            if show_actions:
                # Visualize predicted actions as bars
                actions = result['actions'][0][:8]  # Show first 8 actions
                for i, action in enumerate(actions):
                    bar_height = int(abs(action) * 50)
                    color = (0, 255, 0) if action > 0 else (0, 0, 255)
                    cv2.rectangle(frame, 
                                 (10 + i * 30, 450 - bar_height),
                                 (30 + i * 30, 450),
                                 color, -1)
                                 
            if show_features:
                # Show vision feature heatmap
                features = result['vision_features'][0][:64].reshape(8, 8)
                features = (features - features.min()) / (features.max() - features.min() + 1e-8)
                heatmap = cv2.applyColorMap((features * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (160, 160))
                frame[10:170, 470:630] = heatmap
                
            if self.recording:
                cv2.circle(frame, (600, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (550, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                           
            # Display frame
            cv2.imshow("GR00T Video Capture", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.recording = not self.recording
                if self.recording:
                    print("🔴 Recording started")
                    self.recorded_frames = []
                else:
                    print(f"⏹️ Recording stopped. {len(self.recorded_frames)} frames captured")
            elif key == ord('s'):
                self._save_snapshot(frame)
            elif key == ord('f'):
                show_fps = not show_fps
            elif key == ord('a'):
                show_actions = not show_actions
            elif key == ord('v'):
                show_features = not show_features
                
    def _save_snapshot(self, frame: np.ndarray):
        """Save current frame as image"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"groot_snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved: {filename}")
        
    def save_recording(self, filename: str = "groot_recording.mp4"):
        """Save recorded frames as video"""
        if not self.recorded_frames:
            print("No frames to save")
            return
            
        # Get video properties from first frame
        height, width = self.recorded_frames[0]['frame'].image.shape[:2]
        fps = 30
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Write frames
        for result in self.recorded_frames:
            out.write(result['frame'].image)
            
        out.release()
        print(f"💾 Recording saved: {filename} ({len(self.recorded_frames)} frames)")
        
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        if not self.processing_times:
            return {}
            
        return {
            'avg_fps': np.mean(self.fps_tracker) if self.fps_tracker else 0,
            'avg_process_time': np.mean(self.processing_times),
            'min_process_time': np.min(self.processing_times),
            'max_process_time': np.max(self.processing_times),
            'frames_processed': len(self.processing_times),
            'frames_recorded': len(self.recorded_frames)
        }
        
    def stop(self):
        """Stop video capture and processing"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = self.get_stats()
        if stats:
            print("\n📊 Final Statistics:")
            print(f"  Average FPS: {stats['avg_fps']:.1f}")
            print(f"  Average process time: {stats['avg_process_time']*1000:.2f} ms")
            print(f"  Total frames processed: {stats['frames_processed']}")
            if stats['frames_recorded'] > 0:
                print(f"  Frames recorded: {stats['frames_recorded']}")
                

def demo_video_tasks():
    """Demonstrate different video processing tasks"""
    
    print("\n" + "="*60)
    print("GR00T Video Processing Demos")
    print("="*60)
    
    processor = GR00TVideoProcessor()
    
    try:
        # Task 1: Live camera feed
        print("\n🎥 Task 1: Live Camera Processing")
        print("-" * 40)
        processor.start_capture(0)  # Use default camera
        processor.visualize_stream()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        processor.stop()
        
        # Save recording if available
        if processor.recorded_frames:
            processor.save_recording()
            

def benchmark_video_performance():
    """Benchmark video processing performance"""
    
    print("\n" + "="*60)
    print("GR00T Video Performance Benchmark")
    print("="*60)
    
    processor = GR00TVideoProcessor()
    
    try:
        processor.start_capture(0)
        
        print("\n⏱️ Running 10-second benchmark...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            time.sleep(0.1)
            
        processor.stop()
        
        # Get statistics
        stats = processor.get_stats()
        
        print("\n📊 Benchmark Results:")
        print(f"  Average FPS: {stats['avg_fps']:.1f}")
        print(f"  Average latency: {stats['avg_process_time']*1000:.2f} ms")
        print(f"  Min latency: {stats['min_process_time']*1000:.2f} ms")
        print(f"  Max latency: {stats['max_process_time']*1000:.2f} ms")
        print(f"  Total frames: {stats['frames_processed']}")
        
        # GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 GPU Memory:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        processor.stop()
        

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_video_performance()
    else:
        demo_video_tasks()