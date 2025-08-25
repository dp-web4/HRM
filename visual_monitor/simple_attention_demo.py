#!/usr/bin/env python3
"""
Simple Attention Demo without external dependencies
Demonstrates the attention concept using motion simulation
"""

import random
import time
import sys
import math

class SimpleAttentionDemo:
    """
    Text-based attention demonstration
    Simulates motion detection and attention focusing
    """
    
    def __init__(self):
        self.width = 80
        self.height = 24
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Moving object
        self.obj_x = self.width // 2
        self.obj_y = self.height // 2
        self.obj_vx = random.choice([-1, 1])
        self.obj_vy = random.choice([-1, 1])
        
        # Attention region
        self.attention_x = 0
        self.attention_y = 0
        self.attention_size = 10
        self.attention_active = False
        
        # Motion history
        self.motion_history = []
        self.max_history = 5
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        
    def clear_screen(self):
        """Clear the console screen"""
        print('\033[2J\033[H', end='')
    
    def update_object(self):
        """Update moving object position"""
        # Store previous position for motion detection
        prev_x, prev_y = self.obj_x, self.obj_y
        
        # Update position
        self.obj_x += self.obj_vx
        self.obj_y += self.obj_vy
        
        # Bounce off walls
        if self.obj_x <= 0 or self.obj_x >= self.width - 1:
            self.obj_vx = -self.obj_vx
            self.obj_x = max(0, min(self.width - 1, self.obj_x))
        
        if self.obj_y <= 0 or self.obj_y >= self.height - 1:
            self.obj_vy = -self.obj_vy
            self.obj_y = max(0, min(self.height - 1, self.obj_y))
        
        # Randomly change direction occasionally
        if random.random() < 0.05:
            self.obj_vx = random.choice([-2, -1, 1, 2])
            self.obj_vy = random.choice([-2, -1, 1, 2])
        
        # Detect motion
        motion_detected = (abs(self.obj_x - prev_x) > 0 or abs(self.obj_y - prev_y) > 0)
        
        if motion_detected:
            self.motion_history.append((self.obj_x, self.obj_y))
            if len(self.motion_history) > self.max_history:
                self.motion_history.pop(0)
            
            # Update attention region
            self.update_attention()
    
    def update_attention(self):
        """Update attention region based on motion"""
        if self.motion_history:
            # Calculate center of recent motion
            avg_x = sum(x for x, y in self.motion_history) / len(self.motion_history)
            avg_y = sum(y for x, y in self.motion_history) / len(self.motion_history)
            
            # Smooth transition
            self.attention_x = int(avg_x)
            self.attention_y = int(avg_y)
            self.attention_active = True
            
            # Adaptive attention size based on motion variance
            if len(self.motion_history) > 1:
                var_x = sum((x - avg_x)**2 for x, y in self.motion_history)
                var_y = sum((y - avg_y)**2 for x, y in self.motion_history)
                variance = math.sqrt(var_x + var_y) / len(self.motion_history)
                self.attention_size = min(20, max(8, int(8 + variance * 2)))
    
    def render(self):
        """Render the current frame"""
        # Clear grid
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw attention region (if active)
        if self.attention_active:
            for y in range(max(0, self.attention_y - self.attention_size // 2),
                          min(self.height, self.attention_y + self.attention_size // 2)):
                for x in range(max(0, self.attention_x - self.attention_size // 2),
                              min(self.width, self.attention_x + self.attention_size // 2)):
                    # Draw attention boundary
                    if (y == self.attention_y - self.attention_size // 2 or 
                        y == self.attention_y + self.attention_size // 2 - 1):
                        if 0 <= x < self.width and 0 <= y < self.height:
                            self.grid[y][x] = '-'
                    elif (x == self.attention_x - self.attention_size // 2 or 
                          x == self.attention_x + self.attention_size // 2 - 1):
                        if 0 <= x < self.width and 0 <= y < self.height:
                            self.grid[y][x] = '|'
            
            # Draw attention corners
            corners = [
                (self.attention_x - self.attention_size // 2, 
                 self.attention_y - self.attention_size // 2),
                (self.attention_x + self.attention_size // 2 - 1, 
                 self.attention_y - self.attention_size // 2),
                (self.attention_x - self.attention_size // 2, 
                 self.attention_y + self.attention_size // 2 - 1),
                (self.attention_x + self.attention_size // 2 - 1, 
                 self.attention_y + self.attention_size // 2 - 1),
            ]
            for cx, cy in corners:
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self.grid[cy][cx] = '+'
        
        # Draw motion trail
        for i, (mx, my) in enumerate(self.motion_history):
            if 0 <= mx < self.width and 0 <= my < self.height:
                self.grid[my][mx] = '.' if i < len(self.motion_history) - 1 else 'o'
        
        # Draw object
        if 0 <= self.obj_x < self.width and 0 <= self.obj_y < self.height:
            self.grid[self.obj_y][self.obj_x] = '@'
        
        # Display
        self.clear_screen()
        
        # Header
        print("=" * self.width)
        print("SIMPLE ATTENTION DEMO - Motion-Based Focus")
        print("=" * self.width)
        
        # Grid
        for row in self.grid:
            print(''.join(row))
        
        # Footer info
        print("=" * self.width)
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Simulated metrics (what would come from IRP)
        trust_score = 0.7 + 0.3 * (1.0 - len(self.motion_history) / self.max_history)
        iterations = 2 if self.attention_active else 50
        
        print(f"FPS: {fps:.1f} | Attention: {'ACTIVE' if self.attention_active else 'IDLE'}")
        print(f"Position: ({self.obj_x:2d}, {self.obj_y:2d}) | Velocity: ({self.obj_vx:+2d}, {self.obj_vy:+2d})")
        print(f"Attention Size: {self.attention_size} | Motion Points: {len(self.motion_history)}")
        print(f"[Simulated IRP] Trust: {trust_score:.2f} | Iterations: {iterations}")
        print()
        print("Legend: @ = Object | o = Recent position | . = Motion trail | +--+ = Attention box")
        print("Press Ctrl+C to exit")
    
    def run(self):
        """Main loop"""
        print("Starting Simple Attention Demo...")
        print("This demonstrates motion-based attention without requiring OpenCV or PyTorch")
        print()
        time.sleep(1)
        
        max_frames = 20  # Run for 20 frames only
        
        try:
            while self.frame_count < max_frames:
                self.update_object()
                self.render()
                self.frame_count += 1
                time.sleep(0.1)  # 10 FPS for terminal display
                
        except KeyboardInterrupt:
            print("\n\nDemo terminated.")
            print(f"Total frames: {self.frame_count}")
            print(f"Average FPS: {self.frame_count / (time.time() - self.start_time):.1f}")


def main():
    """Run the simple attention demo"""
    print("\n" + "=" * 80)
    print("ATTENTION CONCEPT DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demo shows how attention works in the HRM/SAGE system:")
    print("1. Motion Detection: The system tracks moving objects")
    print("2. Attention Focus: An attention box forms around areas of motion")
    print("3. Adaptive Sizing: The attention region adapts to motion patterns")
    print("4. IRP Integration: In the full system, this region would be processed by IRP")
    print()
    print("In the actual implementation with camera and GPU:")
    print("- Real camera frames replace the simulated object")
    print("- GPU-accelerated IRP processes attention regions")
    print("- TinyVAE compresses 64x64 crops to 16D latents")
    print("- Trust scores emerge from convergence stability")
    print()
    print("Starting demo automatically...")
    print()
    
    demo = SimpleAttentionDemo()
    demo.run()


if __name__ == "__main__":
    main()