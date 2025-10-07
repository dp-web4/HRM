#!/usr/bin/env python3
"""
GR00T Data Processor Agent
Processes GR00T demo episodes for knowledge distillation into SAGE
"""

import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from dataclasses import dataclass, asdict
import cv2
import time


# Add GR00T to path
GROOT_PATH = Path("/home/dp/ai-workspace/isaac-gr00t")
sys.path.insert(0, str(GROOT_PATH))


@dataclass
class Episode:
    """Container for a single demo episode"""
    episode_id: int
    video_path: str
    actions: List[np.ndarray]
    states: List[np.ndarray]
    rewards: List[float]
    frames: Optional[List[np.ndarray]] = None
    metadata: Optional[Dict] = None
    
    @property
    def length(self) -> int:
        return len(self.actions)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        data['actions'] = [a.tolist() for a in self.actions]
        data['states'] = [s.tolist() for s in self.states]
        if self.frames:
            data['frames'] = None  # Don't serialize frames (too large)
        return data


class GR00TDataProcessor:
    """
    Processes GR00T demonstration episodes
    Extracts vision features, actions, and states for distillation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Data paths
        self.data_root = Path(self.config.get("data_path", 
                                             "/home/dp/ai-workspace/isaac-gr00t/demo_data"))
        self.episode_count = self.config.get("episode_count", 5)
        self.batch_size = self.config.get("batch_size", 16)
        
        # Processing parameters
        self.frame_skip = self.config.get("frame_skip", 1)
        self.resize_shape = self.config.get("resize_shape", (224, 224))
        self.normalize = self.config.get("normalize", True)
        
        # Cache
        self.episodes: List[Episode] = []
        self.processed_features: Dict[int, np.ndarray] = {}
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"📦 GR00T Data Processor initialized")
        print(f"   Data root: {self.data_root}")
        print(f"   Episodes to load: {self.episode_count}")
        print(f"   Device: {self.device}")
    
    def load_episodes(self) -> List[Episode]:
        """Load demo episodes from disk"""
        print(f"\n📂 Loading {self.episode_count} episodes from {self.data_root}")
        
        self.episodes = []
        
        for episode_id in range(self.episode_count):
            episode_dir = self.data_root / f"episode_{episode_id}"
            
            # Check if episode exists (could be real or mock)
            if episode_dir.exists():
                episode = self._load_real_episode(episode_dir, episode_id)
            else:
                # Create mock episode for testing
                episode = self._create_mock_episode(episode_id)
            
            self.episodes.append(episode)
            print(f"  ✅ Loaded episode {episode_id}: {episode.length} steps")
        
        return self.episodes
    
    def _load_real_episode(self, episode_dir: Path, episode_id: int) -> Episode:
        """Load a real GR00T demo episode"""
        
        # Load episode data
        video_path = episode_dir / "rgb.mp4"
        actions_path = episode_dir / "actions.npy"
        states_path = episode_dir / "states.npy"
        metadata_path = episode_dir / "metadata.json"
        
        # Default mock data if files don't exist
        if video_path.exists():
            video_path = str(video_path)
        else:
            video_path = "mock_video.mp4"
        
        if actions_path.exists():
            actions = np.load(actions_path)
            actions = [actions[i] for i in range(len(actions))]
        else:
            # Mock actions (pick and place)
            actions = [np.random.randn(7) for _ in range(100)]
        
        if states_path.exists():
            states = np.load(states_path)
            states = [states[i] for i in range(len(states))]
        else:
            # Mock states
            states = [np.random.randn(14) for _ in range(100)]
        
        # Mock rewards (success at end)
        rewards = [0.0] * (len(actions) - 1) + [1.0]
        
        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return Episode(
            episode_id=episode_id,
            video_path=video_path,
            actions=actions,
            states=states,
            rewards=rewards,
            metadata=metadata
        )
    
    def _create_mock_episode(self, episode_id: int) -> Episode:
        """Create a mock episode for testing"""
        
        # Generate mock pick-and-place trajectory
        num_steps = 100
        
        # Mock actions (7-DOF joint angles)
        actions = []
        for t in range(num_steps):
            # Simulate reaching, grasping, lifting, placing
            phase = t / num_steps
            if phase < 0.25:  # Reach
                action = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
            elif phase < 0.35:  # Grasp
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            elif phase < 0.65:  # Lift and move
                action = np.array([-0.1, 0.2, -0.3, 0.0, 0.0, 0.0, 1.0])
            elif phase < 0.75:  # Lower
                action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
            else:  # Release
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            actions.append(action + np.random.randn(7) * 0.01)
        
        # Mock states (14-DOF: 7 joint positions + 7 joint velocities)
        states = []
        joint_pos = np.zeros(7)
        for action in actions:
            joint_pos += action[:7] * 0.01  # Integrate actions
            joint_vel = action[:7]
            state = np.concatenate([joint_pos, joint_vel])
            states.append(state)
        
        # Rewards (sparse: 1 at success)
        rewards = [0.0] * (num_steps - 1) + [1.0]
        
        return Episode(
            episode_id=episode_id,
            video_path=f"mock_episode_{episode_id}.mp4",
            actions=actions,
            states=states,
            rewards=rewards,
            metadata={"type": "mock", "task": "pick_and_place"}
        )
    
    def extract_frames(self, episode: Episode, skip: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from episode video"""
        
        if skip is None:
            skip = self.frame_skip
        
        frames = []
        
        # Check if video file exists
        if os.path.exists(episode.video_path):
            cap = cv2.VideoCapture(episode.video_path)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip == 0:
                    # Resize frame
                    frame = cv2.resize(frame, self.resize_shape)
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            print(f"  Extracted {len(frames)} frames from {episode.video_path}")
        else:
            # Generate mock frames
            num_frames = len(episode.actions) // skip
            for i in range(num_frames):
                # Create synthetic frame
                frame = np.random.randint(0, 255, 
                                        (*self.resize_shape, 3), dtype=np.uint8)
                frames.append(frame)
            print(f"  Generated {len(frames)} mock frames")
        
        return frames
    
    def process_episode(self, episode: Episode) -> Dict[str, np.ndarray]:
        """Process a single episode into features"""
        
        print(f"\n🔄 Processing episode {episode.episode_id}")
        
        # Extract frames if not already done
        if episode.frames is None:
            episode.frames = self.extract_frames(episode)
        
        # Convert to tensors
        frames_tensor = torch.stack([
            torch.from_numpy(f).float() / 255.0 
            for f in episode.frames
        ]).to(self.device)
        
        actions_tensor = torch.stack([
            torch.from_numpy(a).float() 
            for a in episode.actions[:len(episode.frames)]
        ]).to(self.device)
        
        states_tensor = torch.stack([
            torch.from_numpy(s).float() 
            for s in episode.states[:len(episode.frames)]
        ]).to(self.device)
        
        # Process in batches
        visual_features = []
        for i in range(0, len(frames_tensor), self.batch_size):
            batch = frames_tensor[i:i+self.batch_size]
            
            # Extract features (mock for now)
            with torch.no_grad():
                # In real implementation, would use Eagle model
                features = torch.randn(batch.shape[0], 1536).to(self.device)
                visual_features.append(features)
        
        visual_features = torch.cat(visual_features, dim=0)
        
        # Package processed data
        processed = {
            "visual_features": visual_features.cpu().numpy(),
            "actions": actions_tensor.cpu().numpy(),
            "states": states_tensor.cpu().numpy(),
            "rewards": np.array(episode.rewards[:len(episode.frames)]),
            "episode_id": episode.episode_id
        }
        
        # Cache processed features
        self.processed_features[episode.episode_id] = processed
        
        print(f"  ✅ Processed {len(visual_features)} frames")
        return processed
    
    def process_all_episodes(self) -> List[Dict[str, np.ndarray]]:
        """Process all loaded episodes"""
        
        if not self.episodes:
            self.load_episodes()
        
        print(f"\n🚀 Processing {len(self.episodes)} episodes...")
        
        all_processed = []
        for episode in self.episodes:
            processed = self.process_episode(episode)
            all_processed.append(processed)
        
        print(f"\n✅ Processed {len(all_processed)} episodes")
        return all_processed
    
    def create_training_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Create a training batch from processed episodes"""
        
        if batch_size is None:
            batch_size = self.batch_size
        
        if not self.processed_features:
            self.process_all_episodes()
        
        # Collect all data
        all_visual = []
        all_actions = []
        all_states = []
        all_rewards = []
        
        for episode_data in self.processed_features.values():
            all_visual.append(episode_data["visual_features"])
            all_actions.append(episode_data["actions"])
            all_states.append(episode_data["states"])
            all_rewards.append(episode_data["rewards"])
        
        # Concatenate
        all_visual = np.concatenate(all_visual, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_states = np.concatenate(all_states, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        
        # Random sample
        num_samples = min(batch_size, len(all_visual))
        indices = np.random.choice(len(all_visual), num_samples, replace=False)
        
        batch = {
            "visual_features": torch.from_numpy(all_visual[indices]).float(),
            "actions": torch.from_numpy(all_actions[indices]).float(),
            "states": torch.from_numpy(all_states[indices]).float(),
            "rewards": torch.from_numpy(all_rewards[indices]).float()
        }
        
        return batch
    
    def save_processed_data(self, output_path: str):
        """Save processed episodes to disk"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for efficiency
        with open(output_path, 'wb') as f:
            pickle.dump(self.processed_features, f)
        
        print(f"💾 Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: str):
        """Load processed episodes from disk"""
        
        with open(input_path, 'rb') as f:
            self.processed_features = pickle.load(f)
        
        print(f"📂 Loaded processed data from {input_path}")
        print(f"   Episodes: {list(self.processed_features.keys())}")


def main():
    """Test the GR00T Data Processor"""
    print("🧪 Testing GR00T Data Processor")
    print("=" * 50)
    
    # Create processor
    config = {
        "data_path": "/home/dp/ai-workspace/isaac-gr00t/demo_data",
        "episode_count": 3,
        "batch_size": 8,
        "frame_skip": 5,
        "resize_shape": (224, 224)
    }
    
    processor = GR00TDataProcessor(config)
    
    # Load episodes
    episodes = processor.load_episodes()
    print(f"\n📊 Loaded {len(episodes)} episodes")
    for ep in episodes:
        print(f"  Episode {ep.episode_id}: {ep.length} steps, "
              f"reward sum: {sum(ep.rewards):.1f}")
    
    # Process all episodes
    processed = processor.process_all_episodes()
    
    # Create training batch
    print("\n🎲 Creating training batch...")
    batch = processor.create_training_batch(batch_size=4)
    print(f"  Visual features: {batch['visual_features'].shape}")
    print(f"  Actions: {batch['actions'].shape}")
    print(f"  States: {batch['states'].shape}")
    print(f"  Rewards: {batch['rewards'].shape}")
    
    # Save processed data
    save_path = Path("/home/dp/ai-workspace/HRM/sage/orchestration/data/processed_groot_episodes.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_processed_data(save_path)
    
    # Test loading
    processor2 = GR00TDataProcessor(config)
    processor2.load_processed_data(save_path)
    print(f"\n✅ Successfully loaded {len(processor2.processed_features)} processed episodes")
    
    print("\n✅ GR00T Data Processor test complete!")


if __name__ == "__main__":
    main()