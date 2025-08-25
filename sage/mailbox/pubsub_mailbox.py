#!/usr/bin/env python3
"""
Publish/Subscribe GPU Mailbox System
Extends GPU mailboxes with topic-based routing for IRP plugin communication
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
import queue
import time


@dataclass
class Message:
    """Message structure for mailbox communication"""
    topic: str
    sender_id: str
    data: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float


class PubSubMailbox:
    """
    GPU-accelerated publish/subscribe mailbox for IRP plugins
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Topic subscriptions
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> list of subscriber IDs
        self.subscribers: Dict[str, Callable] = {}  # subscriber ID -> callback function
        
        # Message queues per topic
        self.topic_queues: Dict[str, queue.Queue] = {}
        
        # GPU memory pool for zero-copy transfers
        self.tensor_pool: Dict[str, torch.Tensor] = {}
        
        # Threading
        self.lock = threading.Lock()
        self.running = True
        self.dispatch_thread = threading.Thread(target=self._dispatch_loop)
        self.dispatch_thread.daemon = True
        self.dispatch_thread.start()
        
    def subscribe(self, topic: str, subscriber_id: str, callback: Callable[[Message], None]):
        """
        Subscribe to a topic
        
        Args:
            topic: Topic name (e.g., "camera/left", "camera/right", "attention/map")
            subscriber_id: Unique subscriber identifier
            callback: Function to call when message received
        """
        with self.lock:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = []
                self.topic_queues[topic] = queue.Queue()
                
            if subscriber_id not in self.subscriptions[topic]:
                self.subscriptions[topic].append(subscriber_id)
                
            self.subscribers[subscriber_id] = callback
            
    def unsubscribe(self, topic: str, subscriber_id: str):
        """Unsubscribe from a topic"""
        with self.lock:
            if topic in self.subscriptions:
                if subscriber_id in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(subscriber_id)
                    
    def publish(self, topic: str, sender_id: str, data: Any, metadata: Optional[Dict] = None):
        """
        Publish message to a topic
        
        Args:
            topic: Topic to publish to
            sender_id: ID of the sender
            data: Data to send (numpy array or torch tensor)
            metadata: Optional metadata dict
        """
        # Convert data to tensor if needed
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(self.device)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(self.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        # Create message
        message = Message(
            topic=topic,
            sender_id=sender_id,
            data=tensor,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Queue for dispatch
        if topic in self.topic_queues:
            try:
                self.topic_queues[topic].put_nowait(message)
            except queue.Full:
                # Drop oldest message if queue full
                try:
                    self.topic_queues[topic].get_nowait()
                    self.topic_queues[topic].put_nowait(message)
                except queue.Empty:
                    pass
                    
    def _dispatch_loop(self):
        """Background thread for message dispatch"""
        while self.running:
            # Check all topic queues
            for topic, msg_queue in list(self.topic_queues.items()):
                try:
                    message = msg_queue.get(timeout=0.01)
                    self._dispatch_message(topic, message)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Dispatch error on topic {topic}: {e}")
                    
    def _dispatch_message(self, topic: str, message: Message):
        """Dispatch message to all subscribers"""
        with self.lock:
            if topic not in self.subscriptions:
                return
                
            for subscriber_id in self.subscriptions[topic]:
                if subscriber_id in self.subscribers:
                    try:
                        # Call subscriber callback
                        self.subscribers[subscriber_id](message)
                    except Exception as e:
                        print(f"Subscriber {subscriber_id} error: {e}")
                        
    def allocate_tensor(self, key: str, shape: tuple, dtype: torch.dtype = torch.float32):
        """
        Pre-allocate GPU tensor for zero-copy transfers
        
        Args:
            key: Identifier for the tensor
            shape: Shape of the tensor
            dtype: Data type
        """
        self.tensor_pool[key] = torch.zeros(shape, dtype=dtype, device=self.device)
        
    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        """Get pre-allocated tensor"""
        return self.tensor_pool.get(key)
        
    def shutdown(self):
        """Shutdown mailbox system"""
        self.running = False
        if self.dispatch_thread:
            self.dispatch_thread.join(timeout=2.0)
            

class GPUMailboxBridge:
    """
    Bridge between IRP plugins using GPU mailboxes
    """
    
    def __init__(self, mailbox: PubSubMailbox):
        self.mailbox = mailbox
        self.plugin_outputs: Dict[str, torch.Tensor] = {}
        
    def register_plugin_output(self, plugin_id: str, topic: str):
        """Register a plugin's output topic"""
        # Subscribe to store latest output
        self.mailbox.subscribe(
            topic,
            f"bridge_{plugin_id}",
            lambda msg: self._store_output(plugin_id, msg)
        )
        
    def _store_output(self, plugin_id: str, message: Message):
        """Store plugin output for later retrieval"""
        self.plugin_outputs[plugin_id] = message.data
        
    def get_plugin_output(self, plugin_id: str) -> Optional[torch.Tensor]:
        """Get latest output from a plugin"""
        return self.plugin_outputs.get(plugin_id)
        
    def connect_plugins(self, source_id: str, source_topic: str, 
                        target_id: str, target_callback: Callable):
        """
        Connect output of one plugin to input of another
        
        Args:
            source_id: Source plugin ID
            source_topic: Topic the source publishes to
            target_id: Target plugin ID
            target_callback: Function to call with the data
        """
        self.mailbox.subscribe(
            source_topic,
            f"bridge_{source_id}_to_{target_id}",
            lambda msg: target_callback(msg.data)
        )


# Test the mailbox system
def test_pubsub_mailbox():
    """Test the publish/subscribe mailbox"""
    print("Testing PubSub Mailbox System...")
    
    mailbox = PubSubMailbox()
    
    # Test data collection
    received_messages = []
    
    def camera_callback(msg: Message):
        print(f"Camera received: {msg.topic} from {msg.sender_id}")
        received_messages.append(msg)
        
    def attention_callback(msg: Message):
        print(f"Attention received: {msg.topic}, shape: {msg.data.shape}")
        received_messages.append(msg)
        
    # Subscribe to topics
    mailbox.subscribe("camera/left", "vision_processor", camera_callback)
    mailbox.subscribe("camera/right", "vision_processor", camera_callback)
    mailbox.subscribe("attention/map", "monitor", attention_callback)
    
    # Simulate camera data
    left_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Publish camera frames
    mailbox.publish("camera/left", "left_camera_sensor", left_frame)
    mailbox.publish("camera/right", "right_camera_sensor", right_frame)
    
    # Simulate attention map
    attention = torch.randn(1, 1, 224, 224)
    mailbox.publish("attention/map", "vision_irp", attention)
    
    # Wait for messages
    time.sleep(0.1)
    
    print(f"\nReceived {len(received_messages)} messages")
    for msg in received_messages:
        print(f"  - {msg.topic}: {msg.data.shape}")
        
    # Test bridge
    bridge = GPUMailboxBridge(mailbox)
    bridge.register_plugin_output("vision", "attention/map")
    
    # Publish another attention map
    attention2 = torch.randn(1, 1, 224, 224)
    mailbox.publish("attention/map", "vision_irp", attention2)
    
    time.sleep(0.1)
    
    # Retrieve from bridge
    output = bridge.get_plugin_output("vision")
    if output is not None:
        print(f"\nBridge stored output shape: {output.shape}")
        
    # Cleanup
    mailbox.shutdown()
    print("\nTest complete!")
    

if __name__ == "__main__":
    test_pubsub_mailbox()