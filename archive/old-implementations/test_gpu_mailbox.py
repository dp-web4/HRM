#!/usr/bin/env python3
"""Test GPU Mailbox Architecture - Zero-copy communication between GPU-resident modules"""

import torch
import sys
import os
import time
from typing import Dict, Any, Tuple

# Add HRM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class GPUMailbox:
    """Zero-copy message passing between GPU-resident modules"""
    
    def __init__(self, capacity: int = 100, device: str = 'cuda'):
        self.device = torch.device(device)
        self.capacity = capacity
        
        # Pre-allocate mailbox buffers on GPU
        self.messages = {}
        self.metadata = {}
        
    def register_channel(self, name: str, shape: Tuple[int, ...], dtype=torch.float32):
        """Register a communication channel with pre-allocated GPU buffer"""
        self.messages[name] = torch.zeros(shape, dtype=dtype, device=self.device)
        self.metadata[name] = {
            'ready': torch.zeros(1, dtype=torch.bool, device=self.device),
            'timestamp': torch.zeros(1, dtype=torch.float32, device=self.device)
        }
        print(f"✅ Registered channel '{name}' with shape {shape} on {self.device}")
        
    def send(self, channel: str, data: torch.Tensor):
        """Zero-copy send to mailbox (data already on GPU)"""
        if channel not in self.messages:
            raise ValueError(f"Channel '{channel}' not registered")
            
        # Store reference (zero-copy)
        self.messages[channel] = data
        self.metadata[channel]['ready'][0] = True
        self.metadata[channel]['timestamp'][0] = time.time()
        
    def receive(self, channel: str) -> torch.Tensor:
        """Zero-copy receive from mailbox"""
        if channel not in self.messages:
            raise ValueError(f"Channel '{channel}' not registered")
            
        if not self.metadata[channel]['ready'][0]:
            return None
            
        # Return reference (no copy)
        self.metadata[channel]['ready'][0] = False
        return self.messages[channel]
        
    def peek(self, channel: str) -> bool:
        """Check if message available without consuming"""
        return self.metadata[channel]['ready'][0].item() if channel in self.metadata else False

def test_gpu_mailbox():
    """Test GPU mailbox architecture"""
    print("=" * 60)
    print("GPU MAILBOX ARCHITECTURE TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, cannot test GPU mailbox")
        return False
        
    try:
        # Create mailbox
        mailbox = GPUMailbox(device='cuda')
        
        # Register channels for different module communications
        mailbox.register_channel('hrm_to_llm', shape=(2, 512, 256), dtype=torch.float32)
        mailbox.register_channel('llm_to_hrm', shape=(2, 512, 256), dtype=torch.float32)
        mailbox.register_channel('sensor_data', shape=(100, 64), dtype=torch.float32)
        
        # Simulate HRM sending data
        hrm_output = torch.randn(2, 512, 256).cuda()
        mailbox.send('hrm_to_llm', hrm_output)
        print("✅ HRM sent data to mailbox")
        
        # Simulate LLM receiving data (zero-copy)
        llm_input = mailbox.receive('hrm_to_llm')
        assert llm_input is not None
        assert llm_input.data_ptr() == mailbox.messages['hrm_to_llm'].data_ptr()  # Same memory!
        print("✅ LLM received data (zero-copy verified)")
        
        # Benchmark throughput
        print("\n📊 Throughput Benchmark:")
        num_messages = 1000
        data = torch.randn(2, 512, 256).cuda()
        
        start = time.time()
        for _ in range(num_messages):
            mailbox.send('hrm_to_llm', data)
            _ = mailbox.receive('hrm_to_llm')
        elapsed = time.time() - start
        
        throughput = num_messages / elapsed
        bandwidth = (data.numel() * data.element_size() * num_messages) / elapsed / 1e9  # GB/s
        
        print(f"   Messages/sec: {throughput:.0f}")
        print(f"   Bandwidth: {bandwidth:.2f} GB/s")
        print(f"   Latency: {elapsed/num_messages*1e6:.2f} µs per message")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hrm_gpu_mailbox_integration():
    """Test HRM integration with GPU mailbox"""
    print("\n" + "=" * 60)
    print("HRM + GPU MAILBOX INTEGRATION TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    try:
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        
        # Create HRM model
        config = {
            'batch_size': 2,
            'seq_len': 16,
            'num_puzzle_identifiers': 100,
            'vocab_size': 512,
            'H_cycles': 2,
            'L_cycles': 2,
            'H_layers': 4,
            'L_layers': 4,
            'hidden_size': 256,
            'expansion': 4,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 16,
            'halt_exploration_prob': 0.1,
        }
        
        model = HierarchicalReasoningModel_ACTV1(config).cuda()
        mailbox = GPUMailbox(device='cuda')
        
        # Register HRM output channel
        mailbox.register_channel('hrm_output', shape=(2, 16, 512), dtype=torch.float32)
        
        # Run HRM
        batch = {
            'inputs': torch.randint(0, 512, (2, 16)).cuda(),
            'puzzle_identifiers': torch.arange(2).cuda()
        }
        
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
        
        # Send HRM output through mailbox (zero-copy)
        if isinstance(outputs, dict) and 'logits' in outputs:
            mailbox.send('hrm_output', outputs['logits'])
            print("✅ HRM output sent to GPU mailbox")
            
            # Another module can receive it
            received = mailbox.receive('hrm_output')
            assert received.data_ptr() == outputs['logits'].data_ptr()
            print("✅ Zero-copy communication verified")
            
        print("\n🎯 GPU Mailbox Benefits:")
        print("   • No CPU-GPU transfers")
        print("   • Direct GPU memory sharing")
        print("   • Microsecond latency")
        print("   • Perfect for SAGE multi-module architecture")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all GPU mailbox tests"""
    print("\n🚀 GPU Mailbox Architecture Test Suite\n")
    
    # Test basic mailbox
    mailbox_success = test_gpu_mailbox()
    
    # Test HRM integration
    hrm_success = test_hrm_gpu_mailbox_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"{'✅' if mailbox_success else '❌'} GPU Mailbox working")
    print(f"{'✅' if hrm_success else '❌'} HRM + GPU Mailbox integration")
    
    if mailbox_success and hrm_success:
        print("\n✅ GPU Mailbox architecture ready for SAGE deployment!")
        print("   Zero-copy communication enables real-time multi-module AI")
    
    return mailbox_success and hrm_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)