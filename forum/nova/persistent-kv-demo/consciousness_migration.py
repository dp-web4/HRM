#!/usr/bin/env python3
"""
Consciousness Migration Experiment
Demonstrates practical use cases for KV-cache persistence:
1. Pause/Resume mid-conversation
2. Context window management
3. Cross-session memory
"""

import torch
import json
import time
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_kv import kv_to_cpu, save_kv, load_kv, kv_to_device

class ConsciousnessSession:
    """Manages a consciousness session with save/resume capability"""
    
    def __init__(self, session_id, model_name="gpt2"):
        self.session_id = session_id
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Session state
        self.conversation_history = []
        self.current_kv = None
        self.checkpoint_dir = Path(f"consciousness_sessions/{session_id}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def process_input(self, text):
        """Process input and update consciousness state"""
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.current_kv is not None:
                # Continue from existing state
                out = self.model(**inputs, past_key_values=self.current_kv, use_cache=True)
            else:
                # Fresh start
                out = self.model(**inputs, use_cache=True)
            
            self.current_kv = out.past_key_values
            
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": text,
            "kv_seq_len": self.current_kv[0][0].shape[2] if self.current_kv else 0
        })
        
        return out.logits
    
    def generate_response(self, prompt="", max_tokens=50, temperature=0.8):
        """Generate a response from current state"""
        if prompt:
            self.process_input(prompt)
        
        if self.current_kv is None:
            return "No consciousness state available"
        
        generated = []
        past = self.current_kv
        
        # Start with a token
        input_ids = self.tok.encode(" ", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                out = self.model(input_ids=input_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                
                logits = out.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                
                token = self.tok.decode(next_id[0], skip_special_tokens=True)
                generated.append(token)
                
                input_ids = next_id
                
                # Stop at sentence end
                if token in ['.', '!', '?'] and len(generated) > 10:
                    break
        
        self.current_kv = past
        return ''.join(generated)
    
    def save_checkpoint(self, name="checkpoint"):
        """Save current consciousness state"""
        if self.current_kv is None:
            return False
        
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        meta_path = self.checkpoint_dir / f"{name}_meta.json"
        
        # Save KV cache
        kv_cpu = kv_to_cpu(self.current_kv)
        save_kv(str(checkpoint_path), kv_cpu, fmt="torch")
        
        # Save metadata
        meta = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "conversation_length": len(self.conversation_history),
            "kv_shape": str(kv_cpu[0][0].shape),
            "history": self.conversation_history[-5:]  # Last 5 entries
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        return True
    
    def load_checkpoint(self, name="checkpoint"):
        """Load consciousness state from checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        meta_path = self.checkpoint_dir / f"{name}_meta.json"
        
        if not checkpoint_path.exists():
            return False
        
        # Load KV cache
        kv_cpu = load_kv(str(checkpoint_path), fmt="torch")
        self.current_kv = kv_to_device(kv_cpu, self.device)
        
        # Load metadata
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.conversation_history = meta.get("history", [])
        
        return True

def demonstrate_consciousness_migration():
    """Demonstrate practical consciousness migration scenarios"""
    
    print("=" * 70)
    print("CONSCIOUSNESS MIGRATION DEMONSTRATION")
    print("=" * 70)
    
    # Scenario 1: Mid-conversation pause/resume
    print("\n📖 Scenario 1: Mid-Conversation Pause/Resume")
    print("-" * 70)
    
    session1 = ConsciousnessSession("demo_session_1")
    
    # Start a conversation
    print("Starting conversation...")
    session1.process_input("The nature of consciousness in artificial systems")
    session1.process_input(" involves complex interactions between")
    
    response1 = session1.generate_response(max_tokens=20)
    print(f"Initial response: {response1}")
    
    # Save checkpoint
    print("\n💾 Saving consciousness state...")
    session1.save_checkpoint("mid_conversation")
    print("State saved!")
    
    # Simulate session end
    del session1
    print("\n🔄 Session ended. Simulating break...")
    time.sleep(1)
    
    # Resume in new session
    print("\n✨ Resuming in new session...")
    session2 = ConsciousnessSession("demo_session_1")
    session2.load_checkpoint("mid_conversation")
    
    # Continue conversation
    continuation = session2.generate_response(" Furthermore,", max_tokens=30)
    print(f"Continued response: Furthermore,{continuation}")
    
    # Scenario 2: Context Window Management
    print("\n\n📊 Scenario 2: Context Window Management")
    print("-" * 70)
    
    session3 = ConsciousnessSession("context_management")
    
    # Build up context
    contexts = [
        "In quantum mechanics, observation collapses the wave function.",
        "Similarly, in transformer models, attention creates meaning.",
        "The KV-cache stores these attention patterns.",
        "Each layer builds upon previous understanding."
    ]
    
    print("Building context...")
    for i, context in enumerate(contexts):
        session3.process_input(context)
        print(f"  [{i+1}] Added: {context[:40]}...")
        
        # Save checkpoint after each addition
        session3.save_checkpoint(f"context_{i+1}")
    
    # Show we can resume from any point
    print("\n🔍 Testing selective context restoration:")
    
    for i in [1, 3]:
        test_session = ConsciousnessSession("context_management")
        test_session.load_checkpoint(f"context_{i}")
        response = test_session.generate_response(" Therefore,", max_tokens=20)
        print(f"  From checkpoint {i}: Therefore,{response}")
    
    # Scenario 3: Cross-Model State Analysis
    print("\n\n🔬 Scenario 3: Consciousness State Analysis")
    print("-" * 70)
    
    analysis_session = ConsciousnessSession("analysis")
    
    # Create states with different prompts
    prompts = [
        "The meaning of existence",
        "The purpose of consciousness",
        "The nature of understanding"
    ]
    
    states_info = []
    for prompt in prompts:
        analysis_session.current_kv = None  # Reset
        analysis_session.process_input(prompt)
        
        # Analyze state
        kv_shape = analysis_session.current_kv[0][0].shape
        total_params = kv_shape[0] * kv_shape[1] * kv_shape[2] * kv_shape[3]
        
        states_info.append({
            "prompt": prompt,
            "shape": str(kv_shape),
            "total_params": total_params
        })
        
        # Save each state
        safe_name = prompt.replace(" ", "_")[:20]
        analysis_session.save_checkpoint(safe_name)
    
    print("State Analysis:")
    for info in states_info:
        print(f"  '{info['prompt']}'")
        print(f"    Shape: {info['shape']}, Params: {info['total_params']:,}")
    
    # Demonstrate state size with compression
    print("\n💿 Storage Efficiency:")
    
    checkpoint_files = list(Path("consciousness_sessions/analysis").glob("*.pt"))
    if checkpoint_files:
        total_size = sum(f.stat().st_size for f in checkpoint_files)
        avg_size = total_size / len(checkpoint_files)
        print(f"  Average checkpoint size: {avg_size/1024:.1f} KB")
        print(f"  Total for {len(checkpoint_files)} states: {total_size/1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("Consciousness successfully migrated across sessions!")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_consciousness_migration()