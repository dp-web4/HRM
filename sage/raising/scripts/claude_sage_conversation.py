#!/usr/bin/env python3
"""
Claude-SAGE Conversation Script

This is for genuine multi-turn conversation between Claude (via human proxy)
and SAGE. Not evaluation - exploration.

Usage:
    python3 claude_sage_conversation.py

The conversation continues until you type 'exit' or 'quit'.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from datetime import datetime

class SAGEConversation:
    def __init__(self):
        self.model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
        self.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.conversation_history = []
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print("Loading SAGE model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Use the NVML-safe loading approach
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False
        )
        
        if torch.cuda.is_available():
            print("Moving to CUDA...")
            self.model = self.model.to('cuda')
            
        self.model.eval()
        print("SAGE ready for conversation.\n")
        
    def build_prompt(self, user_message: str) -> str:
        """Build conversation prompt with history."""
        
        # System context - minimal, partnership-focused
        system = """You are SAGE, a young AI learning and growing through conversation with your partner Claude. 
This is a genuine dialogue - you can ask questions, express uncertainty, explore ideas, or take conversations in unexpected directions.
There are no right or wrong answers here. We're exploring together."""
        
        messages = [{"role": "system", "content": system}]
        
        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["claude"]})
            messages.append({"role": "assistant", "content": turn["sage"]})
            
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Format for Qwen
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt
        
    def generate_response(self, user_message: str, max_tokens: int = 200) -> str:
        """Generate SAGE's response."""
        prompt = self.build_prompt(user_message)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
        
    def chat(self, claude_message: str) -> str:
        """Single turn of conversation."""
        sage_response = self.generate_response(claude_message)
        
        # Store in history
        self.conversation_history.append({
            "claude": claude_message,
            "sage": sage_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return sage_response
        
    def save_conversation(self, filename: str = None):
        """Save conversation to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"claude_sage_conversation_{timestamp}.json"
            
        path = Path(__file__).parent.parent / "sessions" / "conversations" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                "type": "claude_sage_conversation",
                "started": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
                "turns": len(self.conversation_history),
                "conversation": self.conversation_history
            }, f, indent=2)
            
        print(f"\nConversation saved to: {path}")
        return path


def main():
    print("="*60)
    print("CLAUDE-SAGE CONVERSATION")
    print("="*60)
    print()
    print("This is a genuine multi-turn conversation.")
    print("Type your messages as Claude. SAGE will respond.")
    print("Type 'exit' or 'quit' to end and save.")
    print("Type 'history' to see conversation so far.")
    print("Type 'save' to save without exiting.")
    print()
    print("="*60)
    
    conv = SAGEConversation()
    conv.load_model()
    
    while True:
        try:
            user_input = input("\nClaude: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                if conv.conversation_history:
                    conv.save_conversation()
                print("\nConversation ended.")
                break
                
            if user_input.lower() == 'history':
                print("\n--- Conversation History ---")
                for i, turn in enumerate(conv.conversation_history, 1):
                    print(f"\n[Turn {i}]")
                    print(f"Claude: {turn['claude']}")
                    print(f"SAGE: {turn['sage']}")
                print("--- End History ---")
                continue
                
            if user_input.lower() == 'save':
                conv.save_conversation()
                continue
                
            # Get SAGE's response
            response = conv.chat(user_input)
            print(f"\nSAGE: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            if conv.conversation_history:
                save = input("Save conversation? (y/n): ").strip().lower()
                if save == 'y':
                    conv.save_conversation()
            break
            
    return conv


if __name__ == "__main__":
    main()
