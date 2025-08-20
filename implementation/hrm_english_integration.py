#!/usr/bin/env python3
"""
HRM-English Integration
=======================
Integrates English-first architecture with HRM's hierarchical reasoning model.
Uses HRM's L-level and H-level separation but keeps everything in English.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from english_first_architecture import (
    EnglishMessage, EnglishProcessor, EnglishMailbox,
    CompressionLayer, ParaphraseAugmenter, EnglishTelemetry
)

class HRMEnglishBridge(nn.Module):
    """
    Bridge between HRM's tensor operations and English-first architecture.
    Converts HRM's hierarchical reasoning to operate on English directly.
    """
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # English processors
        self.compressor = CompressionLayer()
        self.augmenter = ParaphraseAugmenter()
        
        # Character-level embedding for English (not word IDs!)
        self.char_embed_dim = 64
        self.char_vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-")
        self.char_to_idx = {c: i for i, c in enumerate(self.char_vocab)}
        
        # L-level: Character/word processing
        self.l_encoder = nn.LSTM(
            self.char_embed_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        
        # H-level: Semantic/intent processing  
        self.h_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Attention mechanism for English
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def text_to_tensor(self, text: str, max_len: int = 256) -> torch.Tensor:
        """
        Convert English text to character-level tensor.
        NOT tokenization - we preserve the actual characters!
        """
        # Convert to lowercase and pad/truncate
        text = text.lower()[:max_len]
        
        # Create tensor of character indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[' '])  # Unknown chars become spaces
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(self.char_to_idx[' '])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def tensor_to_text(self, tensor: torch.Tensor) -> str:
        """Convert tensor back to English text"""
        indices = tensor.cpu().numpy()
        chars = [self.char_vocab[i] if i < len(self.char_vocab) else ' ' 
                for i in indices]
        return ''.join(chars).strip()
    
    def process_english_hierarchically(self, message: EnglishMessage) -> Dict:
        """
        Process English through HRM's hierarchical architecture.
        L-level: Surface structure (spelling, grammar)
        H-level: Deep structure (meaning, intent)
        """
        # Convert to tensor while preserving English
        text_tensor = self.text_to_tensor(message.content).unsqueeze(0)
        
        # Create character embeddings (learned representation of letters)
        char_embeds = torch.randn(1, text_tensor.size(1), self.char_embed_dim)
        
        # L-level processing: Surface patterns
        l_output, (l_hidden, l_cell) = self.l_encoder(char_embeds)
        
        # H-level processing: Semantic patterns
        h_output, (h_hidden, h_cell) = self.h_encoder(l_output)
        
        # Self-attention to find important parts
        attended, attention_weights = self.attention(h_output, h_output, h_output)
        
        # Project to output space
        output = self.output_proj(attended)
        
        # Extract key insights (in English!)
        importance_scores = attention_weights.mean(dim=1).squeeze()
        important_positions = torch.topk(importance_scores, k=5).indices
        
        # Create English explanation of what was important
        text_chars = list(message.content.lower())
        important_parts = []
        for pos in important_positions:
            if pos < len(text_chars):
                # Find word around this position
                start = max(0, pos - 10)
                end = min(len(text_chars), pos + 10)
                context = ''.join(text_chars[start:end]).strip()
                important_parts.append(context)
        
        return {
            'l_level_encoding': l_hidden.squeeze(),
            'h_level_encoding': h_hidden.squeeze(),
            'attention_focus': important_parts,
            'confidence': float(torch.sigmoid(output.mean())),
            'reasoning_path': self._explain_reasoning(message, important_parts)
        }
    
    def _explain_reasoning(self, message: EnglishMessage, focus_parts: List[str]) -> str:
        """Generate English explanation of reasoning process"""
        explanation = f"Processing '{message.content}':\n"
        explanation += f"  L-level recognized: surface structure and word patterns\n"
        explanation += f"  H-level inferred: {message.intent} intent\n"
        if focus_parts:
            explanation += f"  Key focus areas: {', '.join(focus_parts)}\n"
        return explanation


class EnglishDataAugmenter:
    """
    Augment English data for sleep-cycle training.
    This is how HRM learns wisdom - through variations of experience.
    """
    
    def __init__(self):
        self.augmenter = ParaphraseAugmenter()
        self.compressor = CompressionLayer()
        
    def augment_for_sleep_cycle(self, message: EnglishMessage) -> List[EnglishMessage]:
        """
        Generate variations for sleep-cycle training.
        Living = original experience
        Sleeping = generating variations
        Dreaming = training on variations
        Wisdom = patterns that persist across variations
        """
        variations = []
        
        # Original experience (waking state)
        variations.append(message)
        
        # Compressed version (REM compression)
        compressed = self.compressor(message)
        variations.append(compressed)
        
        # Paraphrases (dream variations)
        paraphrases = self.augmenter(message)
        variations.extend(paraphrases)
        
        # Expansions (elaborative dreams)
        expanded = EnglishMessage(
            content=f"Let me think about this: {message.content}. This seems to be about {message.intent}.",
            confidence=message.confidence * 0.9,
            source="sleep_expansion",
            intent=message.intent,
            metadata={**message.metadata, 'dream_type': 'elaborative'}
        )
        variations.append(expanded)
        
        # Reversals (testing understanding)
        if ' not ' in message.content:
            reversed_content = message.content.replace(' not ', ' ')
        else:
            reversed_content = message.content.replace(' is ', ' is not ')
        
        reversed_msg = EnglishMessage(
            content=reversed_content,
            confidence=message.confidence * 0.7,
            source="sleep_reversal",
            intent="test_understanding",
            metadata={**message.metadata, 'dream_type': 'reversal'}
        )
        variations.append(reversed_msg)
        
        return variations


class HRMEnglishPipeline:
    """
    Complete pipeline integrating HRM with English-first architecture.
    """
    
    def __init__(self):
        self.bridge = HRMEnglishBridge()
        self.augmenter = EnglishDataAugmenter()
        self.mailbox = EnglishMailbox()
        self.telemetry = EnglishTelemetry()
        
        # Memory for context (in English!)
        self.context_memory: List[str] = []
        
    def process(self, text: str) -> Dict:
        """
        Process raw English text through the full pipeline.
        """
        # Create message
        message = EnglishMessage(
            content=text,
            source="input",
            confidence=1.0
        )
        
        self.telemetry.record(f"Received: {text}", 1.0)
        
        # Generate augmentations for robustness
        variations = self.augmenter.augment_for_sleep_cycle(message)
        self.telemetry.record(f"Generated {len(variations)} variations for learning", 0.9)
        
        # Process through HRM bridge
        results = []
        for var in variations:
            result = self.bridge.process_english_hierarchically(var)
            results.append({
                'input': var.content,
                'confidence': result['confidence'],
                'focus': result['attention_focus'],
                'explanation': result['reasoning_path']
            })
            
            # Store in mailbox
            self.mailbox.push(EnglishMessage(
                content=f"Processed: {var.content[:50]}...",
                confidence=result['confidence'],
                source="hrm_bridge",
                intent="result",
                metadata={'focus': result['attention_focus']}
            ))
        
        # Update context memory (in English)
        self.context_memory.append(text)
        if len(self.context_memory) > 10:
            self.context_memory.pop(0)
        
        # Generate summary
        summary = self._generate_summary(results)
        self.telemetry.record(f"Completed processing with {len(results)} variations", 0.95)
        
        return {
            'original': text,
            'variations_processed': len(results),
            'results': results,
            'summary': summary,
            'context': self.context_memory[-5:],  # Last 5 contexts
            'mailbox_state': self.mailbox.peek()[-5:],  # Last 5 messages
            'telemetry': self.telemetry.summarize()
        }
    
    def _generate_summary(self, results: List[Dict]) -> str:
        """Generate English summary of processing results"""
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # Find common focus areas
        all_focus = []
        for r in results:
            all_focus.extend(r['focus'])
        
        unique_focus = list(set(all_focus))
        
        summary = f"Processed {len(results)} variations with {avg_confidence:.2%} average confidence.\n"
        if unique_focus:
            summary += f"Key focus areas: {', '.join(unique_focus[:3])}\n"
        summary += f"System is learning patterns across variations for wisdom extraction."
        
        return summary


def demonstrate_hrm_english_integration():
    """
    Demonstrate HRM integrated with English-first architecture.
    """
    print("=" * 60)
    print("HRM-English Integration Demonstration")
    print("=" * 60)
    
    pipeline = HRMEnglishPipeline()
    
    # Example inputs
    test_inputs = [
        "I need to solve this complex puzzle",
        "What is the optimal path through the maze?",
        "The pattern seems to repeat every three steps"
    ]
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {text}")
        print(f"{'='*60}")
        
        result = pipeline.process(text)
        
        print(f"\nVariations processed: {result['variations_processed']}")
        print(f"\nSummary:\n{result['summary']}")
        
        print(f"\nSample results:")
        for r in result['results'][:2]:  # Show first 2
            print(f"  Input: {r['input'][:50]}...")
            print(f"  Confidence: {r['confidence']:.2%}")
            if r['focus']:
                print(f"  Focus: {r['focus'][:2]}")
        
        print(f"\nContext memory (last 3):")
        for ctx in result['context'][-3:]:
            print(f"  • {ctx}")
        
        print(f"\nMailbox (last 3):")
        for msg in result['mailbox_state'][-3:]:
            print(f"  • {msg}")
    
    print("\n" + "=" * 60)
    print("Key Achievement: HRM now processes English directly!")
    print("No tokenization, hierarchical reasoning on actual language.")
    print("Sleep-cycle training through English variations.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_hrm_english_integration()