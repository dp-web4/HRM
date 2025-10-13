#!/usr/bin/env python3
"""
Cognitive File Watcher - Claude's Active Participation in SAGE Conversations

This script implements Claude as a file-watching cognitive IRP:
1. Watches /tmp/sage_user_speech.txt for new user input
2. Formulates thoughtful response (YOU write the response manually)
3. Logs metrics for learning (response time, confidence, etc.)
4. Writes response to /tmp/sage_claude_response.txt

**IMPORTANT**: This is a TEMPLATE for the human operating Claude Code.
When user speech appears, YOU (human) respond by writing to the response file.
This script just provides the infrastructure and metrics logging.

The goal: Collect data about cognitive processing to inform IRP design.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

class CognitiveMetrics:
    """Track metrics for cognitive processing"""

    def __init__(self, log_file: str = "/tmp/cognitive_irp_metrics.jsonl"):
        self.log_file = log_file
        self.session_start = time.time()
        self.exchange_count = 0

    def log_exchange(self,
                    user_input: str,
                    user_confidence: float,
                    response: str,
                    response_time: float,
                    metadata: Dict[str, Any] = None):
        """Log one conversation exchange with metrics"""

        self.exchange_count += 1

        # Compute simple metrics
        input_words = len(user_input.split())
        response_words = len(response.split())
        response_chars = len(response)

        # Estimate complexity (very simple heuristic)
        input_complexity = self._estimate_complexity(user_input)

        # Check for confidence markers in response
        response_confidence = self._estimate_response_confidence(response)

        # Create log entry
        entry = {
            'exchange_id': self.exchange_count,
            'timestamp': datetime.now().isoformat(),
            'session_time_s': time.time() - self.session_start,
            'input': {
                'text': user_input,
                'transcription_confidence': user_confidence,
                'word_count': input_words,
                'complexity_score': input_complexity
            },
            'response': {
                'text': response,
                'word_count': response_words,
                'char_count': response_chars,
                'response_time_s': response_time,
                'estimated_confidence': response_confidence
            },
            'metrics': {
                'words_per_second': response_words / response_time if response_time > 0 else 0,
                'latency_s': response_time,
                'atp_estimate': self._estimate_atp_cost(response_words, response_time)
            },
            'metadata': metadata or {}
        }

        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        return entry

    def _estimate_complexity(self, text: str) -> float:
        """Estimate input complexity (0-1)"""
        # Simple heuristics:
        # - Short greetings = low complexity
        # - Questions = medium complexity
        # - Long, multi-sentence = high complexity

        words = text.split()
        word_count = len(words)

        # Length factor
        if word_count < 5:
            length_score = 0.2
        elif word_count < 15:
            length_score = 0.5
        else:
            length_score = 0.8

        # Question factor
        question_score = 0.3 if '?' in text else 0.0

        # Multiple sentences
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        sentence_score = min(0.3, sentence_count * 0.1)

        return min(1.0, length_score + question_score + sentence_score)

    def _estimate_response_confidence(self, text: str) -> float:
        """Estimate confidence from response text"""
        text_lower = text.lower()

        # High confidence markers
        certain_phrases = ['i\'m certain', 'definitely', 'absolutely', 'clearly', 'obviously']
        certain_count = sum(1 for phrase in certain_phrases if phrase in text_lower)

        # Low confidence markers
        uncertain_phrases = ['i think', 'maybe', 'perhaps', 'possibly', 'i\'m not sure', 'might']
        uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in text_lower)

        # Base confidence
        base = 0.7

        # Adjust based on markers
        confidence = base + (certain_count * 0.1) - (uncertain_count * 0.1)

        return max(0.0, min(1.0, confidence))

    def _estimate_atp_cost(self, word_count: int, time_s: float) -> float:
        """Estimate ATP cost for this cognitive processing"""
        # Rough model:
        # - Base cost: 1.0 ATP for invocation
        # - Word cost: 0.05 ATP per word generated
        # - Time cost: 0.1 ATP per second

        base_cost = 1.0
        word_cost = word_count * 0.05
        time_cost = time_s * 0.1

        return base_cost + word_cost + time_cost

    def print_summary(self):
        """Print session summary"""
        if not os.path.exists(self.log_file):
            print("No exchanges logged yet")
            return

        with open(self.log_file, 'r') as f:
            entries = [json.loads(line) for line in f]

        if not entries:
            print("No exchanges logged yet")
            return

        print("\n" + "="*60)
        print("COGNITIVE IRP SESSION SUMMARY")
        print("="*60)
        print(f"Exchanges: {len(entries)}")
        print(f"Session duration: {time.time() - self.session_start:.1f}s")

        # Aggregate metrics
        total_atp = sum(e['metrics']['atp_estimate'] for e in entries)
        avg_response_time = sum(e['metrics']['latency_s'] for e in entries) / len(entries)
        avg_complexity = sum(e['input']['complexity_score'] for e in entries) / len(entries)

        print(f"\nTotal ATP spent: {total_atp:.1f}")
        print(f"Average response time: {avg_response_time:.2f}s")
        print(f"Average input complexity: {avg_complexity:.2f}")

        print("\n" + "="*60)


class CognitiveFileWatcher:
    """Watch for user speech and facilitate response"""

    def __init__(self):
        self.input_file = "/tmp/sage_user_speech.txt"
        self.response_file = "/tmp/sage_claude_response.txt"
        self.last_input_mtime = 0
        self.metrics = CognitiveMetrics()

        print("="*60)
        print("COGNITIVE FILE WATCHER - Ready")
        print("="*60)
        print(f"Watching: {self.input_file}")
        print(f"Response: {self.response_file}")
        print(f"Metrics:  {self.metrics.log_file}")
        print()
        print("When user speaks:")
        print("1. You'll see their transcription")
        print("2. WRITE YOUR RESPONSE to response file")
        print("3. Metrics will be logged automatically")
        print("="*60)
        print()

    def watch(self):
        """Watch for new user input"""
        print("👀 Watching for user speech...")
        print("   (Press Ctrl+C to exit)\n")

        try:
            while True:
                if os.path.exists(self.input_file):
                    mtime = os.path.getmtime(self.input_file)

                    if mtime > self.last_input_mtime:
                        self.last_input_mtime = mtime
                        self._handle_new_input()

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n✅ Cognitive watcher stopped")
            self.metrics.print_summary()

    def _handle_new_input(self):
        """Handle new user input"""
        # Read input
        with open(self.input_file, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        user_text = lines[0]
        confidence = 0.0

        # Parse confidence if present
        for line in lines[1:]:
            if line.startswith('confidence:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    pass

        print("\n" + "🔔 NEW USER INPUT " + "="*40)
        print(f"👤 USER [{confidence:.2f}]: {user_text}")
        print("="*60)
        print()
        print("💭 PLEASE WRITE YOUR RESPONSE TO:")
        print(f"   {self.response_file}")
        print()
        print("   Waiting for your response...")

        # Wait for response file to be written
        response_timeout = 300  # 5 minutes max
        start_wait = time.time()
        last_response_mtime = os.path.getmtime(self.response_file) if os.path.exists(self.response_file) else 0

        while True:
            if os.path.exists(self.response_file):
                mtime = os.path.getmtime(self.response_file)

                if mtime > last_response_mtime:
                    # New response written
                    response_time = time.time() - start_wait

                    with open(self.response_file, 'r') as f:
                        response = f.read().strip()

                    if response:
                        print(f"\n✅ Response received ({response_time:.2f}s)")
                        print(f"🤖 SAGE: {response[:100]}{'...' if len(response) > 100 else ''}")

                        # Log metrics
                        self.metrics.log_exchange(
                            user_input=user_text,
                            user_confidence=confidence,
                            response=response,
                            response_time=response_time
                        )

                        print("\n👂 Listening for next input...\n")
                        break

            # Check timeout
            if time.time() - start_wait > response_timeout:
                print(f"\n⚠️  Response timeout (5 minutes)")
                break

            time.sleep(1)


def main():
    """Main entry point"""
    watcher = CognitiveFileWatcher()
    watcher.watch()


if __name__ == "__main__":
    main()
