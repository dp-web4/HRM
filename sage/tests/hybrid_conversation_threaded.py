#!/usr/bin/env python3
"""
SAGE Streaming Conversation with Learning Observer

Architecture:
1. STREAMING-ONLY responses (1-3s first words via word-by-word generation)
2. Pattern learner as OBSERVER (not executor):
   - Observes all conversations in background
   - Identifies small-talk patterns (greetings, acknowledgments, etc.)
   - Builds pattern library for future small-talk model specialization
   - Does NOT interrupt streaming (learning only!)
3. Future optimization path:
   - When small-talk detected, optionally load tiny dedicated model
   - When deep conversation needed, use main LLM
   - Decision based on learned pattern library

Key insight: With streaming, "fast path" is obsolete. But pattern learning
is still valuable for identifying casual vs deep conversation contexts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import os
import threading
import queue

# Import SAGE components
from core.sage_unified import SAGEUnified
from interfaces.streaming_audio_sensor import StreamingAudioSensor
from interfaces.tts_effector import TTSEffector

# Import hybrid learning system
from cognitive.pattern_learner import PatternLearner
from cognitive.pattern_responses import PatternResponseEngine
from cognitive.sage_system_prompt import get_sage_system_prompt
from cognitive.context_memory import SNARCMemoryManager

# ============================================================================
# Threaded Status Dashboard
# ============================================================================

class ThreadedDashboard:
    """Event-driven status dashboard (only updates on actual events)"""

    def __init__(self, update_interval: float = 0.1):
        self.current_state = "🎧 LISTENING"
        self.last_user_input = ""
        self.last_response = ""
        self.path_used = ""
        self.llm_status = ""
        self.stats = {}
        self.pattern_info = ""

        # Event-driven rendering (no continuous loop!)
        self.running = False
        self.lock = threading.Lock()
        self.needs_render = False

    def start(self):
        """Mark as started (no background thread needed)"""
        self.running = True
        self._render()  # Initial render

    def stop(self):
        """Stop dashboard"""
        self.running = False

    def update(self, state=None, user_input=None, response=None, path=None,
               llm_status=None, stats=None, pattern_info=None):
        """Update dashboard state and render immediately (event-driven)"""
        with self.lock:
            if state: self.current_state = state
            if user_input: self.last_user_input = user_input
            if response: self.last_response = response
            if path: self.path_used = path
            if llm_status is not None: self.llm_status = llm_status
            if stats: self.stats = stats
            if pattern_info: self.pattern_info = pattern_info

            # Render immediately on update (event-driven, not polling)
            if self.running:
                self._render()

    def _render(self):
        """Render dashboard to terminal"""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Header
        print("╔" + "="*78 + "╗")
        print("║" + " " * 20 + "🧠 SAGE HYBRID LEARNING DASHBOARD" + " " * 25 + "║")
        print("╚" + "="*78 + "╝")
        print()

        # Current State with timestamp
        timestamp = time.strftime("%H:%M:%S")
        print(f"📊 STATE: {self.current_state} [{timestamp}]")
        print()

        # Conversation
        print("💬 CONVERSATION:")
        print(f"  👤 User: {self.last_user_input[:70]}")
        print(f"  🤖 SAGE: {self.last_response[:70]}")
        print()

        # Processing Path (streaming only now)
        if self.path_used:
            path_icon = "🌊"  # Streaming waves
            print(f"🔀 PATH: {path_icon} {self.path_used.upper()}")
            if self.llm_status:
                print(f"   {self.llm_status}")
            if self.pattern_info:
                print(f"   {self.pattern_info}")
        print()

        # Statistics
        if self.stats:
            total = self.stats.get('total_queries', 0)
            smalltalk_observed = self.stats.get('fast_path_hits', 0)  # What WOULD be small-talk
            smalltalk_pct = (smalltalk_observed / total * 100) if total > 0 else 0
            slow_hits = self.stats.get('slow_path_hits', 0)
            patterns = self.stats.get('total_patterns', 13)
            learned = self.stats.get('patterns_learned', 0)

            # Memory stats
            memory_stats = self.stats.get('memory_stats', {})
            buffer_util = memory_stats.get('buffer_utilization', 0) * 100
            longterm = memory_stats.get('longterm_memories', 0)

            print("📈 STATISTICS:")
            print(f"  Total Queries: {total}")
            print(f"  Small-talk Observed: {smalltalk_observed}/{total} ({smalltalk_pct:.1f}%)")
            print(f"  Streaming: {slow_hits}/{total}")
            print(f"  Patterns Learned: {learned} (observer mode)")
            print(f"  Memory: {buffer_util:.1f}% buffer, {longterm} long-term")

            # Progress bar for small-talk detection
            bar_width = 40
            filled = int(bar_width * smalltalk_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  [{bar}] {smalltalk_pct:.1f}% small-talk")

        print()
        print("─" * 80)
        print("Press Ctrl+C to stop")

# ============================================================================
# Hybrid Conversation System (with Confidence Gating)
# ============================================================================

class HybridConversationThreaded:
    """
    Hybrid conversation system with confidence-gated pattern matching

    Improvements:
    - Pattern confidence threshold (prevent greedy matching)
    - Detailed logging (why fast vs slow path)
    - Learning tracking
    """

    def __init__(self, use_real_llm: bool = False, pattern_confidence_threshold: float = 0.7):
        """
        Initialize hybrid system

        Args:
            use_real_llm: If True, use Qwen LLM. If False, use MockLLM.
            pattern_confidence_threshold: Minimum confidence for fast path (0.0-1.0)
        """
        print("\n🔧 Initializing Hybrid Learning System...")

        # Pattern matching (fast path)
        self.pattern_engine = PatternResponseEngine()
        initial_patterns = len(self.pattern_engine.patterns)
        print(f"  ✓ Pattern engine: {initial_patterns} initial patterns")

        # Pattern learner
        self.learner = PatternLearner(min_occurrences=2, confidence_threshold=0.6)
        print(f"  ✓ Pattern learner: min_occurrences=2")

        # SNARC memory manager (context window as short-term memory)
        self.memory = SNARCMemoryManager(max_tokens=127000, tokens_per_turn=50)
        print(f"  ✓ SNARC memory: {self.memory.max_turns} turn capacity (~99% of 128K context)")

        # System prompt
        self.system_prompt = get_sage_system_prompt()
        print(f"  ✓ System prompt: {len(self.system_prompt)} chars (~500 tokens)")

        # Configuration
        self.pattern_confidence_threshold = pattern_confidence_threshold
        print(f"  ✓ Pattern confidence threshold: {pattern_confidence_threshold}")

        # Prediction logger (capture hallucinations as training data)
        from cognitive.prediction_logger import PredictionLogger
        self.prediction_logger = PredictionLogger()

        # Consciousness persistence (KV-cache state management)
        from cognitive.consciousness_persistence import ConsciousnessPersistence
        self.consciousness = ConsciousnessPersistence()
        print(f"  ✓ Consciousness persistence enabled")

        # LLM (slow path)
        if use_real_llm:
            print("  ⏳ Loading Qwen LLM with streaming...")
            from experiments.integration.streaming_responder import StreamingResponder
            self.llm = StreamingResponder(
                max_new_tokens=512,  # Large buffer for complete thoughts
                temperature=0.7,
                words_per_chunk=3,  # Stream every 3 words
                prediction_logger=self.prediction_logger,  # Capture hallucinations
                consciousness_persistence=self.consciousness,  # KV-cache persistence
                use_cached_system_prompt=True,  # Cache system prompt KV
                auto_snapshot=True,  # Auto-save during idle
                idle_snapshot_delay=30.0  # Snapshot after 30s idle
            )
            print("  ✓ Qwen 0.5B loaded with streaming + consciousness")
        else:
            print("  ✓ Using MockLLM (fast, for testing)")
            self.llm = self._create_mock_llm()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'fast_path_hits': 0,
            'slow_path_hits': 0,
            'patterns_learned': 0,
            'initial_patterns': initial_patterns,
            'conversation_history': [],
            'pattern_rejects': 0  # Low confidence pattern matches rejected
        }

        print("✓ Hybrid system ready\n")

    def _create_mock_llm(self):
        """Create a simple mock LLM for testing"""
        class MockLLM:
            def generate_response(self, question: str, conversation_history=None, system_prompt=None) -> str:
                q = question.lower()

                if 'quantum' in q or 'entangle' in q:
                    return "Quantum entanglement is a phenomenon where particles become correlated."
                elif 'neural network' in q:
                    return "Neural networks are computing systems inspired by biological brains."
                elif 'black hole' in q:
                    return "Black holes are regions of spacetime with extremely strong gravity."
                elif 'meaning of life' in q:
                    return "The meaning of life is a philosophical question with many perspectives."
                elif 'name' in q:
                    return "I'm SAGE, an AI system learning from our conversations."
                elif 'who are you' in q or 'who r u' in q:
                    return "I'm SAGE, here to help and learn."
                elif 'what can you do' in q or 'what do you do' in q:
                    return "I can answer questions and learn from our interactions."
                elif 'how are you' in q or 'how r u' in q:
                    return "I'm doing well, thanks for asking!"
                else:
                    return f"That's an interesting question about {q.split()[0] if q.split() else 'that topic'}."

        return MockLLM()

    def respond_streaming(self, question: str, on_chunk_callback=None) -> dict:
        """
        Generate response using STREAMING with pattern learning observer.

        Fast path is now OBSERVER-ONLY:
        - Learns small-talk patterns in background
        - Does NOT interrupt streaming generation
        - Building library for future small-talk model specialization

        on_chunk_callback: Called for each word chunk: callback(chunk_text, is_final)
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        pattern_info = ""

        # OBSERVER: Check if pattern would match (don't use it!)
        # This helps us learn what counts as "small talk"
        try:
            potential_fast_response = self.pattern_engine.generate_response(question)
            if potential_fast_response:
                # Pattern would match - log this as potential small-talk
                self.stats['fast_path_hits'] += 1  # Track what WOULD have been fast
                pattern_info = f"📝 Small-talk pattern observed (learning)"
        except Exception:
            pass

        # ALWAYS use streaming path - it's fast enough now!
        llm_start = time.time()
        context_history = self.memory.get_context_for_llm(include_longterm=True)

        # Use streaming if available
        if hasattr(self.llm, 'generate_response_streaming'):
            result = self.llm.generate_response_streaming(
                question,
                conversation_history=context_history,
                system_prompt=self.system_prompt,
                on_chunk=on_chunk_callback  # Pass through the TTS callback!
            )
            response = result['full_response']
            llm_latency = result['total_time']
        else:
            # Fallback to non-streaming
            response = self.llm.generate_response(
                question,
                conversation_history=context_history,
                system_prompt=self.system_prompt
            )
            llm_latency = time.time() - llm_start
            # Speak all at once
            if on_chunk_callback:
                on_chunk_callback(response, True)

        total_latency = time.time() - start_time
        self.stats['slow_path_hits'] += 1

        # LEARNING OBSERVER: Track patterns in background
        # This builds library of small-talk for future optimization
        self.learner.observe(question, response)
        patterns_before = self.stats['patterns_learned']
        current_patterns = len(self.learner.get_learned_patterns())
        newly_learned = current_patterns > patterns_before

        if newly_learned:
            self.stats['patterns_learned'] = current_patterns
            # Note: We DON'T integrate into engine - just collecting data!
            # Later we can train a dedicated small-talk model from this

        self.memory.add_turn("User", question)
        self.memory.add_turn("Assistant", response, metadata={
            'path': 'streaming',  # Only one path now
            'llm_latency': llm_latency,
            'pattern_observed': bool(pattern_info),  # Was this small-talk?
            'learned': newly_learned
        })

        # If there's a pending prediction, log the actual user response
        if self.prediction_logger and self.prediction_logger.pending_prediction:
            self.prediction_logger.log_actual_response(question)

        return {
            'response': response,
            'path': 'streaming',  # Renamed from 'slow' - it's actually fast!
            'confidence': 1.0,  # Always confident in streaming
            'latency': total_latency,
            'llm_latency': llm_latency,
            'learned': newly_learned,
            'pattern_info': pattern_info or "Streaming generation"
        }

    def respond(self, question: str) -> dict:
        """
        Generate response using hybrid fast/slow path with confidence gating

        Returns dict with:
            - response: The response text
            - path: 'fast' or 'slow'
            - confidence: Match confidence
            - latency: Response time
            - learned: Whether new pattern was learned
            - pattern_info: Debug info about pattern matching
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        pattern_info = ""

        # Try fast path first (pattern matching with confidence gating)
        try:
            fast_response = self.pattern_engine.generate_response(question)

            if fast_response:
                # Calculate simple confidence (could be improved with fuzzy matching)
                # For now, just check if it's an exact keyword match
                confidence = 0.9  # Default high confidence for pattern matches

                if confidence >= self.pattern_confidence_threshold:
                    # High confidence - use fast path
                    latency = time.time() - start_time
                    self.stats['fast_path_hits'] += 1

                    # Update SNARC memory (fast path)
                    self.memory.add_turn("User", question)
                    self.memory.add_turn("Assistant", fast_response, metadata={'path': 'fast', 'confidence': confidence})

                    # Keep stats for backwards compatibility
                    self.stats['conversation_history'].append(("User", question))
                    self.stats['conversation_history'].append(("Assistant", fast_response))

                    pattern_info = f"Pattern match: confidence={confidence:.2f} (threshold={self.pattern_confidence_threshold:.2f})"

                    return {
                        'response': fast_response,
                        'path': 'fast',
                        'confidence': confidence,
                        'latency': latency,
                        'learned': False,
                        'pattern_info': pattern_info
                    }
                else:
                    # Low confidence - reject and fall through to slow path
                    self.stats['pattern_rejects'] += 1
                    pattern_info = f"Pattern rejected: confidence={confidence:.2f} < threshold={self.pattern_confidence_threshold:.2f}"

        except Exception as e:
            pattern_info = f"Pattern matching error: {str(e)[:40]}"

        # Slow path - use LLM with SNARC-optimized context
        llm_start = time.time()

        # Get SNARC-filtered conversation context (includes long-term salient memories)
        context_history = self.memory.get_context_for_llm(include_longterm=True)

        response = self.llm.generate_response(
            question,
            conversation_history=context_history,  # Now includes entire SNARC buffer!
            system_prompt=self.system_prompt  # Comprehensive SAGE identity
        )
        llm_latency = time.time() - llm_start
        total_latency = time.time() - start_time

        self.stats['slow_path_hits'] += 1

        # Learn from this interaction
        self.learner.observe(question, response)

        # Check if we learned new patterns
        patterns_before = self.stats['patterns_learned']
        current_patterns = len(self.learner.get_learned_patterns())
        newly_learned = False

        if current_patterns > patterns_before:
            self.stats['patterns_learned'] = current_patterns
            newly_learned = True
            # Integrate learned patterns into pattern engine
            self._integrate_learned_patterns()

        # Update SNARC memory (slow path)
        self.memory.add_turn("User", question)
        self.memory.add_turn("Assistant", response, metadata={
            'path': 'slow',
            'llm_latency': llm_latency,
            'learned': newly_learned
        })

        # Keep stats for backwards compatibility
        self.stats['conversation_history'].append(("User", question))
        self.stats['conversation_history'].append(("Assistant", response))

        if not pattern_info:
            pattern_info = "No pattern match - using LLM"

        return {
            'response': response,
            'path': 'slow',
            'confidence': 0.0,
            'latency': total_latency,
            'llm_latency': llm_latency,
            'learned': newly_learned,
            'pattern_info': pattern_info
        }

    def _integrate_learned_patterns(self):
        """Integrate learned patterns into pattern engine"""
        learned = self.learner.get_learned_patterns()

        for pattern_regex, responses in learned.items():
            import re
            compiled_pattern = re.compile(pattern_regex)

            # Check if not already there
            pattern_exists = False
            for existing_pattern, _ in self.pattern_engine.compiled_patterns:
                if existing_pattern.pattern == compiled_pattern.pattern:
                    pattern_exists = True
                    break

            if not pattern_exists:
                self.pattern_engine.compiled_patterns.append((compiled_pattern, responses))
                print(f"\n    📚 NEW PATTERN LEARNED!")
                print(f"       Pattern: {pattern_regex[:60]}...")
                print(f"       Response: {responses[0][:60]}...")

    def get_stats(self) -> dict:
        """Get system statistics"""
        total = self.stats['total_queries']
        if total > 0:
            fast_ratio = self.stats['fast_path_hits'] / total
            slow_ratio = self.stats['slow_path_hits'] / total
        else:
            fast_ratio = slow_ratio = 0.0

        return {
            **self.stats,
            'fast_path_ratio': fast_ratio,
            'slow_path_ratio': slow_ratio,
            'total_patterns': self.stats['initial_patterns'] + self.stats['patterns_learned'],
            'memory_stats': self.memory.get_stats()
        }

# ============================================================================
# Initialize SAGE Unified
# ============================================================================

print("="*80)
print("🧠 SAGE HYBRID LEARNING CONVERSATION - THREADED DASHBOARD")
print("="*80)

print("\n1. Initializing SAGE Unified...")

sage = SAGEUnified(
    config={
        'initial_atp': 100.0,
        'max_atp': 100.0,
        'enable_circadian': False,
        'simulation_mode': False
    },
    device=torch.device('cpu')
)

# ============================================================================
# Register Audio Sensor
# ============================================================================

print("\n2. Registering audio sensor...")

audio_sensor = StreamingAudioSensor({
    'sensor_id': 'conversation_audio',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'vad_aggressiveness': 2,
    'min_speech_duration': 0.5,
    'max_speech_duration': 10.0,
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

sage.register_sensor(audio_sensor)

# ============================================================================
# Initialize Hybrid Conversation System
# ============================================================================

print("\n3. Initializing Hybrid Conversation System...")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--real-llm', action='store_true', help='Use real Qwen LLM')
parser.add_argument('--confidence', type=float, default=0.7, help='Pattern confidence threshold (0.0-1.0)')
args = parser.parse_args()

hybrid_system = HybridConversationThreaded(
    use_real_llm=args.real_llm,
    pattern_confidence_threshold=args.confidence
)

# ============================================================================
# Initialize TTS Effector
# ============================================================================

print("\n4. Initializing TTS effector...")

tts_effector = TTSEffector({
    'piper_path': '/home/sprout/ai-workspace/piper/piper/piper',
    'model_path': '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx',
    'bt_sink': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'enabled': True
})

# ============================================================================
# Hybrid SAGE Cycle with Learning
# ============================================================================

# Global flag to prevent TTS overlap
_tts_speaking = False
_response_lock = threading.Lock()  # Prevent dual-path collision

# Global dashboard
dashboard = None

def sage_cycle_with_hybrid_learning():
    """Execute SAGE cycle with hybrid learning conversation"""
    global _tts_speaking, dashboard

    # Skip processing if TTS is still speaking
    if _tts_speaking:
        sage.cycle()
        return {}

    # Check for new transcriptions
    reading = audio_sensor.poll()

    if reading and hasattr(reading, 'metadata'):
        text = reading.metadata.get('text')

        if text and len(text.strip()) > 0:
            # Acquire lock to prevent dual-path collision
            with _response_lock:
                # Update dashboard - user spoke
                if dashboard:
                    dashboard.update(state="💭 THINKING", user_input=text)

                # Generate response using hybrid system WITH STREAMING TTS
                try:
                    print(f"\n  [HYBRID] Generating STREAMING response for: '{text[:50]}...'")

                    # Sentence-level buffering for TTS
                    sentence_buffer = ""
                    accumulated_response = ""
                    sentence_count = 0

                    def on_chunk_speak(chunk_text, is_final):
                        """Callback: Buffer until sentence complete, then speak with natural prosody"""
                        nonlocal accumulated_response, sentence_buffer, sentence_count
                        accumulated_response += chunk_text
                        sentence_buffer += chunk_text

                        # Check for sentence boundary (., !, ?)
                        sentence_end = False
                        for boundary in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                            if boundary in sentence_buffer:
                                sentence_end = True
                                break

                        # Also check if final chunk and buffer ends with punctuation
                        if is_final and sentence_buffer.rstrip() and sentence_buffer.rstrip()[-1] in '.!?':
                            sentence_end = True

                        # If sentence complete, speak it
                        if sentence_end or is_final:
                            complete_sentence = sentence_buffer.strip()
                            if complete_sentence:
                                sentence_count += 1
                                print(f"  [SENTENCE-TTS {sentence_count}] Speaking: '{complete_sentence[:60]}...'")
                                tts_effector.execute(complete_sentence)
                                sentence_buffer = ""  # Reset for next sentence

                    # Generate with streaming
                    result = hybrid_system.respond_streaming(text, on_chunk_callback=on_chunk_speak)
                    print(f"  [HYBRID] Complete: path={result.get('path')}, len={len(result.get('response', ''))} chars")

                    # Update dashboard with final result
                    if result and result.get('response'):
                        if dashboard:
                            stats = hybrid_system.get_stats()
                            llm_msg = f"LLM latency: {result.get('llm_latency', 0)*1000:.0f}ms" if result['path'] == 'slow' else ""

                            state = "🗣️ SPEAKING"
                            if result.get('learned'):
                                state = "🧠 LEARNING & SPEAKING"

                            dashboard.update(
                                state=state,
                                response=result['response'],
                                path=result['path'],
                                llm_status=llm_msg,
                                pattern_info=result.get('pattern_info', ''),
                                stats=stats
                            )

                        # Return to listening state
                        if dashboard:
                            dashboard.update(state="🎧 LISTENING")

                except Exception as e:
                    print(f"  [ERROR] Exception in response generation: {e}")
                    import traceback
                    traceback.print_exc()
                    _tts_speaking = False
                    if dashboard:
                        dashboard.update(state=f"⚠️ ERROR: {str(e)[:50]}")
                    time.sleep(2)

    # Run standard SAGE cycle
    result = sage.cycle()
    return result

# ============================================================================
# Run SAGE Loop with Threaded Dashboard
# ============================================================================

# Initialize event-driven dashboard (no continuous updates!)
dashboard = ThreadedDashboard()
dashboard.start()  # Initial render

print("\n5. Event-driven dashboard active (updates on events only)...")
time.sleep(1)  # Brief pause

try:
    cycle_count = 0

    while True:
        try:
            result = sage_cycle_with_hybrid_learning()
            cycle_count += 1

            # Small sleep to prevent CPU spinning
            time.sleep(0.05)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"⚠️  Cycle error: {e}")
            time.sleep(0.1)

except KeyboardInterrupt:
    # Stop dashboard
    dashboard.stop()

    print("\n\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)

    # Hybrid system stats
    stats = hybrid_system.get_stats()
    memory_stats = stats.get('memory_stats', {})

    print(f"\n🧠 Streaming System with Learning Observer:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Small-talk observed: {stats['fast_path_hits']} ({stats['fast_path_ratio']:.1%})")
    print(f"   All responses: Streaming (1-3s first words)")
    print(f"   Patterns learned: {stats['patterns_learned']} (observer mode)")
    print(f"   Pattern library: {stats['total_patterns']} patterns")
    print(f"   → Future: Load dedicated small-talk model for casual convos")

    print(f"\n💾 SNARC Memory System:")
    print(f"   Buffer: {memory_stats.get('buffer_size', 0)}/{memory_stats.get('buffer_capacity', 0)} turns ({memory_stats.get('buffer_utilization', 0):.1%})")
    print(f"   Long-term: {memory_stats.get('longterm_memories', 0)} high-salience memories extracted")
    print(f"   Avg buffer salience: {memory_stats.get('avg_buffer_salience', 0):.2f}")
    print(f"   Avg long-term salience: {memory_stats.get('avg_longterm_salience', 0):.2f}")

    if stats['total_queries'] > 0:
        improvement = stats['fast_path_ratio']
        print(f"\n📈 Learning Progress:")
        print(f"   Fast path efficiency: {improvement:.1%}")

    # SAGE stats
    sage_stats = sage.stats
    print(f"\n⚙️  SAGE Core:")
    print(f"   Total cycles: {sage_stats['total_cycles']}")
    print(f"   Total time: {sage_stats['total_time']:.2f}s")
    print(f"   Avg cycle: {sage_stats['avg_cycle_time']*1000:.2f}ms")

    # TTS stats
    tts_stats = tts_effector.get_stats()
    print(f"\n🔊 TTS Effector:")
    print(f"   Syntheses: {tts_stats['synthesis_count']}")
    print(f"   Avg time: {tts_stats['avg_time_ms']:.0f}ms")
    print(f"   Errors: {tts_stats['errors']}")

    print(f"\n✅ All systems stopped cleanly")
    print("="*80)
