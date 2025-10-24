#!/usr/bin/env python3
"""
SAGE Hybrid Learning Conversation - Real-Time Integration

Combines:
1. Real-time audio (microphone → Whisper → TTS)
2. Hybrid fast/slow path (pattern matching + LLM)
3. Learning from conversations (develops reflexes from experience)

This is the complete system:
- Fast path: Pattern matching (<1ms, procedural memory)
- Slow path: LLM generation (300-500ms, deliberate reasoning)
- Learning: Extract patterns from successful LLM responses
- Real-time: Speak and hear SAGE learn in real-time
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import os

# Import SAGE components
from core.sage_unified import SAGEUnified
from interfaces.streaming_audio_sensor import StreamingAudioSensor
from interfaces.tts_effector import TTSEffector

# Import hybrid learning system
from cognitive.pattern_learner import PatternLearner
from cognitive.pattern_responses import PatternResponseEngine

# ============================================================================
# Status Dashboard
# ============================================================================

class StatusDashboard:
    """Real-time status dashboard for SAGE"""

    def __init__(self):
        self.current_state = "🎧 LISTENING"
        self.last_user_input = ""
        self.last_response = ""
        self.path_used = ""
        self.llm_status = ""
        self.stats = {}

    def clear(self):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def update(self, state=None, user_input=None, response=None, path=None, llm_status=None, stats=None):
        """Update dashboard state"""
        if state: self.current_state = state
        if user_input: self.last_user_input = user_input
        if response: self.last_response = response
        if path: self.path_used = path
        if llm_status is not None: self.llm_status = llm_status
        if stats: self.stats = stats

    def render(self):
        """Render dashboard to terminal"""
        self.clear()

        # Header
        print("╔" + "="*78 + "╗")
        print("║" + " " * 20 + "🧠 SAGE HYBRID LEARNING DASHBOARD" + " " * 25 + "║")
        print("╚" + "="*78 + "╝")
        print()

        # Current State
        print(f"📊 STATE: {self.current_state}")
        print()

        # Conversation
        print("💬 CONVERSATION:")
        print(f"  👤 User: {self.last_user_input[:70]}")
        print(f"  🤖 SAGE: {self.last_response[:70]}")
        print()

        # Processing Path
        if self.path_used:
            path_icon = "⚡" if self.path_used == "fast" else "🧠"
            print(f"🔀 PATH: {path_icon} {self.path_used.upper()}")
            if self.llm_status:
                print(f"   {self.llm_status}")
        print()

        # Statistics
        if self.stats:
            fast_pct = self.stats.get('fast_path_ratio', 0) * 100
            total = self.stats.get('total_queries', 0)
            fast_hits = self.stats.get('fast_path_hits', 0)
            patterns = self.stats.get('total_patterns', 13)
            learned = self.stats.get('patterns_learned', 0)

            print("📈 STATISTICS:")
            print(f"  Total Queries: {total}")
            print(f"  Fast Path: {fast_hits}/{total} ({fast_pct:.1f}%)")
            print(f"  Patterns: {patterns} (+{learned} learned)")

            # Progress bar
            bar_width = 40
            filled = int(bar_width * fast_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  [{bar}] {fast_pct:.1f}%")

        print()
        print("─" * 80)
        print("Press Ctrl+C to stop")

print("="*80)
print("🧠 SAGE HYBRID LEARNING CONVERSATION - REAL-TIME")
print("="*80)

# ============================================================================
# Hybrid Conversation System (with Learning)
# ============================================================================

class HybridConversationRealtime:
    """
    Hybrid conversation system with real-time learning

    Architecture:
    - Fast path: Pattern matching (procedural memory)
    - Slow path: LLM generation (deliberate reasoning)
    - Learning: Pattern extraction from LLM responses
    - Real-time: Audio in/out with live learning updates
    """

    def __init__(self, use_real_llm: bool = False):
        """
        Initialize hybrid system

        Args:
            use_real_llm: If True, use Phi-2. If False, use MockLLM.
        """
        print("\n🔧 Initializing Hybrid Learning System...")

        # Pattern matching (fast path)
        self.pattern_engine = PatternResponseEngine()
        initial_patterns = len(self.pattern_engine.patterns)
        print(f"  ✓ Pattern engine: {initial_patterns} initial patterns")

        # Pattern learner
        self.learner = PatternLearner(min_occurrences=2, confidence_threshold=0.6)
        print(f"  ✓ Pattern learner: min_occurrences=2")

        # LLM (slow path)
        if use_real_llm:
            print("  ⏳ Loading Phi-2 LLM...")
            from experiments.integration.phi2_responder import Phi2Responder
            self.llm = Phi2Responder(max_new_tokens=50, temperature=0.7)
            print("  ✓ Phi-2 loaded")
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
            'conversation_history': []
        }

        print("✓ Hybrid system ready\n")

    def _create_mock_llm(self):
        """Create a simple mock LLM for testing"""
        class MockLLM:
            def generate_response(self, question: str, conversation_history=None, system_prompt=None) -> str:
                q = question.lower()

                if 'name' in q:
                    return "I'm SAGE, an AI system learning from our conversations."
                elif 'who are you' in q or 'who r u' in q:
                    return "I'm SAGE, here to help and learn."
                elif 'what can you do' in q or 'what do you do' in q:
                    return "I can answer questions and learn from our interactions."
                elif 'how are you' in q or 'how r u' in q:
                    return "I'm doing well, thanks for asking!"
                elif 'tell me about yourself' in q or 'about yourself' in q:
                    return "I'm SAGE, a learning AI assistant. I get better at conversations the more we talk!"
                elif 'thank' in q:
                    return "You're welcome! Happy to help."
                elif 'bye' in q or 'goodbye' in q:
                    return "Goodbye! Talk to you soon."
                else:
                    return f"That's an interesting question. Let me think about that."

        return MockLLM()

    def respond(self, question: str) -> dict:
        """
        Generate response using hybrid fast/slow path with learning

        Returns dict with:
            - response: The response text
            - path: 'fast' or 'slow'
            - confidence: Match confidence
            - latency: Response time
            - learned: Whether new pattern was learned
        """
        start_time = time.time()
        self.stats['total_queries'] += 1

        # Try fast path first (pattern matching)
        try:
            fast_response = self.pattern_engine.generate_response(question)
            if fast_response:
                # Fast path hit!
                latency = time.time() - start_time
                self.stats['fast_path_hits'] += 1

                # Still update conversation history for fast path
                self.stats['conversation_history'].append(("User", question))
                self.stats['conversation_history'].append(("Assistant", fast_response))

                return {
                    'response': fast_response,
                    'path': 'fast',
                    'confidence': 0.9,
                    'latency': latency,
                    'learned': False
                }
        except Exception as e:
            pass  # Fall through to slow path

        # Slow path - use LLM
        llm_start = time.time()
        response = self.llm.generate_response(
            question,
            conversation_history=self.stats['conversation_history'][-5:],
            system_prompt="You are SAGE, a learning AI assistant."
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

        # Update conversation history (as tuples for Phi2Responder)
        self.stats['conversation_history'].append(("User", question))
        self.stats['conversation_history'].append(("Assistant", response))

        return {
            'response': response,
            'path': 'slow',
            'confidence': 0.0,
            'latency': total_latency,
            'llm_latency': llm_latency,
            'learned': newly_learned
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
            'total_patterns': self.stats['initial_patterns'] + self.stats['patterns_learned']
        }

# ============================================================================
# Initialize SAGE Unified
# ============================================================================

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
parser.add_argument('--real-llm', action='store_true', help='Use real Phi-2 LLM')
args = parser.parse_args()

hybrid_system = HybridConversationRealtime(use_real_llm=args.real_llm)

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

# Global dashboard
dashboard = None

def sage_cycle_with_hybrid_learning():
    """
    Execute SAGE cycle with hybrid learning conversation

    Flow:
    1. Poll audio sensor for transcriptions
    2. Try fast path (pattern matching)
    3. If miss, use slow path (LLM)
    4. Learn patterns from LLM responses
    5. Synthesize speech response
    6. Track learning progress
    """
    global _tts_speaking, dashboard

    # Skip processing if TTS is still speaking
    if _tts_speaking:
        sage.cycle()  # Still run SAGE cycle
        return {}

    # Check for new transcriptions
    reading = audio_sensor.poll()

    if reading and hasattr(reading, 'metadata'):
        text = reading.metadata.get('text')

        if text and len(text.strip()) > 0:
            # Update dashboard - user spoke
            if dashboard:
                dashboard.update(state="💭 THINKING", user_input=text)
                dashboard.render()

            # Generate response using hybrid system
            try:
                # Check if this will use slow path (LLM)
                # Try pattern match first to see if we'll need LLM
                try:
                    test_response = hybrid_system.pattern_engine.generate_response(text)
                    if not test_response and dashboard:
                        # No pattern match - will use LLM
                        dashboard.update(state="🧠 LLM PROCESSING...")
                        dashboard.render()
                except:
                    pass

                result = hybrid_system.respond(text)

                # Only proceed if we got a valid response
                if result and result.get('response'):
                    # Update dashboard with response
                    if dashboard:
                        stats = hybrid_system.get_stats()
                        llm_msg = f"LLM latency: {result.get('llm_latency', 0)*1000:.0f}ms" if result['path'] == 'slow' else ""
                        dashboard.update(
                            state="🗣️ SPEAKING",
                            response=result['response'],
                            path=result['path'],
                            llm_status=llm_msg,
                            stats=stats
                        )
                        dashboard.render()

                    # Synthesize speech (with overlap protection)
                    _tts_speaking = True
                    tts_effector.execute(result['response'])
                    # Wait a bit for TTS to start, then reset flag
                    # (TTS is non-blocking, so we estimate based on text length)
                    import time
                    estimated_duration = len(result['response']) * 0.08  # ~80ms per character
                    time.sleep(min(estimated_duration, 5.0))  # Cap at 5 seconds
                    _tts_speaking = False

            except Exception as e:
                _tts_speaking = False  # Reset on error
                if dashboard:
                    dashboard.update(state=f"⚠️ ERROR: {str(e)[:50]}")
                    dashboard.render()
                time.sleep(2)

    # Run standard SAGE cycle
    result = sage.cycle()
    return result

# ============================================================================
# Run SAGE Loop with Hybrid Learning
# ============================================================================

# Initialize dashboard
dashboard = StatusDashboard()
dashboard.update(
    state="🎧 LISTENING",
    user_input="(waiting for speech)",
    response="(no response yet)",
    stats=hybrid_system.get_stats()
)
dashboard.render()

time.sleep(2)  # Show initial dashboard

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
    print("\n\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)

    # Hybrid system stats
    stats = hybrid_system.get_stats()
    print(f"\n🧠 Hybrid Learning System:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Fast path: {stats['fast_path_hits']} ({stats['fast_path_ratio']:.1%})")
    print(f"   Slow path: {stats['slow_path_hits']} ({stats['slow_path_ratio']:.1%})")
    print(f"   Patterns learned: {stats['patterns_learned']}")
    print(f"   Total patterns: {stats['total_patterns']} (started with {stats['initial_patterns']})")

    if stats['total_queries'] > 0:
        improvement = stats['fast_path_ratio'] - (stats['initial_patterns'] / (stats['initial_patterns'] + 1))
        print(f"\n📈 Learning Progress:")
        print(f"   System became {improvement:.1%} faster through learning")

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
