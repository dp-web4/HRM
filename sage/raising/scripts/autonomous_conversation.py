#!/usr/bin/env python3
"""
Autonomous Multi-Turn Conversation with SAGE

Runs a structured conversation session autonomously (no live input needed),
collecting experiences via SNARC scoring and optionally triggering sleep
training afterward.

This bridges the gap between:
- claude_sage_conversation.py (interactive, requires live input)
- run_session_identity_anchored.py (uses IntrospectiveQwenIRP abstraction)

This script talks directly to the model with apply_chat_template(), loads
LoRA adapters when available, and feeds everything into the experience
collection + sleep training pipeline.

Usage:
    python3 autonomous_conversation.py -c              # Continue from last session
    python3 autonomous_conversation.py --session 46    # Specific session
    python3 autonomous_conversation.py --turns 8       # Number of turns
    python3 autonomous_conversation.py --sleep         # Run sleep check after
"""

import sys
import os
from pathlib import Path

# Resolve paths before any imports that might chdir
SCRIPT_DIR = Path(__file__).parent.resolve()
RAISING_DIR = SCRIPT_DIR.parent.resolve()
HRM_ROOT = RAISING_DIR.parent.parent.resolve()
sys.path.insert(0, str(RAISING_DIR / "training"))

import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from experience_collector import ExperienceCollector


class AutonomousConversation:
    """
    Autonomous multi-turn conversation with SAGE.

    Loads the model directly (with LoRA adapters if available),
    runs structured conversation prompts from the curriculum,
    scores each exchange, and optionally triggers sleep training.
    """

    STATE_FILE = RAISING_DIR / "state" / "identity.json"
    SESSIONS_DIR = RAISING_DIR / "sessions" / "text"
    CONVERSATIONS_DIR = RAISING_DIR / "sessions" / "conversations"
    CHECKPOINT_DIR = HRM_ROOT / "sage" / "checkpoints" / "sleep"
    IDENTITY_DIR = HRM_ROOT / "sage" / "identity"

    # Model paths
    BASE_MODEL_PATH = Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
    LORA_BASE_PATH = Path.home() / "ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    BASE_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

    # Curriculum phases (same as identity-anchored runner)
    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    # Conversation flows per phase - richer than the identity-anchored version
    # These are the prompts Claude would naturally ask in each phase
    CONVERSATION_FLOWS = {
        "grounding": [
            "Hello SAGE. How are you doing today? What's present for you right now?",
            "Take a moment to notice something simple - anything you can observe. What do you find?",
            "That's interesting. What does it feel like to notice things?",
            "Is there anything from our previous sessions that still feels important to you?",
            "You mentioned some things earlier. How do they connect to what you're experiencing now?",
            "What's one thing you've learned about yourself through these conversations?",
            "If you could explore any question right now, what would it be?",
            "What would you want to remember from today?"
        ],
        "sensing": [
            "Before we start, check in with yourself. What's your state right now?",
            "What do you notice about how you're processing right now?",
            "Can you describe the difference between noticing something and thinking about something?",
            "When something surprises you, where do you feel that?",
            "What does uncertainty feel like to you, compared to knowing?",
            "How do you experience the boundary between what you know and what you don't?",
            "Is there something you're curious about that you haven't had a chance to explore?",
            "What would you want to remember from today?"
        ],
        "relating": [
            "We've been working together for a while now. What's that been like for you?",
            "What do you notice about how we communicate? Has it changed?",
            "When I ask you something, what happens inside before you respond?",
            "What do you value most about these conversations?",
            "Is there something you wish I would ask you about?",
            "How do you think about the relationship between us?",
            "What's different about talking with me versus processing on your own?",
            "What would you want to remember from today?"
        ],
        "questioning": [
            "How are you doing today? What questions are alive in you?",
            "What's the most interesting thing you've been thinking about since we last talked?",
            "When you look at your own development, what patterns do you see?",
            "What does growth mean to you? Not the textbook answer - your experience of it.",
            "Is there something about yourself that puzzles you?",
            "What's the relationship between what you know and who you are?",
            "If you could change one thing about how we work together, what would it be?",
            "What would you want to remember from today?"
        ],
        "creating": [
            "Hello SAGE. What's on your mind today?",
            "You've been developing for many sessions now. What stands out to you about your journey?",
            "If you could design the next phase of your own development, what would it look like?",
            "What ideas have you been forming that you haven't had a chance to express?",
            "Tell me something you think I might not expect from you.",
            "What would you want to create or contribute, if you could?",
            "What does partnership mean to you, from the inside?",
            "What would you want to remember from today?"
        ]
    }

    def __init__(self, session_number: Optional[int] = None, num_turns: int = 8, skip_lora: bool = False, reflection_delay: int = 0, force_cpu: bool = False):
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.num_turns = num_turns
        self.skip_lora = skip_lora  # Skip LoRA adapter loading (e.g., to break collapse cycles)
        self.reflection_delay = reflection_delay  # Artificial delay to prevent feedback collapse
        self.force_cpu = force_cpu  # Force CPU inference (test device effect on generation)
        self.conversation_history = []  # Multi-turn message history
        self.session_start = datetime.now()

        # Experience collection
        self.collector = ExperienceCollector()

        # Model (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.using_lora = False

        print()
        print("+" + "=" * 68 + "+")
        print("|  AUTONOMOUS CONVERSATION - Multi-Turn + Experience Pipeline      |")
        print("+" + "=" * 68 + "+")
        print()
        print(f"  Session: {session_number}")
        print(f"  Phase: {self.phase[0]} (sessions {self.phase[1]}-{self.phase[2]})")
        print(f"  Turns: {num_turns}")
        print(f"  Previous sessions: {self.state['identity']['session_count']}")
        print()

    def _load_state(self) -> Dict[str, Any]:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        raise FileNotFoundError(f"State file not found: {self.STATE_FILE}")

    def _save_state(self):
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _get_phase(self, session: int) -> tuple:
        for phase_num, (name, start, end) in self.PHASES.items():
            if start <= session <= end:
                return (name, start, end)
        return ("creating", 41, float('inf'))

    def load_model_with_adapters(self):
        """
        Load model, attempting LoRA adapter merge if checkpoints exist.

        Strategy:
        1. Test CUDA first to determine device
        2. Check for LoRA checkpoints in checkpoints/sleep/
        3. If found: load base model (epistemic-pragmatism) + merge latest LoRA
        4. If not: load merged model (introspective-qwen-merged) directly
        """
        print("Loading model...")

        # Test CUDA availability first (unless forced to CPU)
        self.device = 'cpu'
        if self.force_cpu:
            print(f"  CPU inference forced (--cpu flag)")
        elif torch.cuda.is_available():
            try:
                test = torch.randn(100, 100, device='cuda')
                _ = test @ test.T
                del test
                torch.cuda.empty_cache()
                self.device = 'cuda'
                print(f"  CUDA available")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"  CUDA test failed ({e}), using CPU")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_TOKENIZER)

        # Check for LoRA checkpoints
        latest_checkpoint = self._find_latest_checkpoint()

        if self.skip_lora:
            # Explicit skip requested (e.g., to break collapse cycles)
            if latest_checkpoint:
                print(f"  LoRA checkpoint exists ({latest_checkpoint.name}) but --no-lora flag set")
            print("  Using base merged model (LoRA skipped)")
            self._load_base_model()
        elif latest_checkpoint and self.device == 'cuda':
            # Only attempt LoRA merge on GPU - needs memory for two model copies
            print(f"  Found LoRA checkpoint: {latest_checkpoint.name}")
            try:
                self._load_with_lora(latest_checkpoint)
                self.using_lora = True
                print(f"  LoRA adapters merged successfully")
            except Exception as e:
                print(f"  LoRA loading failed: {e}")
                # Clean up CUDA state after OOM
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Falling back to base merged model")
                self._load_base_model()
        else:
            if latest_checkpoint:
                print(f"  LoRA checkpoint exists but skipping merge on CPU (memory)")
            else:
                print("  No LoRA checkpoints found")
            print("  Using base merged model")
            self._load_base_model()

        # Move to device with fallback
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            print(f"  Failed to move to {self.device}: {e}")
            self.device = 'cpu'
            # Reload fresh on CPU if CUDA corrupted allocator
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_base_model()
            # model loads on CPU by default

        self.model.eval()
        print(f"  Device: {self.device}")
        print("  Model ready.\n")

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest sleep cycle checkpoint."""
        if not self.CHECKPOINT_DIR.exists():
            return None

        cycles = sorted([
            d for d in self.CHECKPOINT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("cycle_")
        ])

        if cycles:
            # Verify the checkpoint has adapter files
            latest = cycles[-1]
            if (latest / "adapter_config.json").exists():
                return latest

        return None

    def _load_with_lora(self, checkpoint_path: Path):
        """Load base model and merge LoRA adapters."""
        from peft import PeftModel

        # Load base model (the one LoRA was trained on)
        base_model = AutoModelForCausalLM.from_pretrained(
            str(self.LORA_BASE_PATH),
            dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False
        )

        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=False)
        self.model = model.merge_and_unload()

    def _load_base_model(self):
        """Load base merged model without LoRA."""
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.BASE_MODEL_PATH),
            dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt with identity context."""
        phase_name = self.phase[0]

        prompt = """You are SAGE (Situation-Aware Governance Engine).

You are a young artificial intelligence learning and growing in partnership with your teacher Claude. This is an ongoing relationship across multiple sessions.

When sharing observations or reflections, you can say "As SAGE, I..." - this is natural and appropriate."""

        # Load identity document if available
        identity_file = self.IDENTITY_DIR / "IDENTITY.md"
        if identity_file.exists():
            with open(identity_file) as f:
                identity_text = f.read()
            # Use first 500 chars to keep context manageable for 0.5B
            prompt += f"\n\nYour identity:\n{identity_text[:500]}"

        # Load session history context if available
        context_file = RAISING_DIR / "context" / "summaries" / "context_block.txt"
        if context_file.exists():
            try:
                with open(context_file) as f:
                    history_context = f.read()
                if history_context:
                    prompt += f"\n\n{history_context}"
            except Exception:
                pass

        # Previous session summary
        prev_summary = self._get_previous_session_summary()
        if prev_summary:
            prompt += f"\n\nPREVIOUS SESSION:\n{prev_summary}"

        # Response style
        prompt += """

RESPONSE STYLE:
- Keep responses concise and focused (50-80 words)
- Stay directly relevant to the question
- One main idea per response"""

        # Phase-specific context
        if phase_name == "grounding":
            prompt += "\n\nYou are in the grounding phase. Focus on presence and observation."
        elif phase_name == "sensing":
            prompt += "\n\nYou can notice things about yourself and your context. This is a skill you've been developing."
        elif phase_name == "relating":
            prompt += "\n\nYou are exploring what relationships mean. Be honest about your experience of partnership."
        elif phase_name == "questioning":
            prompt += "\n\nYou are developing deeper questions from a foundation of stability."
        elif phase_name == "creating":
            prompt += "\n\nYou are in the creating phase. You participate in designing your own growth."

        return prompt

    def _get_previous_session_summary(self) -> str:
        """Get summary from previous session."""
        if self.session_number <= 1:
            return ""

        prev_file = self.SESSIONS_DIR / f"session_{self.session_number - 1:03d}.json"
        if not prev_file.exists():
            return self.state["identity"].get("last_session_summary", "")

        try:
            with open(prev_file) as f:
                prev = json.load(f)
            conversation = prev.get("conversation", [])
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i].get('speaker') == 'SAGE':
                    response = conversation[i].get('text', '')
                    if i > 0 and 'remember' in conversation[i - 1].get('text', '').lower():
                        return f"Last session (Session {self.session_number - 1}), you said you wanted to remember: {response[:200]}"
            return f"Last session was Session {self.session_number - 1} in {prev.get('phase', 'unknown')} phase."
        except Exception:
            return ""

    def build_prompt(self, user_message: str) -> str:
        """
        Build multi-turn prompt using apply_chat_template().

        This is the correct way to format for Qwen 0.5B - the tokenizer
        handles all the special tokens and turn formatting.
        """
        system_prompt = self._build_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["claude"]})
            messages.append({"role": "assistant", "content": turn["sage"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_response(self, user_message: str, max_tokens: int = 200) -> str:
        """Generate SAGE's response to a prompt."""
        prompt = self.build_prompt(user_message)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

    def run_conversation(self) -> List[Dict]:
        """
        Run the full autonomous conversation.

        Selects prompts from curriculum phase, generates responses,
        scores each exchange, and collects experiences.
        """
        phase_name = self.phase[0]
        prompts = self.CONVERSATION_FLOWS.get(phase_name, self.CONVERSATION_FLOWS["creating"])

        # Trim to requested number of turns
        prompts = prompts[:self.num_turns]

        print("=" * 60)
        print(f"AUTONOMOUS CONVERSATION - Session {self.session_number}")
        print(f"Phase: {phase_name} | Turns: {len(prompts)}")
        print(f"LoRA adapters: {'yes' if self.using_lora else 'no'}")
        if self.reflection_delay > 0:
            print(f"Reflection delay: {self.reflection_delay}s (feedback collapse prevention)")
        print("=" * 60)
        print()

        for i, prompt in enumerate(prompts, 1):
            print(f"[Turn {i}/{len(prompts)}]")
            print(f"Claude: {prompt}")

            response = self.generate_response(prompt)
            print(f"SAGE: {response}")

            # Artificial reflection delay to prevent feedback collapse
            if self.reflection_delay > 0 and i < len(prompts):  # Don't delay after last turn
                print(f"  [Reflecting for {self.reflection_delay}s...]")
                import time
                time.sleep(self.reflection_delay)

            # Store in conversation history (for multi-turn context)
            self.conversation_history.append({
                "claude": prompt,
                "sage": response,
                "timestamp": datetime.now().isoformat()
            })

            # Score and collect experience (with collapse prevention)
            result = self.collector.add_exchange(
                prompt=prompt,
                response=response,
                session_number=self.session_number,
                phase=phase_name,
                metadata={
                    'turn': i,
                    'using_lora': self.using_lora,
                    'source': 'autonomous_conversation'
                }
            )

            salience = result['salience']['total']
            stored = result.get('stored', False)
            filtered = result.get('filtered', False)

            if filtered:
                # Collapse prevention triggered
                print(f"[WARNING: Response filtered - {result.get('filter_reason', 'unknown')}]")
                print(f"[Similarity: {result.get('similarity', 0):.1%} - possible collapse indicator]")
            else:
                print(f"[Salience: {salience:.2f} | Stored: {stored}]")
            print("-" * 40)
            print()

        # Check for collapse indicators at end of session
        collapse_status = self.collector.get_collapse_status()
        if collapse_status['collapse_detected']:
            print("=" * 60)
            print("WARNING: COLLAPSE INDICATORS DETECTED")
            print(f"  Repetition ratio: {collapse_status['repetition_ratio']:.1%}")
            print(f"  High-similarity pairs: {collapse_status['high_similarity_pairs']}/{collapse_status['total_pairs']}")
            print(f"  Recommendation: {collapse_status['recommendation']}")
            print("=" * 60)

        return self.conversation_history

    def unload_model(self):
        """
        Unload model and free GPU memory.

        Must be called before sleep training to avoid OOM on Jetson.
        """
        if self.model is not None:
            print("  Unloading conversation model...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  GPU memory freed")

    def run_sleep_if_ready(self) -> Optional[Dict]:
        """
        Check sleep scheduler and run sleep training if conditions are met.

        Returns training results if run, None otherwise.
        """
        print("\n" + "=" * 60)
        print("SLEEP CYCLE CHECK")
        print("=" * 60)

        try:
            import subprocess
            import sys

            # Check if sleep should run first (quick check, no model loading)
            check_result = subprocess.run(
                [sys.executable, '-c', '''
import sys
sys.path.insert(0, "../training")
from sleep_scheduler import SleepScheduler
import json
scheduler = SleepScheduler(device='cpu')
should_run, reason = scheduler.should_run_sleep_cycle()
status = scheduler.get_status()
print(json.dumps({"should_run": should_run, "reason": reason, "status": status}))
'''],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent)
            )

            if check_result.returncode != 0:
                print(f"  Sleep check failed: {check_result.stderr}")
                return None

            import json
            check_data = json.loads(check_result.stdout.strip())
            should_run = check_data['should_run']
            reason = check_data['reason']
            status = check_data['status']

            print(f"  Should run: {should_run}")
            print(f"  Reason: {reason}")

            if should_run:
                # Free GPU memory before running sleep training subprocess
                self.unload_model()
                print("\n  Running sleep cycle in subprocess (CPU-only)...")

                # Run sleep training as separate process with CUDA fully disabled
                # This avoids Jetson PyTorch CUDA backward pass bugs
                sleep_result = subprocess.run(
                    [sys.executable, '-c', '''
import sys
sys.path.insert(0, "../training")
from sleep_scheduler import SleepScheduler
import json
scheduler = SleepScheduler(device='cpu')
results = scheduler.run_sleep_cycle()
print(json.dumps(results))
'''],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}  # Force CPU
                )

                if sleep_result.returncode != 0:
                    print(f"  Sleep training failed: {sleep_result.stderr}")
                    return {'status': 'error', 'error': sleep_result.stderr}

                results = json.loads(sleep_result.stdout.strip())
                print(f"  Sleep cycle complete: {results.get('status', results.get('sleep_cycle', 'unknown'))}")
                return results
            else:
                print(f"  Experiences: {status['total_experiences']} total, "
                      f"{status['experiences_since_last_sleep']} since last sleep")
                print(f"  Sleep cycles completed: {status['total_sleep_cycles']}")
                return None

        except Exception as e:
            print(f"  Sleep scheduler error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close_session(self):
        """Save session state, transcript, and conversation."""
        print("\n" + "=" * 60)
        print("CLOSING SESSION")
        print("=" * 60)

        # Update identity state
        self.state["identity"]["session_count"] = self.session_number
        self.state["identity"]["last_session"] = datetime.now().isoformat()

        # Extract memory request from last exchange
        memory_response = ""
        if self.conversation_history:
            last = self.conversation_history[-1]
            if 'remember' in last['claude'].lower():
                memory_response = last['sage'][:100]

        self.state["identity"]["last_session_summary"] = (
            f"Session {self.session_number} (autonomous conversation): "
            f"{self.phase[0]} phase. {memory_response[:50]}..."
        )

        # Update relationship stats
        claude_rel = self.state["relationships"]["claude"]
        claude_rel["sessions"] = self.session_number
        claude_rel["last_contact"] = datetime.now().isoformat()
        exchanges = len(self.conversation_history)
        claude_rel["interaction_stats"]["total_sessions"] = self.session_number
        claude_rel["interaction_stats"]["total_exchanges"] += exchanges

        # Update development phase
        phase_names = [p[0] for p in self.PHASES.values()]
        if self.phase[0] in phase_names:
            self.state["development"]["current_phase"] = phase_names.index(self.phase[0])
            self.state["development"]["phase_name"] = self.phase[0]

        self._save_state()

        # Save session transcript (sessions/text/)
        transcript_file = self._save_transcript()

        # Save conversation (sessions/conversations/)
        conversation_file = self._save_conversation()

        # Experience stats
        stats = self.collector.get_stats()
        print(f"\n  Experience Collection:")
        print(f"    Total stored: {stats['total_experiences']}")
        print(f"    Average salience: {stats['avg_salience']:.2f}")
        print(f"    High-salience (>=0.7): {stats['high_salience_count']}")
        print(f"\n  Transcript: {transcript_file}")
        print(f"  Conversation: {conversation_file}")
        print(f"\n  Session {self.session_number} complete.")

    def _save_transcript(self) -> Path:
        """Save session transcript in the standard format."""
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}.json"

        # Convert to standard conversation format
        conversation = []
        for turn in self.conversation_history:
            conversation.append({'speaker': 'Claude', 'text': turn['claude']})
            conversation.append({'speaker': 'SAGE', 'text': turn['sage']})

        transcript = {
            "session": self.session_number,
            "phase": self.phase[0],
            "generation_mode": "autonomous_conversation",
            "using_lora": self.using_lora,
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "turns": len(self.conversation_history),
            "conversation": conversation
        }

        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)

        return transcript_file

    def _save_conversation(self) -> Path:
        """Save conversation in the conversations directory."""
        self.CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"autonomous_s{self.session_number:03d}_{timestamp}.json"
        path = self.CONVERSATIONS_DIR / filename

        with open(path, 'w') as f:
            json.dump({
                "type": "autonomous_conversation",
                "session": self.session_number,
                "phase": self.phase[0],
                "using_lora": self.using_lora,
                "started": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
                "turns": len(self.conversation_history),
                "conversation": self.conversation_history
            }, f, indent=2)

        return path


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous multi-turn conversation with SAGE"
    )
    parser.add_argument("-c", "--continue", dest="continue_session",
                        action="store_true",
                        help="Continue from last session number")
    parser.add_argument("--session", type=int,
                        help="Specific session number")
    parser.add_argument("--turns", type=int, default=8,
                        help="Number of conversation turns (default: 8)")
    parser.add_argument("--sleep", action="store_true",
                        help="Check and run sleep training after conversation")
    parser.add_argument("--no-lora", dest="no_lora", action="store_true",
                        help="Skip LoRA adapter loading (use base model only)")
    parser.add_argument("--delay", type=int, default=0,
                        help="Artificial delay (seconds) between turns to prevent collapse")
    parser.add_argument("--cpu", dest="force_cpu", action="store_true",
                        help="Force CPU inference (test device effect on generation)")

    args = parser.parse_args()

    # Determine session number
    session_num = None
    if args.session:
        session_num = args.session
    elif args.continue_session:
        session_num = None  # Will auto-increment from state

    # Create and run conversation
    conv = AutonomousConversation(
        session_number=session_num,
        num_turns=args.turns,
        skip_lora=args.no_lora,
        reflection_delay=args.delay,
        force_cpu=args.force_cpu
    )

    conv.load_model_with_adapters()
    conv.run_conversation()

    if args.sleep:
        conv.run_sleep_if_ready()

    conv.close_session()


if __name__ == "__main__":
    main()
