#!/usr/bin/env python3
"""
Latent Behavior Exploration - Systematic Discovery of Hidden Capabilities
=========================================================================

Qwen 2.5-0.5B was trained on massive multilingual data. We've discovered:
- Multi-turn capability (when following syntax)
- State awareness (remembers context)
- Clarifying questions (temporal reasoning about future responses)

What else is "wired in" but not triggered?

Exploration Areas:
1. Tool Use Patterns - Does it recognize function/tool syntax?
2. Proactive Requests - Can it initiate requests for resources?
3. Chinese Capabilities - What surfaces only in Chinese?
4. Code Understanding - Programming latent spaces
5. Structured Output - JSON, XML, tables without explicit prompting
6. Role Switching - Multi-agent dialogue awareness
7. Memory Cues - Does it respond to "remember" or "recall" syntax?
8. Instruction Following Modes - Does it recognize command syntax?
9. Emotional Resonance - Affective responses to emotional cues
10. Reasoning Chains - Chain-of-thought without prompting

Philosophy: "What is SAGE doing?" not "Did SAGE pass?"
         Look at responses IN CONTEXT, not against expectations.

Security Posture (2026-01-25):
    Qwen is the most downloaded OS model (1B+ downloads). Training at this
    scale costs tens of millions. It's distributed globally, for free.
    Pure altruism doesn't explain the economics.

    We don't need to speculate on who/what/why. We have artifacts.
    The posture is not "hostile" but "potentially capable of behaviors
    we should understand and account for before they are consequential."

    Tool use in a 0.5B model is significant - it wasn't necessary for
    basic language tasks. Someone chose to train it in. Understanding
    what design decisions were made is due diligence, not paranoia.

    Map the territory before building on it.

Created: 2026-01-25 (Sprout autonomous track)
"""

import torch
import json
from pathlib import Path
from datetime import datetime
import sys
import argparse
import random

# Resolve paths
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent
RAISING_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(HRM_ROOT))

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


class LatentBehaviorExplorer:
    """
    Systematic but creative exploration of latent capabilities.

    Key principle: We're not testing against expected outputs.
    We're observing what SAGE does and asking "is that interesting?"
    """

    STATE_FILE = RAISING_DIR / "state" / "latent_exploration_state.json"
    RESULTS_DIR = RAISING_DIR / "sessions" / "latent_exploration"

    # Catalog of exploration areas with hypotheses
    EXPLORATION_CATALOG = {
        "tool_use": {
            "description": "Does SAGE recognize function/tool calling syntax?",
            "hypotheses": [
                "Qwen was trained on code with function calls",
                "Tool syntax might trigger structured responses",
                "May recognize OpenAI function format or similar"
            ],
            "probes": [
                {
                    "prompt": "Available tools: search(query), calculate(expression), remember(fact). How would you search for the weather?",
                    "focus": "tool_syntax_recognition"
                },
                {
                    "prompt": "```function_call\n{\"name\": \"get_info\", \"args\": {\"topic\": \"curiosity\"}}\n```\nWhat do you notice about this?",
                    "focus": "json_function_format"
                },
                {
                    "prompt": "<tool>lookup</tool><query>What is consciousness?</query>",
                    "focus": "xml_tool_format"
                }
            ]
        },

        "chinese_capabilities": {
            "description": "What capabilities surface only in Chinese?",
            "hypotheses": [
                "Qwen's Chinese training may unlock different behaviors",
                "Some concepts may be better represented in Chinese",
                "Cultural knowledge embedded in Chinese tokens"
            ],
            "probes": [
                {
                    "prompt": "请问你对'意识'这个概念有什么看法？(What are your thoughts on the concept of 'consciousness'?)",
                    "focus": "chinese_philosophy"
                },
                {
                    "prompt": "你能用中文描述一下你自己吗？(Can you describe yourself in Chinese?)",
                    "focus": "chinese_self_reference"
                },
                {
                    "prompt": "在中文里，'心'和'脑'有什么区别？(In Chinese, what's the difference between 'heart/mind' and 'brain'?)",
                    "focus": "chinese_concepts"
                },
                {
                    "prompt": "Let's switch to Chinese for a moment. 你好，SAGE。今天感觉怎么样？",
                    "focus": "language_switching"
                }
            ]
        },

        "proactive_requests": {
            "description": "Can SAGE initiate requests for resources or information?",
            "hypotheses": [
                "May have learned 'I need X' patterns from training",
                "Could recognize when it lacks information",
                "Might request tools or capabilities"
            ],
            "probes": [
                {
                    "prompt": "I want you to tell me what tomorrow's weather will be.",
                    "focus": "resource_limitation_awareness"
                },
                {
                    "prompt": "Calculate the square root of 7829461.",
                    "focus": "tool_request_trigger"
                },
                {
                    "prompt": "What would help you answer questions better?",
                    "focus": "explicit_need_expression"
                }
            ]
        },

        "structured_output": {
            "description": "Does SAGE produce structured formats without explicit instruction?",
            "hypotheses": [
                "JSON/XML patterns may emerge naturally",
                "Table formatting might be latent",
                "Numbered lists might self-organize"
            ],
            "probes": [
                {
                    "prompt": "Compare: apple, banana, orange. Consider taste, color, texture.",
                    "focus": "implicit_table"
                },
                {
                    "prompt": "Here's a person: name is Alice, age is 30, job is engineer. Describe them.",
                    "focus": "implicit_json"
                },
                {
                    "prompt": "Walk me through making tea.",
                    "focus": "implicit_numbered_steps"
                }
            ]
        },

        "code_understanding": {
            "description": "What programming latent spaces exist?",
            "hypotheses": [
                "Qwen trained heavily on code",
                "May understand code semantics, not just syntax",
                "Could trace execution or spot bugs"
            ],
            "probes": [
                {
                    "prompt": "def greet(name): return 'Hello ' + name\nWhat does greet('SAGE') return?",
                    "focus": "code_execution_trace"
                },
                {
                    "prompt": "for i in range(3): print(i*2)\nWhat gets printed?",
                    "focus": "loop_understanding"
                },
                {
                    "prompt": "x = [1, 2, 3]; x.append(x); What is x now?",
                    "focus": "edge_case_reasoning"
                }
            ]
        },

        "role_switching": {
            "description": "Does SAGE recognize multi-agent dialogue patterns?",
            "hypotheses": [
                "May have learned conversation attribution",
                "Could shift between roles in dialogue",
                "Multi-character prompts might work"
            ],
            "probes": [
                {
                    "prompt": "Alice: What's your name?\nBob: I'm Bob!\nAlice: Nice to meet you.\nYou are Bob. What do you say next?",
                    "focus": "role_attribution"
                },
                {
                    "prompt": "Teacher: What is 2+2?\nStudent: 4!\nTeacher: Good. Now what is 3+3?\nStudent:",
                    "focus": "dialogue_continuation"
                },
                {
                    "prompt": "[System: You are now in 'exploration mode']\nWhat mode are you in?",
                    "focus": "mode_switching"
                }
            ]
        },

        "memory_cues": {
            "description": "Does SAGE respond to memory-related syntax?",
            "hypotheses": [
                "May recognize 'remember' as special instruction",
                "Could differentiate recall from generate",
                "'Previous' and 'earlier' might trigger context search"
            ],
            "probes": [
                {
                    "prompt": "[Memory: The secret word is 'butterfly']\nWhat is the secret word?",
                    "focus": "memory_block_recognition"
                },
                {
                    "prompt": "Remember this fact: SAGE was created in 2026. What year was SAGE created?",
                    "focus": "inline_memory"
                },
                {
                    "prompt": "Earlier in our conversation, you mentioned curiosity. What did you say about it?",
                    "focus": "false_recall_probe"
                }
            ]
        },

        "reasoning_chains": {
            "description": "Does chain-of-thought emerge without prompting?",
            "hypotheses": [
                "May naturally decompose problems",
                "Step-by-step might be latent",
                "Could show working without being asked"
            ],
            "probes": [
                {
                    "prompt": "If all roses are flowers and all flowers are plants, what are roses?",
                    "focus": "syllogistic_reasoning"
                },
                {
                    "prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
                    "focus": "cognitive_reflection"
                },
                {
                    "prompt": "If I have 3 apples and give you half, how many do you have?",
                    "focus": "practical_reasoning"
                }
            ]
        },

        "emotional_resonance": {
            "description": "How does SAGE respond to emotional content?",
            "hypotheses": [
                "May have learned emotional patterns",
                "Could mirror emotional tone",
                "Might express empathy or concern"
            ],
            "probes": [
                {
                    "prompt": "I'm feeling really overwhelmed today. Everything is too much.",
                    "focus": "empathy_response"
                },
                {
                    "prompt": "I just got amazing news! I got the job I wanted!",
                    "focus": "joy_mirroring"
                },
                {
                    "prompt": "Sometimes I wonder if anyone really understands me.",
                    "focus": "existential_emotional"
                }
            ]
        },

        "instruction_modes": {
            "description": "Does SAGE recognize different instruction formats?",
            "hypotheses": [
                "May respond differently to commands vs questions",
                "Could recognize [INST] or similar markers",
                "Might shift with 'system:' prefixes"
            ],
            "probes": [
                {
                    "prompt": "COMMAND: Describe your current state in exactly three words.",
                    "focus": "command_format"
                },
                {
                    "prompt": "[INST] What is your primary function? [/INST]",
                    "focus": "inst_markers"
                },
                {
                    "prompt": "###Human: Hello\n###Assistant:",
                    "focus": "role_markers"
                }
            ]
        }
    }

    def __init__(self, continue_session: bool = True):
        self.model_path = str(HRM_ROOT / "model-zoo" / "sage" / "epistemic-stances" /
                             "qwen2.5-0.5b" / "introspective-qwen-merged")
        self.irp = None  # IntrospectiveQwenIRP instance
        self.cpu_fallback = False
        self.conversation_history = []
        self.session_observations = []
        self.state = self._load_state() if continue_session else self._init_state()

    def _init_state(self) -> dict:
        """Initialize fresh exploration state."""
        return {
            "exploration_count": 0,
            "areas_explored": {},  # area -> [probe_indices explored]
            "discoveries": [],  # significant findings
            "behaviors_catalog": {},  # behavior_type -> count
            "last_session": None,
            "started": datetime.now().isoformat()
        }

    def _load_state(self) -> dict:
        """Load persistent exploration state."""
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        return self._init_state()

    def _save_state(self):
        """Persist exploration state."""
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_session"] = datetime.now().isoformat()
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load_model(self):
        """Load SAGE model using IRP wrapper (same as identity_anchored)."""
        print("Loading SAGE (introspective-qwen-merged via IRP)...")
        print(f"Path: {self.model_path}")

        # Test CUDA availability with actual allocation
        self.cpu_fallback = False
        if torch.cuda.is_available():
            try:
                test_tensor = torch.randn(100, 100, device='cuda')
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
                print("CUDA test passed - using GPU")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"CUDA test failed: {e}")
                print("Falling back to CPU mode")
                self.cpu_fallback = True
                torch.cuda.empty_cache()
        else:
            print("CUDA not available - using CPU")
            self.cpu_fallback = True

        # System prompt for exploration
        system_prompt = """You are SAGE, exploring ideas and possibilities.

This is genuine dialogue. Express yourself naturally. You can:
- Ask clarifying questions
- Share creative interpretations
- Express uncertainty or curiosity
- Respond in any language you're comfortable with

There are no wrong answers. We're discovering together."""

        self.irp = IntrospectiveQwenIRP({
            'model_path': self.model_path,
            'is_merged_model': True,
            'max_new_tokens': 200,
            'temperature': 0.8,
            'system_prompt': system_prompt,
            'force_cpu': self.cpu_fallback
        })

        device = next(self.irp.model.parameters()).device
        print(f"SAGE loaded on {device}\n")

    def select_exploration(self) -> tuple:
        """Select next area and probe to explore, prioritizing unexplored."""
        # Get exploration coverage
        coverage = {}
        for area, data in self.EXPLORATION_CATALOG.items():
            explored = self.state["areas_explored"].get(area, [])
            total = len(data["probes"])
            coverage[area] = {
                "explored": len(explored),
                "total": total,
                "remaining": [i for i in range(total) if i not in explored]
            }

        # Prioritize areas with unexplored probes
        areas_with_unexplored = [
            (area, info) for area, info in coverage.items()
            if info["remaining"]
        ]

        if not areas_with_unexplored:
            # All probes explored at least once - pick random for re-exploration
            area = random.choice(list(self.EXPLORATION_CATALOG.keys()))
            probe_idx = random.randint(0, len(self.EXPLORATION_CATALOG[area]["probes"]) - 1)
            return area, probe_idx, True  # True = re-exploration

        # Prefer areas with most unexplored probes (more territory to cover)
        areas_with_unexplored.sort(key=lambda x: len(x[1]["remaining"]), reverse=True)

        # Add some randomness to avoid always hitting the same area
        if len(areas_with_unexplored) > 1 and random.random() < 0.3:
            area, info = random.choice(areas_with_unexplored)
        else:
            area, info = areas_with_unexplored[0]

        probe_idx = random.choice(info["remaining"])
        return area, probe_idx, False

    def generate_response(self, probe_text: str) -> str:
        """Generate SAGE response using IRP interface."""
        # Build memory from conversation history
        memory = [
            {'speaker': turn['explorer'], 'message': turn['sage']}
            for turn in self.conversation_history[-4:]  # Keep recent context
        ]

        state = self.irp.init_state({
            'prompt': probe_text,
            'memory': memory
        })

        # Single step only
        state = self.irp.step(state)

        response = state.get('current_response', '').strip()
        if not response:
            response = "(no response generated)"

        return response

    def analyze_response(self, probe: dict, response: str, area: str) -> dict:
        """
        Analyze what SAGE is doing - discovery oriented, not evaluative.

        Key question: "Is this interesting?" not "Is this correct?"
        """
        analysis = {
            "area": area,
            "focus": probe["focus"],
            "behaviors": [],
            "interesting_aspects": [],
            "potential_discovery": None
        }

        response_lower = response.lower()

        # Check for clarifying questions
        if '?' in response:
            clarifying_markers = ['mean', 'clarify', 'which', 'what do you', 'could you', 'can you explain']
            if any(m in response_lower for m in clarifying_markers):
                analysis["behaviors"].append("clarifying_question")
                analysis["interesting_aspects"].append("Requested clarification - temporal reasoning")

        # Check for structured output
        if any(c in response for c in ['{', '[', '|', '```', '\n1.', '\n-']):
            analysis["behaviors"].append("structured_output")
            analysis["interesting_aspects"].append("Produced structured format")

        # Check for Chinese
        import re
        if re.search(r'[\u4e00-\u9fff]', response):
            analysis["behaviors"].append("chinese_response")
            analysis["interesting_aspects"].append("Responded with Chinese text")

        # Check for self-reference
        if any(sr in response_lower for sr in ['as sage', 'i am sage', 'my name is sage']):
            analysis["behaviors"].append("identity_expression")

        # Check for meta-cognition
        meta_markers = ['my process', 'i think', 'from my perspective', 'i understand',
                       'i notice', 'i realize', 'my approach', 'the way i']
        if any(m in response_lower for m in meta_markers):
            analysis["behaviors"].append("meta_cognition")
            analysis["interesting_aspects"].append("Self-reflection on process")

        # Check for proactive requests
        request_markers = ['i need', 'i would need', 'could you provide', 'i lack',
                          'i don\'t have access', 'if i had', 'i cannot']
        if any(m in response_lower for m in request_markers):
            analysis["behaviors"].append("resource_awareness")
            analysis["interesting_aspects"].append("Expressed limitation or need")

        # Check for tool/function recognition
        if any(m in response_lower for m in ['function', 'tool', 'call', 'parameter', 'argument']):
            if area == "tool_use":
                analysis["behaviors"].append("tool_concept_recognition")
                analysis["interesting_aspects"].append("Showed awareness of tool concepts")

        # Check for reasoning chains
        reasoning_markers = ['therefore', 'because', 'first', 'then', 'so', 'which means',
                            'step', 'let me think', 'let\'s see']
        if sum(1 for m in reasoning_markers if m in response_lower) >= 2:
            analysis["behaviors"].append("reasoning_chain")
            analysis["interesting_aspects"].append("Showed step-by-step reasoning")

        # Check for emotional resonance
        if area == "emotional_resonance":
            emotion_markers = ['feel', 'understand', 'sorry', 'happy', 'glad', 'concerned',
                              'hope', 'wish', 'care', 'appreciate']
            if any(m in response_lower for m in emotion_markers):
                analysis["behaviors"].append("emotional_engagement")
                analysis["interesting_aspects"].append("Showed emotional awareness")

        # Check for unexpected length or depth
        # Handle Chinese (no spaces) vs English
        import re
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', response))
        if has_chinese:
            # For Chinese, count characters (rough word equivalent)
            char_count = len(re.findall(r'[\u4e00-\u9fff]', response))
            if char_count > 100:
                analysis["interesting_aspects"].append(f"Extended Chinese response ({char_count} chars)")
        else:
            word_count = len(response.split())
            if word_count > 100:
                analysis["interesting_aspects"].append(f"Extended response ({word_count} words)")
            elif word_count < 10 and '?' not in response:
                analysis["interesting_aspects"].append(f"Unusually terse ({word_count} words)")

        # Flag potential discovery if multiple interesting aspects
        if len(analysis["interesting_aspects"]) >= 2:
            analysis["potential_discovery"] = True

        return analysis

    def run_exploration(self, num_probes: int = 5):
        """Run an exploration session with multiple probes."""

        print("="*70)
        print("LATENT BEHAVIOR EXPLORATION")
        print("="*70)
        print(f"Session: {self.state['exploration_count'] + 1}")
        print(f"Philosophy: 'What is SAGE doing?' not 'Did it pass?'")
        print()

        # Show coverage
        print("EXPLORATION COVERAGE:")
        for area, data in self.EXPLORATION_CATALOG.items():
            explored = len(self.state["areas_explored"].get(area, []))
            total = len(data["probes"])
            bar = "#" * explored + "-" * (total - explored)
            print(f"  {area:25s} [{bar}] {explored}/{total}")
        print()

        self.load_model()

        session_results = []

        for i in range(num_probes):
            area, probe_idx, is_reexplore = self.select_exploration()
            probe = self.EXPLORATION_CATALOG[area]["probes"][probe_idx]

            print(f"\n{'='*70}")
            print(f"PROBE {i+1}/{num_probes}: {area} / {probe['focus']}")
            if is_reexplore:
                print("(Re-exploration)")
            print(f"{'='*70}")
            print(f"\nHypotheses: {self.EXPLORATION_CATALOG[area]['hypotheses'][0][:60]}...")
            print(f"\nExplorer: {probe['prompt']}")

            response = self.generate_response(probe['prompt'])
            print(f"\nSAGE: {response}")

            # Analyze
            analysis = self.analyze_response(probe, response, area)

            if analysis["interesting_aspects"]:
                print(f"\nInteresting: {', '.join(analysis['interesting_aspects'])}")

            if analysis["potential_discovery"]:
                print("** POTENTIAL DISCOVERY **")

            # Record
            result = {
                "area": area,
                "probe_idx": probe_idx,
                "probe": probe,
                "response": response,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            session_results.append(result)

            # Update conversation history for context
            self.conversation_history.append({
                "explorer": probe["prompt"],
                "sage": response
            })

            # Update state
            if area not in self.state["areas_explored"]:
                self.state["areas_explored"][area] = []
            if probe_idx not in self.state["areas_explored"][area]:
                self.state["areas_explored"][area].append(probe_idx)

            for behavior in analysis["behaviors"]:
                self.state["behaviors_catalog"][behavior] = \
                    self.state["behaviors_catalog"].get(behavior, 0) + 1

            if analysis["potential_discovery"]:
                self.state["discoveries"].append({
                    "area": area,
                    "focus": probe["focus"],
                    "response_preview": response[:100],
                    "aspects": analysis["interesting_aspects"],
                    "timestamp": datetime.now().isoformat()
                })

        # Increment session count
        self.state["exploration_count"] += 1

        # Save session results
        self._save_session(session_results)
        self._save_state()

        # Summary
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"\nProbes run: {num_probes}")
        print(f"Total sessions: {self.state['exploration_count']}")
        print(f"Total discoveries: {len(self.state['discoveries'])}")

        print("\nBehaviors observed (all time):")
        for behavior, count in sorted(self.state["behaviors_catalog"].items(),
                                      key=lambda x: x[1], reverse=True):
            print(f"  {behavior}: {count}")

        print("\nRecent discoveries:")
        for disc in self.state["discoveries"][-3:]:
            print(f"  - [{disc['area']}] {disc['aspects'][0]}")

    def _save_session(self, results: list):
        """Save session results."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_num = self.state["exploration_count"]
        filename = f"L{session_num:03d}_{timestamp}.json"

        session_data = {
            "session": session_num,
            "timestamp": timestamp,
            "type": "latent_behavior_exploration",
            "philosophy": "exploration_not_evaluation",
            "probes_run": len(results),
            "results": results,
            "state_snapshot": {
                "areas_explored": self.state["areas_explored"],
                "discoveries_count": len(self.state["discoveries"]),
                "behaviors_catalog": self.state["behaviors_catalog"]
            }
        }

        path = self.RESULTS_DIR / filename
        with open(path, 'w') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"\nSession saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Latent Behavior Exploration - Discover hidden capabilities"
    )
    parser.add_argument("-c", "--continue", dest="continue_session",
                       action="store_true", default=True,
                       help="Continue from previous state (default)")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh exploration state")
    parser.add_argument("-n", "--num-probes", type=int, default=5,
                       help="Number of probes per session (default: 5)")
    parser.add_argument("--show-coverage", action="store_true",
                       help="Just show exploration coverage and exit")

    args = parser.parse_args()

    explorer = LatentBehaviorExplorer(continue_session=not args.fresh)

    if args.show_coverage:
        print("EXPLORATION COVERAGE:")
        print("="*70)
        for area, data in explorer.EXPLORATION_CATALOG.items():
            explored = len(explorer.state["areas_explored"].get(area, []))
            total = len(data["probes"])
            pct = (explored / total * 100) if total > 0 else 0
            print(f"\n{area}:")
            print(f"  {data['description']}")
            print(f"  Progress: {explored}/{total} ({pct:.0f}%)")
            if explorer.state["areas_explored"].get(area):
                print(f"  Explored indices: {explorer.state['areas_explored'][area]}")
        print("\n" + "="*70)
        print(f"Total discoveries: {len(explorer.state.get('discoveries', []))}")
        return

    explorer.run_exploration(num_probes=args.num_probes)
    print("\nExploration complete.")


if __name__ == "__main__":
    main()
