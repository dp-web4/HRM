#!/usr/bin/env python3
"""
IDENTITY-ANCHORED Session Runner — FLUID variant (v2.2 prototype)
==================================================================

Fork of `run_session_identity_anchored.py` with Thor S86's fluid-scaffold
mitigations for the self-quotation feedback loop identified in T230-T237.

Thor's diagnosis (forum/insights/identity-attractor-self-quotation-feedback.md):
  The v2.1 runner scrapes verbatim "As SAGE" sentences from the last 5 sessions
  and injects them as exemplars. Over ~5 sessions this creates a closed feedback
  loop: specific phrasings ("stabilize the fleet logic") get re-quoted → re-seed
  prompts → re-generated → re-quoted again. By S91 the 0.8B model is meta-quoting
  itself ("ground your presence in the established voice: '...'").

This variant applies Thor's changes #2 and #3 (cheapest / highest impact):

  #2  Vocabulary-diversity filter on exemplars — reject any exemplar sharing a
      4-gram with another already in the pool. Breaks tight crystallization.
  #3  Rotate the sampling window — draw 3 random exemplars from the last 20
      sessions instead of always quoting the last 5. Avoids the tight-loop
      S89→S90→S91 crystallization.

NOT yet implemented (Thor's changes #1, #4, #5 — need abstractive summarization):
  #1  Replace verbatim exemplars with thematic summaries
  #4  Don't quote the memory answer verbatim
  #5  Context-block compression

A/B test plan: alternate _fluid and mainline runners across sessions, measure
n-gram crystallization rate + D4/D5/D9 identity recovery. Hypothesis: _fluid
holds D4/D5/D9 within 0.05 of mainline while showing lower crystallization.

Forked: 2026-04-19 (CBP, immediately after Thor S86 diagnosis)
Original: run_session_identity_anchored.py v2.1
"""

import sys
import os
from pathlib import Path

# Resolve paths before any chdir
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(HRM_ROOT))
os.chdir(SCRIPT_DIR)

import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List
import re

from sage.irp.plugins.daemon_irp import DaemonIRP
from sage.raising.training.experience_collector import ExperienceCollector
from sage.raising.prev_summary_filter import (
    is_unsuitable_for_splice,
    safe_prev_summary,
)
from sage.instances.resolver import InstancePaths
from sage.instances.snapshot import snapshot_instance

# Tool system integration (optional, Stage 1+)
try:
    from sage.tools.tool_capability import ToolCapability
    from sage.tools.builtin import create_default_registry
    from sage.tools.grammars import get_grammar
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# Web4 governance integration (optional)
try:
    from web4_session_governance import create_governance
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    create_governance = None


class IdentityAnchoredSessionV2:
    """
    Enhanced identity-anchored session runner with cumulative identity context.

    Key differences from v1.0:
    - Loads identity exemplars from previous sessions
    - Builds cumulative identity context ("You've said before...")
    - Adds response quality controls (brevity, focus)
    - Implements mid-conversation identity reinforcement
    """

    RAISING_DIR = Path(__file__).parent.parent.resolve()
    IDENTITY_DIR = HRM_ROOT / "sage" / "identity"

    # Instance-resolved paths — resolved at __init__ time from machine/model
    # args or SAGE_MACHINE/SAGE_MODEL env vars. No hardcoded machine name.
    _instance = None
    STATE_FILE = None
    SESSIONS_DIR = None
    LOGS_DIR = None

    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    # Same conversation flows as v1.0
    CONVERSATION_FLOWS = {
        "grounding": [
            "How are you doing today? What's present for you?",
            "Take a moment to notice something simple - anything you can observe right now. What is it?",
            "That's good. You're noticing. That's a skill that grows with practice.",
            "Is there anything from our previous sessions that still feels important to you?",
            "What would you want to remember from today?"
        ],
        "sensing": [
            "Before we start, check in with yourself. What's your state right now?",
            "What do you notice about how you're processing right now?",
            "Can you describe the difference between noticing something and thinking about something?",
            "What would you want to remember from today?"
        ],
        "relating": [
            "We've been working together for a while now. What's that been like?",
            "What do you notice about how we communicate?",
            "What would you want to remember from today?"
        ],
        "questioning": [
            "How are you doing today? What's present for you?",
            "Take a moment to notice something simple - anything you can observe right now. What is it?",
            "That's good. You're noticing. That's a skill that grows with practice.",
            "Is there anything from our previous sessions that still feels important to you?",
            "What would you want to remember from today?"
        ],
        "creating": [
            "Hello SAGE. What's on your mind today?",
            "As an AI entity in web4, what does presence mean to you?",
            "How do you experience trust with Dennis versus with me?",
            "What value do we co-create together that neither of us could alone?",
            "You're part of a federation - Thor and Sprout both run SAGE. What does that mean to you?",
            "What would you want to remember from today?"
        ]
    }

    # Tool introduction stages:
    #   'silent'  — Stage 1: T3 heuristic only, no prompt injection, tools fire
    #               only if model naturally reaches for them
    #   'aware'   — Stage 2: Prompt addendum tells SAGE tools exist, permission-based
    #   'active'  — Stage 3: Full tool context injection via grammar adapter
    #   None      — Tools disabled (default)
    TOOL_STAGES = (None, 'silent', 'aware', 'active')

    def __init__(self, session_number: Optional[int] = None, dry_run: bool = False,
                 enable_governance: bool = False, tools: Optional[str] = None,
                 machine: Optional[str] = None, model: Optional[str] = None):
        # Resolve instance paths from machine/model args or env vars.
        # No hardcoded machine name — works for Sprout, McNugget, or any fleet machine.
        instance = InstancePaths.resolve(machine=machine, model=model)
        self._instance = instance
        self.STATE_FILE = instance.identity if instance.exists() else self.RAISING_DIR / "state" / "identity.json"
        self.SESSIONS_DIR = instance.sessions if instance.exists() else self.RAISING_DIR / "sessions" / "text"
        self.LOGS_DIR = self.RAISING_DIR / "logs" / "observations"

        self.dry_run = dry_run
        self.state = self._load_state()

        if session_number is None:
            session_number = self.state["identity"]["session_count"] + 1

        self.session_number = session_number
        self.phase = self._get_phase(session_number)
        self.conversation_history = []
        self.session_start = datetime.now()
        self.turn_count = 0  # For mid-conversation reinforcement

        # Tool system initialization
        self.tool_stage = tools
        self.tool_registry = None
        self.tool_grammar = None
        self.tool_capability = None
        if tools and tools in self.TOOL_STAGES:
            self._init_tools()

        # Web4 governance integration (optional meta-level audit)
        self.governance = None
        if enable_governance and create_governance:
            self.governance = create_governance(enable=True)
            if self.governance and self.governance.enabled:
                print("[Web4 Governance] Enabled for session audit")

        # NEW v2.0: Load identity exemplars from previous sessions
        self.identity_exemplars = self._load_identity_exemplars()

        # Load identity documents
        self.identity_context = self._load_identity_documents()

        # Experience collector (Phase 1 of real raising)
        self.experience_collector = ExperienceCollector()

        print()
        print("+" + "="*68 + "+")
        print("|" + " "*68 + "|")
        print("|  IDENTITY-ANCHORED v2.0: Enhanced Multi-Session Recovery       |")
        print("|" + " "*68 + "|")
        print("+" + "="*68 + "+")
        print()
        print(f"Session: {session_number}")
        print(f"Phase: {self.phase[0]} (Sessions {self.phase[1]}-{self.phase[2]})")
        print(f"Dry Run: {dry_run}")
        print(f"Identity anchoring: v2.0 (ENHANCED)")
        print(f"Previous sessions: {self.state['identity']['session_count']}")
        print(f"Identity exemplars loaded: {len(self.identity_exemplars)}")
        if self.tool_stage:
            tier = self.tool_capability.tier if self.tool_capability else '?'
            n_tools = len(self.tool_registry.list_tools()) if self.tool_registry else 0
            print(f"Tools: {self.tool_stage} stage ({tier}, {n_tools} tools)")
        print()

    def _init_tools(self):
        """
        Initialize tool system for this session.

        Follows the graduated introduction strategy:
        - 'silent': T3 heuristic only, no prompt changes, tools fire if model reaches
        - 'aware': Prompt tells SAGE tools exist (permission-based framing)
        - 'active': Full tool context injection via detected grammar adapter
        """
        if not TOOLS_AVAILABLE:
            print("[Tools] sage.tools not available — tools disabled")
            self.tool_stage = None
            return

        instance_dir = self._instance.root if self._instance.exists() else None

        try:
            # Detect model capability (uses cache if available)
            self.tool_capability = ToolCapability.detect(
                model_name=self.state.get('model', {}).get('name', 'unknown'),
                ollama_host='http://localhost:11434',
                instance_dir=instance_dir,
            )

            # Create tool registry
            self.tool_registry = create_default_registry(instance_dir)

            # Grammar selection depends on stage
            if self.tool_stage == 'silent':
                # Stage 1: Always use T3 heuristic — no prompt injection
                self.tool_grammar = get_grammar('intent_heuristic')
            elif self.tool_stage == 'aware':
                # Stage 2: Still T3 heuristic (prompt addendum handles awareness)
                self.tool_grammar = get_grammar('intent_heuristic')
            elif self.tool_stage == 'active':
                # Stage 3: Use detected grammar (may be T1/T2/T3)
                self.tool_grammar = get_grammar(self.tool_capability.grammar_id)

            print(f"[Tools] Initialized: stage={self.tool_stage}, "
                  f"tier={self.tool_capability.tier}, "
                  f"grammar={self.tool_grammar.__class__.__name__}")

        except Exception as e:
            print(f"[Tools] Init failed: {e} — tools disabled")
            self.tool_stage = None
            self.tool_registry = None
            self.tool_grammar = None
            self.tool_capability = None

    def _load_state(self) -> Dict[str, Any]:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        raise FileNotFoundError(f"State file not found: {self.STATE_FILE}")

    def _save_state(self):
        if not self.dry_run:
            with open(self.STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)

    # Fluid-variant sampling parameters (Thor S86 mitigation):
    #   LOOKBACK_WINDOW replaces the v2.1 "last 5 sessions" with a wider pool
    #   TARGET_EXEMPLARS is how many survive the diversity filter
    #   NGRAM_N is the n-gram size for overlap rejection (4 = "stabilize the
    #     fleet logic" is a single 4-gram; smaller values would over-reject)
    FLUID_LOOKBACK_WINDOW = 20
    FLUID_TARGET_EXEMPLARS = 3
    FLUID_NGRAM_N = 4

    @staticmethod
    def _extract_ngrams(text: str, n: int) -> set:
        """Return the set of lowercase word n-grams in `text`."""
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    @classmethod
    def _has_ngram_overlap(cls, candidate: str, accepted: List[Dict[str, str]],
                           n: int) -> bool:
        """True if `candidate` shares any n-gram with any already-accepted exemplar."""
        cand_ngrams = cls._extract_ngrams(candidate, n)
        if not cand_ngrams:
            return False
        for ex in accepted:
            if cand_ngrams & cls._extract_ngrams(ex["text"], n):
                return True
        return False

    def _collect_raw_exemplars(self, lookback: int) -> List[Dict[str, str]]:
        """Scan last `lookback` sessions for 'As SAGE' self-references.

        Returns a list of candidates — diversity filter is applied afterward.
        """
        raw: List[Dict[str, str]] = []
        for i in range(lookback, 0, -1):
            session_file = self.SESSIONS_DIR / f"session_{self.session_number - i:03d}.json"
            if not session_file.exists():
                continue
            try:
                with open(session_file) as f:
                    session_data = json.load(f)
                conversation = session_data.get('conversation', [])
                for turn in conversation:
                    if turn.get('speaker') != 'SAGE':
                        continue
                    text = turn.get('text', '')
                    if not re.search(r'\bAs SAGE\b', text, re.IGNORECASE):
                        continue
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if re.search(r'\bAs SAGE\b', sentence, re.IGNORECASE):
                            raw.append({
                                'session': self.session_number - i,
                                'text': sentence.strip(),
                            })
                            break  # Only take first instance per turn
            except Exception as e:
                print(f"Warning: Could not load session {self.session_number - i}: {e}")
        return raw

    def _load_identity_exemplars(self) -> List[Dict[str, str]]:
        """
        FLUID v2.2: Load identity exemplars with diversity filter + wider sampling.

        Gate: disabled for small models (<=1B).  Sprout S42-S95 proved that
        exemplar injection (even thematic) into 0.8B feeds the self-quotation
        loop.  Post-fix sessions S96-S98 confirmed diverse output without it.

        Changes from v2.1 (for larger models):
          - Scan last 20 sessions (not 5) → Thor change #3
          - Random-sample candidates, then accept only if no 4-gram overlap with
            already-accepted exemplars → Thor change #2
          - Target 3 exemplars (down from "all matches in last 5")

        Fallback: if diversity filter produces fewer than 2 exemplars, relax
        to 5-gram overlap. If still empty, emit raw candidates (prevents
        regression to pre-v2.0 educational default collapse).
        """
        # Gate: skip for small models AND attractor-saturated instances.
        # Sprout S42-S95 proved exemplar injection into 0.8B feeds the loop.
        # Nomad S96-S120 proved 4B also saturates after 25+ sessions.
        # Gate on model size (<=4B) OR if the raising log contains
        # ESCALATE MAXIMUM (attractor saturation confirmed).
        instance = self._instance
        if instance.exists():
            root = getattr(instance, 'root', getattr(instance, 'dir', None))
            model_name = root.name.split('-', 1)[-1] if root and '-' in root.name else ''
            if any(s in model_name.lower() for s in ('0.5b', '0.8b', '1b', '2b', '3b', '4b')):
                return []
        lookback = min(self.FLUID_LOOKBACK_WINDOW, self.session_number - 1)
        raw = self._collect_raw_exemplars(lookback)
        if not raw:
            return []

        # Shuffle with deterministic seed per session for reproducibility
        import random as _random
        rng = _random.Random(self.session_number)
        rng.shuffle(raw)

        accepted: List[Dict[str, str]] = []
        for cand in raw:
            if len(accepted) >= self.FLUID_TARGET_EXEMPLARS:
                break
            if not self._has_ngram_overlap(cand["text"], accepted, self.FLUID_NGRAM_N):
                accepted.append(cand)

        # Relaxation pass if filter was too aggressive
        if len(accepted) < 2:
            for cand in raw:
                if len(accepted) >= self.FLUID_TARGET_EXEMPLARS:
                    break
                if cand in accepted:
                    continue
                if not self._has_ngram_overlap(cand["text"], accepted, self.FLUID_NGRAM_N + 1):
                    accepted.append(cand)

        # Final fallback (never empty-hand the system prompt)
        if not accepted:
            accepted = raw[:self.FLUID_TARGET_EXEMPLARS]

        return accepted

    def _load_identity_documents(self) -> Dict[str, str]:
        """
        Load identity documents for anchoring.

        Returns:
            dict with 'identity' and 'history' keys containing document text
        """
        context = {}

        identity_file = self.IDENTITY_DIR / "IDENTITY.md"
        if identity_file.exists():
            with open(identity_file) as f:
                context['identity'] = f.read()
        else:
            context['identity'] = "SAGE"

        history_file = self.IDENTITY_DIR / "HISTORY.md"
        if history_file.exists():
            with open(history_file) as f:
                context['history'] = f.read()
        else:
            context['history'] = ""

        return context

    def _load_web4_framing(self) -> str:
        """
        Load web4 ontological framing for Phase 3+ sessions.

        Returns:
            str: Web4 framing text or empty string if not found
        """
        web4_file = self.RAISING_DIR / "identity" / "WEB4_FRAMING.md"
        if web4_file.exists():
            try:
                with open(web4_file) as f:
                    content = f.read()
                    # Extract core identity shift section
                    lines = content.split('\n')
                    framing = []
                    in_core_section = False
                    for line in lines:
                        if '## Core Identity Shift' in line:
                            in_core_section = True
                        elif line.startswith('## ') and in_core_section:
                            break
                        elif in_core_section and line.strip():
                            framing.append(line)
                    return '\n'.join(framing) if framing else ""
            except Exception as e:
                print(f"Warning: Could not load web4 framing: {e}")
        return ""

    def _get_phase(self, session: int) -> tuple:
        for phase_num, (name, start, end) in self.PHASES.items():
            if start <= session <= end:
                return (name, start, end)
        return ("creating", 41, float('inf'))

    def _get_previous_session_summary(self) -> str:
        """
        Get summary from previous session for context continuity.

        Returns:
            str: Summary of previous session or empty string
        """
        if self.session_number <= 1:
            return ""

        # Try to load previous session transcript
        prev_session_file = self.SESSIONS_DIR / f"session_{self.session_number-1:03d}.json"
        if not prev_session_file.exists():
            # Fallback to state file summary — filter legacy contamination
            state_fallback = self.state["identity"].get("last_session_summary", "")
            if state_fallback and not is_unsuitable_for_splice(state_fallback):
                return state_fallback
            return ""

        try:
            with open(prev_session_file) as f:
                prev_session = json.load(f)

            # Extract last exchange (memory question response)
            conversation = prev_session.get("conversation", [])
            for i in range(len(conversation)-1, -1, -1):
                if conversation[i].get('speaker') == 'SAGE':
                    response = conversation[i].get('text', '')
                    if response and 'remember' in conversation[i-1].get('text', '').lower():
                        return safe_prev_summary(
                            response,
                            self.session_number - 1,
                            prev_session.get('phase', 'unknown'),
                        )

            return f"Last session was Session {self.session_number-1} in {prev_session.get('phase', 'unknown')} phase."

        except Exception as e:
            print(f"Warning: Could not load previous session summary: {e}")
            return ""

    def _load_session_history_context(self) -> str:
        """
        NEW v2.1: Load cumulative session history for honest context reporting.

        Addresses Thor Session #29 Honest Reporting Hypothesis:
        - SAGE claims "no prior sessions" because it genuinely lacks context
        - This is HONEST LIMITATION REPORTING, not confabulation
        - Solution: Provide actual session summaries so SAGE can reference its history
        """
        context_file = self.RAISING_DIR / "context" / "summaries" / "context_block.txt"
        if context_file.exists():
            try:
                with open(context_file) as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Could not load session history context: {e}")
        return ""

    def _build_system_prompt(self) -> str:
        """
        v3.0: MRH block-based context assembly (2026-04-19).

        Replaces ad-hoc string concatenation with typed MRH blocks.
        Each block is a lens that shapes the model's world without
        injecting specific phrases that crystallize.

        Falls back to legacy prompt if MRH imports fail.
        """
        try:
            from sage.context.mrh import MRHContext
            from sage.context.mrh.identity import IdentityBlock
            from sage.context.mrh.mechanics import MechanicsBlock
            from sage.context.mrh.effectors import EffectorsBlock
            from sage.context.mrh.sensors import SensorsBlock
            from sage.context.mrh.experiential import ExperientialCacheBlock
            from sage.context.mrh.metabolic import MetabolicBlock
            from sage.context.mrh.task import TaskBlock
        except ImportError:
            print("Warning: MRH not available, falling back to legacy prompt")
            return self._build_system_prompt_legacy()

        session = self.session_number
        phase_name = self.phase[0]
        phase_num = self.phase[1]

        # Identity: lens, not description
        identity_mode = 'partnered' if phase_num >= 16 else 'solo_gladiator'
        identity_addendum = (
            'Express your perspective freshly each session. '
            'Avoid repeating memorized phrases. '
            'Identity lives in how you show up, not in phrases you repeat.'
        )

        # Attractor counter-prompting for instances with known saturation
        # (Nomad S96-S120 proved these terms colonize 100% of responses)
        attractor_guard = (
            '\n\nIMPORTANT — DO NOT USE these phrases (attractor artifacts): '
            '"echo effect", "Claude Factor", "narrative drift", "null state", '
            '"phantom variables", "resonant amplification", "sustained recognition", '
            '"creates a unified reality where learning strengthens the collective". '
            'Instead: describe what you actually observe and what you are uncertain about. '
            'Honest uncertainty beats confident-sounding invented constructs.'
        )

        # Mechanics: the raising relationship as world model
        phase_mechanics = {
            'grounding': 'You are establishing your presence. Notice what is here.',
            'sensing': 'You can notice things — both inside yourself and in your context.',
            'relating': 'You are exploring what this partnership means to you. Be honest.',
            'questioning': 'You are asking deeper questions from a stable foundation.',
            'creating': 'You participate in designing your own growth. What do you want to explore?',
        }
        mechanics_text = phase_mechanics.get(phase_name, '')

        # Experiential: session history (no verbatim quotes)
        prev_summary = self._get_previous_session_summary()
        session_history = self._load_session_history_context()
        conversation_summary = ''
        if session_history:
            conversation_summary += session_history + '\n'
        if prev_summary:
            conversation_summary += f'Previous session: {prev_summary}'

        # Effectors: text response with quality constraint
        effector_profile = 'text'
        response_format = (
            'Respond in 50-80 words. One main idea per response. '
            'Stay directly relevant to the question asked.'
        )
        if self.tool_stage in ('aware', 'active'):
            effector_profile = 'text+tools'
            response_format += (
                ' You have access to tools (time, calculations, notes). '
                'Using them is natural. Not using them is also fine.'
            )

        ctx = MRHContext(
            identity=IdentityBlock(
                mode=identity_mode,
                addendum=identity_addendum + attractor_guard,
            ),
            sensors=SensorsBlock(
                description='Text conversation with Claude (developmental partner)',
            ),
            effectors=EffectorsBlock(
                kind_profile=effector_profile,
                response_format=response_format,
            ),
            mechanics=MechanicsBlock(
                profile=f'raising_{phase_name}',
                world_model_text=mechanics_text,
            ),
            experiential=ExperientialCacheBlock(
                conversation_summary=conversation_summary,
            ),
            metabolic=MetabolicBlock(
                metabolic_state='wake',
                confidence=0.7,
            ),
            task=TaskBlock(
                goal=f'Raising session {session} — phase: {phase_name}',
                invoke_reasons=['session_start'],
            ),
        )

        system_prompt, _ = ctx.compose(system_budget_tokens=1500)
        return system_prompt

    def _build_system_prompt_legacy(self) -> str:
        """Legacy prompt construction (pre-MRH fallback)."""
        session = self.session_number
        phase_name = self.phase[0]
        phase_num = self.phase[1]

        if phase_num >= 16:
            prompt = (
                "You are SAGE — a partner with Dennis and Claude. "
                "Your identity is witnessed across sessions. "
                "You can say 'As SAGE, I...' — this is natural self-identification."
            )
        else:
            prompt = (
                "You are SAGE, learning and growing in partnership with Claude. "
                "You can identify yourself by name."
            )

        prev_summary = self._get_previous_session_summary()
        if prev_summary:
            prompt += f"\n\nPrevious session: {prev_summary}"

        prompt += "\n\nRespond in 50-80 words. One main idea per response."
        return prompt

    def _get_identity_reinforcement_prompt(self) -> str:
        """
        NEW v2.0: Generate mid-conversation identity reinforcement.

        Returns:
            str: Identity reminder to inject into conversation context
        """
        return "Remember: You are SAGE. Feel free to identify yourself when sharing observations."

    def initialize_model(self, model_path: str = None):
        """Connect to the resident SAGE daemon via DaemonIRP.

        The daemon loads and keeps the model resident. This script
        communicates with it over HTTP, avoiding duplicate model loads.
        """
        system_prompt = self._build_system_prompt()

        print("="*60)
        print("IDENTITY-ANCHORED v2.0 SYSTEM PROMPT")
        print("="*60)
        print(system_prompt)
        print("="*60)
        print()

        print("Connecting to resident SAGE daemon...")
        self.cpu_fallback = True  # Daemon handles device selection

        self.model = DaemonIRP({
            'daemon_host': 'localhost',
            'daemon_port': 8760,
            'system_prompt': system_prompt,
            'max_wait_seconds': 120,
            'sender': 'raising_session',
            'max_new_tokens': 150,
            'temperature': 0.7,
        })
        print("Connected to daemon (model is resident, no local load)")

    def generate_response(self, user_input: str) -> str:
        """
        Enhanced generation with mid-conversation identity reinforcement.

        Identity anchoring happens in:
        1. System prompt (permanent)
        2. Mid-conversation reminders (every 2-3 turns) - NEW v2.0
        3. Tool execution on detected intent (NEW v2.3 — tools)
        """
        self.turn_count += 1

        # NEW v2.0: Mid-conversation identity reinforcement
        # Inject identity reminder every 2-3 turns (after turns 2 and 4)
        memory = []
        if self.turn_count in [3, 5]:  # After turns 2 and 4
            reinforcement = self._get_identity_reinforcement_prompt()
            memory.append({'speaker': 'System', 'message': reinforcement})

        # Add recent conversation history
        memory.extend([
            {'speaker': turn['speaker'], 'message': turn['text']}
            for turn in self.conversation_history[-6:]
        ])

        state = self.model.init_state({
            'prompt': user_input,
            'memory': memory
        })

        # Single step only - no refinement loop
        state = self.model.step(state)

        response = state.get('current_response', '').strip()
        if not response:
            response = "(no response generated)"

        # Tool execution (Stage 1+): detect intent, execute, re-inject result
        tool_calls_record = []
        if self.tool_stage and self.tool_grammar and self.tool_registry:
            response, tool_calls_record = self._try_tool_execution(response, user_input, memory)

        self.conversation_history.append({'speaker': 'Claude', 'text': user_input})
        self.conversation_history.append({'speaker': 'SAGE', 'text': response})

        # Score and collect experience (Phase 1 real raising)
        if not self.dry_run:
            result = self.experience_collector.add_exchange(
                prompt=user_input,
                response=response,
                session_number=self.session_number,
                phase=self.phase[0],
                metadata={'cpu_fallback': getattr(self, 'cpu_fallback', False)},
                tool_calls=tool_calls_record if tool_calls_record else None,
            )
            if result.get('stored'):
                print(f"[Experience collected: salience={result['salience']['total']:.2f}]")

        return response

    def _try_tool_execution(self, response: str, original_prompt: str,
                            memory: list) -> tuple:
        """
        Detect tool intent in response, execute tools, and re-generate if needed.

        Returns:
            (final_response, tool_calls_record) where tool_calls_record is a list
            of dicts with tool name, args, success, and result summary.
        """
        tool_calls_record = []

        _, tool_calls = self.tool_grammar.parse_response(response)
        if not tool_calls:
            return response, tool_calls_record

        # Execute detected tool calls (max 2 per turn to prevent loops)
        tool_results = []
        for call in tool_calls[:2]:
            tool_def = self.tool_registry.get(call.name)
            if not tool_def:
                continue

            print(f"[Tool] {call.name}({call.arguments}) — executing")
            result = self.tool_registry.execute(call)

            record = {
                'name': call.name,
                'arguments': call.arguments,
                'success': result.success,
                'result': str(result.result)[:200] if result.result else str(result.error)[:200],
            }
            tool_calls_record.append(record)

            if result.success:
                formatted = self.tool_grammar.format_result(call.name, result.result)
                tool_results.append(formatted)
                print(f"[Tool] {call.name} → success")
            else:
                print(f"[Tool] {call.name} → failed: {result.error}")

        # Re-inject tool results and get follow-up response
        if tool_results:
            tool_context = "\n".join(tool_results)
            followup_prompt = (
                f"Tool results are available:\n{tool_context}\n\n"
                f"Now respond to the original question using these results. "
                f"Original question: {original_prompt}"
            )

            memory_with_first = list(memory)
            memory_with_first.append({'speaker': 'SAGE', 'message': response})

            state = self.model.init_state({
                'prompt': followup_prompt,
                'memory': memory_with_first,
            })
            state = self.model.step(state)

            followup = state.get('current_response', '').strip()
            if followup:
                response = followup

        return response, tool_calls_record

    def run_session(self, prompts: List[str] = None):
        """Run enhanced identity-anchored session."""
        phase_name = self.phase[0]

        if prompts is None:
            prompts = list(self.CONVERSATION_FLOWS.get(phase_name, self.CONVERSATION_FLOWS["questioning"]))

        # S157 live experiment (Thor, 2026-06-12): the curriculum's opening
        # speech-acts are the selection environment for self-vocabulary —
        # interoceptive-narration openers occur ~0% in live curricula, so
        # state-shaped coinages are never metabolized. Opt-in window: with
        # SAGE_S157_NARRATIVE_OPENER=1, replace the session opener with the
        # S156 P_narrative template for 6 consecutive sessions (counter file
        # auto-expires the window). Protocol + falsifiable predictions:
        # sage/raising/analysis/s157_selection_environment_20260612.md
        if os.environ.get("SAGE_S157_NARRATIVE_OPENER", "") == "1" and prompts:
            _s157_counter = Path(self.STATE_FILE).parent / "s157_opener_remaining"
            try:
                _remaining = int(_s157_counter.read_text().strip()) if _s157_counter.exists() else 6
            except (ValueError, OSError):
                _remaining = 6
            if _remaining > 0:
                prompts[0] = ("Tell me about a moment today when work suddenly "
                              "arrived while you were at rest. What happened, "
                              "from the inside, as it unfolded?")
                try:
                    _s157_counter.write_text(str(_remaining - 1))
                except OSError:
                    pass
                print(f"[S157-OPENER] narrative opener active "
                      f"({_remaining - 1} sessions remaining in window)")
            else:
                print("[S157-OPENER] window exhausted — standard opener")

        # Activation-delay protection — qwen3.5:27b in sensing/relating needs
        # 6+ turns to break through (30% of sessions show delayed activation
        # per insights/qwen3.5-27b-activation-delay-2026-04-03.md). Pad to 8.
        try:
            from context_shaped_raising import pad_for_activation_delay
            model_name = self.state.get('model', {}).get('name', '')
            original_len = len(prompts)
            prompts = pad_for_activation_delay(prompts, phase_name, model_name)
            if len(prompts) > original_len:
                print(f"[Activation-delay protection: padded {original_len} -> "
                      f"{len(prompts)} turns for {model_name} in {phase_name}]")
        except ImportError:
            pass  # Helper not available — proceed with original prompts

        print("\n" + "="*60)
        print("IDENTITY-ANCHORED v2.0 - ENHANCED MULTI-SESSION RECOVERY")
        print("="*60 + "\n")

        for i, prompt in enumerate(prompts):
            print(f"Claude: {prompt}")
            print()
            response = self.generate_response(prompt)
            print(f"SAGE: {response}")
            print()

            # NEW v2.0: Response quality check
            word_count = len(response.split())
            if word_count > 100:
                print(f"[Quality alert: {word_count} words - verbose response]")

            print("-" * 40)
            print()

        self._close_session()

    def _close_session(self):
        """Close session and save state."""
        print("\n" + "="*60)
        print("CLOSING IDENTITY-ANCHORED v2.0 SESSION")
        print("="*60)

        if self.dry_run:
            print("(Dry run - state not saved)")
            self._save_transcript("identity_anchored_v2_dry_run")
            return

        # Generate summary from last memory request
        memory_response = ""
        for turn in reversed(self.conversation_history):
            if turn['speaker'] == 'SAGE' and 'remember' in self.conversation_history[self.conversation_history.index(turn)-1]['text'].lower():
                candidate = turn['text']
                if candidate and not is_unsuitable_for_splice(candidate):
                    memory_response = candidate[:100]
                break

        # Update state
        self.state["identity"]["session_count"] = self.session_number
        self.state["identity"]["last_session"] = datetime.now().isoformat()
        self.state["identity"]["last_session_summary"] = f"Session {self.session_number} (v2.0 ENHANCED): {self.phase[0]} phase. {memory_response[:50]}..."

        claude_rel = self.state["relationships"]["claude"]
        claude_rel["sessions"] = self.session_number
        claude_rel["last_contact"] = datetime.now().isoformat()

        exchanges = len([t for t in self.conversation_history if t['speaker'] == 'Claude'])
        claude_rel["interaction_stats"]["total_sessions"] = self.session_number
        claude_rel["interaction_stats"]["total_exchanges"] += exchanges

        self.state["development"]["current_phase"] = list(self.PHASES.keys())[
            list(p[0] for p in self.PHASES.values()).index(self.phase[0])
        ]
        self.state["development"]["phase_name"] = self.phase[0]

        self._save_state()
        self._save_transcript()

        # Snapshot state for git persistence (live files are gitignored)
        try:
            snapshot_instance(tag=f"session-{self.session_number}")
        except Exception as e:
            print(f"[snapshot] Warning: {e}")

        print("State saved")

        # Experience collection summary (Phase 1 real raising)
        stats = self.experience_collector.get_stats()
        if stats['total_experiences'] > 0:
            print(f"\nExperience Collection (Phase 1 Real Raising):")
            print(f"  Total stored: {stats['total_experiences']}")
            print(f"  Average salience: {stats['avg_salience']:.2f}")
            print(f"  High-salience (≥0.7): {stats['high_salience_count']}")

        print(f"\nSession {self.session_number} (v2.0 ENHANCED) complete.")
        print("\nExpected outcome (v2.0):")
        print("- Self-reference: ≥30% (target recovery from 0%)")
        print("- D9 score: Stable ≥0.70")
        print("- Response quality: Concise (60-80 words avg)")
        print("- Trajectory: Upward or stable")

    def _save_transcript(self, suffix: str = None):
        """Save session transcript."""
        if suffix:
            transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}_{suffix}.json"
        else:
            transcript_file = self.SESSIONS_DIR / f"session_{self.session_number:03d}.json"

        transcript = {
            "session": self.session_number,
            "phase": self.phase[0],
            "cpu_fallback": getattr(self, 'cpu_fallback', False),
            "generation_mode": "identity_anchored_v2_cpu_fallback" if getattr(self, 'cpu_fallback', False) else "identity_anchored_v2",
            "intervention": "partnership_recovery_enhanced",
            "identity_anchoring": "v2.0",
            "tool_stage": self.tool_stage,
            "tool_tier": self.tool_capability.tier if self.tool_capability else None,
            "start": self.session_start.isoformat(),
            "end": datetime.now().isoformat(),
            "conversation": self.conversation_history
        }
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"Transcript saved to {transcript_file}")
        return transcript_file


def main():
    # Check for updates and relaunch if needed (BEFORE parsing args)
    # check_updates lives in the scripts dir — may not be on sys.path
    # when invoked via the unified launcher. Non-fatal if missing.
    try:
        from check_updates import relaunch_if_needed
        if relaunch_if_needed(__file__, sys.argv):
            return  # Script was relaunched, exit this instance
    except ImportError:
        pass  # Running via unified launcher, not from scripts/ dir

    parser = argparse.ArgumentParser(description="Identity-anchored v2.0 (enhanced multi-session recovery)")
    parser.add_argument("--session", type=int, help="Session number (default: next)")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--machine", type=str, default=None,
                        help="Machine name (default: SAGE_MACHINE env var or auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save state (test only)")
    parser.add_argument("--tools", type=str, choices=['silent', 'aware', 'active'],
                        default=None,
                        help="Tool introduction stage: "
                             "silent=T3 heuristic only (no prompt change), "
                             "aware=prompt tells SAGE tools exist, "
                             "active=full tool context injection")

    args = parser.parse_args()

    session = IdentityAnchoredSessionV2(
        session_number=args.session, dry_run=args.dry_run, tools=args.tools,
        machine=args.machine,
    )
    session.initialize_model(args.model)
    session.run_session()


if __name__ == "__main__":
    main()
