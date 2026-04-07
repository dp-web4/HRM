#!/usr/bin/env python3
"""
SAGE Puzzle Solver v10 — Golden Ratio Validation (Thor Session #65).

Extends v8 with coherence (C) measurement to test the golden ratio hypothesis:
"C → φ - 1 ≈ 0.618 is the developmental attractor for optimal consciousness" (Session #64).

NEW in v10:
- CoherenceTracker: Estimates SNARC components from gameplay dynamics
  - Surprise: New patterns discovered
  - Novelty: New object types encountered
  - Arousal: Action effectiveness / grid changes
  - Reward: Level-ups and progress
  - Conflict: Planning difficulty / action failures
- Golden ratio zone detection: |C - 0.618| < 0.05
- Extended ATP logs with C measurement and distance from φ - 1
- Tests Prediction #4: "ATP highest when C ≈ φ - 1" (Session #64)

Hypothesis: ATP peaks at C ≈ 0.618, not 0.64 or 0.5.

Usage:
    cd ~/ai-workspace/SAGE
    python3 arc-agi-3/experiments/sage_solver_v10.py --game lp85 -v --log-atp
"""

import sys, os, time, json, re, random
import numpy as np
import requests
from collections import deque

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")
sys.path.insert(0, "sage/core")  # For metabolic controller

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, background_color, color_name, find_color_regions
from arc_spatial import SpatialTracker
from arc_action_model import ActionEffectModel
from arc_narrative import SessionNarrative
from arc_context import ContextConstructor
from arc_federation import FederatedKnowledge

# Import metabolic controller (Thor Session #60 infrastructure)
try:
    from metabolic_controller import MetabolicController
    METABOLIC_AVAILABLE = True
except ImportError:
    METABOLIC_AVAILABLE = False
    print("Warning: MetabolicController not available. ATP logging disabled.")

# Import context budget from raising (shared module)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sage', 'raising', 'scripts'))
    from context_shaped_raising import ContextBudget
except ImportError:
    class ContextBudget:
        """Fallback if raising module not available."""
        def __init__(self, max_tokens=131072): self.max_tokens = max_tokens; self.layers = {}
        def record(self, name, text): self.layers[name] = len(text)
        def total_chars(self): return sum(self.layers.values())
        def utilization(self): return (self.total_chars() / 4) / self.max_tokens
        def report(self):
            total = self.total_chars() // 4
            lines = [f"Context: ~{total} tokens ({self.utilization():.1%} of {self.max_tokens})"]
            for n, c in sorted(self.layers.items(), key=lambda x: -x[1]):
                lines.append(f"  {n}: ~{c//4} tokens")
            lines.append(f"  FREE: ~{self.max_tokens - total} tokens")
            return "\n".join(lines)

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "")

# Golden ratio constant (Session #64)
PHI = 1.618033988749895
PHI_MINUS_1 = 0.618033988749895
GOLDEN_ZONE_THRESHOLD = 0.05

# Context budget targets (from context-budget-and-raising.md)
TARGET_BUDGET = {
    "layer4_metacognitive": 500,
    "layer3_membot": 2000,
    "layer2_game_kb": 3000,
    "layer1_narrative": 5000,
}

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT", 6: "CLICK", 7: "UNDO"}


# ★ NEW: Coherence Tracker (Thor Session #65)
class CoherenceTracker:
    """
    Estimates coherence (C) from SNARC-like components during ARC-AGI gameplay.

    Session #64: C → φ - 1 ≈ 0.618 is developmental attractor.

    SNARC Components (0-1 scale):
    - Surprise (S): New patterns discovered (exponential decay)
    - Novelty (N): New object types encountered (recent window)
    - Arousal (A): Action effectiveness / grid changes (moving average)
    - Reward (R): Level-ups and progress (exponential decay)
    - Conflict (C_conflict): Planning difficulty / failures (inverted success rate)

    Coherence: C = (S + N + A + R + C_conflict) / 5
    """

    def __init__(self, window_size=10):
        self.window_size = window_size

        # Surprise: unique patterns seen (decays over time)
        self.patterns_seen = set()
        self.surprise_decay = 0.95
        self.surprise_value = 0.5  # baseline

        # Novelty: new objects in recent window
        self.objects_seen = set()
        self.objects_recent = deque(maxlen=window_size)

        # Arousal: action effectiveness (moving average)
        self.arousal_history = deque(maxlen=window_size)

        # Reward: level-ups (decays over time)
        self.reward_value = 0.0
        self.reward_decay = 0.9
        self.last_level = 0

        # Conflict: planning difficulty (inverted success)
        self.conflict_history = deque(maxlen=window_size)

        # History for analysis
        self.coherence_history = []

    def update(self, pattern_detected=None, new_objects=None, action_effective=False,
               current_level=0, planning_failed=False):
        """Update SNARC components based on gameplay events."""

        # Surprise: decay baseline, spike on new pattern
        self.surprise_value *= self.surprise_decay
        if pattern_detected and pattern_detected not in self.patterns_seen:
            self.patterns_seen.add(pattern_detected)
            self.surprise_value = min(1.0, self.surprise_value + 0.3)

        # Novelty: track new objects in recent window
        if new_objects:
            for obj in new_objects:
                if obj not in self.objects_seen:
                    self.objects_seen.add(obj)
                self.objects_recent.append(obj)

        # Arousal: action effectiveness
        self.arousal_history.append(1.0 if action_effective else 0.0)

        # Reward: level-up detection with decay
        self.reward_value *= self.reward_decay
        if current_level > self.last_level:
            self.reward_value = min(1.0, self.reward_value + 0.5)
            self.last_level = current_level

        # Conflict: planning difficulty
        self.conflict_history.append(1.0 if planning_failed else 0.0)

    def get_coherence(self):
        """Calculate current coherence C = mean(S, N, A, R, C_conflict)."""

        # Surprise: current decay value
        S = self.surprise_value

        # Novelty: proportion of recent objects that are unique
        if len(self.objects_recent) > 0:
            unique_recent = len(set(self.objects_recent))
            N = unique_recent / len(self.objects_recent)
        else:
            N = 0.5  # baseline

        # Arousal: mean effectiveness
        A = np.mean(self.arousal_history) if self.arousal_history else 0.5

        # Reward: current decay value
        R = self.reward_value

        # Conflict: mean difficulty (higher = more conflict)
        C_conflict = np.mean(self.conflict_history) if self.conflict_history else 0.5

        # Coherence: simple mean (Session #64: optimal formulation)
        C = (S + N + A + R + C_conflict) / 5.0

        # Log for analysis
        components = {'S': S, 'N': N, 'A': A, 'R': R, 'C_conflict': C_conflict, 'C': C}
        self.coherence_history.append(components)

        return C

    def distance_from_golden_ratio(self):
        """Distance from φ - 1 ≈ 0.618."""
        C = self.get_coherence()
        return abs(C - PHI_MINUS_1)

    def in_golden_zone(self):
        """Check if C is in golden ratio zone (±0.05)."""
        return self.distance_from_golden_ratio() < GOLDEN_ZONE_THRESHOLD

    def get_snapshot(self):
        """Get detailed coherence state for logging."""
        C = self.get_coherence()
        dist = self.distance_from_golden_ratio()
        in_zone = self.in_golden_zone()

        components = self.coherence_history[-1] if self.coherence_history else {}

        return {
            'coherence': round(C, 4),
            'distance_from_phi': round(dist, 4),
            'in_golden_zone': in_zone,
            'components': {k: round(v, 4) for k, v in components.items()},
            'phi_target': PHI_MINUS_1
        }


def _detect_model():
    """Auto-detect best available Ollama model."""
    global MODEL
    if MODEL:
        return
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []
    for preferred in ["gemma4:e4b", "gemma3:12b", "phi4:14b", "qwen3.5:0.8b",
                       "qwen2.5:3b", "qwen3.5:2b"]:
        if preferred in models:
            MODEL = preferred
            return
    chat_models = [m for m in models if "embed" not in m]
    MODEL = chat_models[0] if chat_models else "gemma4:e4b"


def _is_thinking_model():
    return "gemma4" in MODEL


# ─── LLM ───

def ask_llm(prompt: str, max_tokens: int = -1) -> str:
    opts = {"temperature": 0.3}
    if max_tokens == -1 and any(s in MODEL for s in ["0.8b", "0.5b", "1b", "2b", "3b"]):
        opts["num_predict"] = 300
    elif max_tokens > 0:
        opts["num_predict"] = max_tokens

    payload = {
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": opts,
    }
    if not _is_thinking_model():
        payload["think"] = False

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content if content else data.get("message", {}).get("thinking", "").strip()
    except Exception as e:
        return f"[error: {e}]"
    return ""


# ─── LAYER 4: Metacognitive principles ───

METACOGNITIVE = """METACOGNITIVE PRINCIPLES (from fleet raising experience):
- Track state transitions. When actions have periodic effects, count the period.
- Hold multiple hypotheses. Test the most distinguishing one first.
- Uncertainty is information. Not knowing what a button does = click it ONCE.
- Persistence ≠ perseveration. If an approach isn't producing new signal, change approach.
- Count before planning. Find cycle lengths before attempting solutions.
- Distinguish correlation from causation. Large pixel change ≠ progress toward goal."""


# ─── NAMING ───

def obj_name(obj):
    return f"{color_name(obj.color)}_{obj.id}"


def obj_catalog(tracker):
    lines = []
    interactive = tracker.get_interactive_objects()
    if interactive:
        lines.append("INTERACTIVE (proven — use these):")
        for o in interactive:
            lines.append(f"  {obj_name(o)} ({o.w}x{o.h}) at ({o.cx},{o.cy}) — {o.click_effectiveness:.0%}")
    static = tracker.get_static_objects()
    if static:
        lines.append(f"NO EFFECT ({len(static)} objects — skip)")
    untested = tracker.get_untested_objects()
    if untested:
        small = [o for o in untested if o.size <= 64]
        if small:
            lines.append(f"UNTESTED ({len(small)} small objects — explore if needed):")
            for o in sorted(small, key=lambda x: x.size)[:3]:
                lines.append(f"  {obj_name(o)} ({o.w}x{o.h}) at ({o.cx},{o.cy})")
    return "\n".join(lines)


# ─── PROBE (same as v8) ───

def probe(env, fd, grid, available, budget=20):
    model = ActionEffectModel()
    tracker = SpatialTracker()
    tracker.update(grid)
    steps = 0
    start_levels = fd.levels_completed
    level_ups = 0

    for action in [a for a in available if a != 6]:
        for _ in range(2):
            if steps >= budget:
                break
            prev = grid.copy()
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except:
                continue
            if fd is None:
                return model, tracker, fd, grid, steps, level_ups
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(action, prev, grid, diff)
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                return model, tracker, fd, grid, steps, level_ups

    if 6 in available:
        regions = find_color_regions(grid, min_size=4)
        by_size = sorted(regions, key=lambda x: x["size"])
        interleaved = []
        left, right = 0, len(by_size) - 1
        while left <= right:
            interleaved.append(by_size[left])
            if left != right:
                interleaved.append(by_size[right])
            left += 1
            right -= 1
        seen = set()
        for r in interleaved:
            key = (r["color"], r["w"], r["h"])
            if key in seen:
                continue
            seen.add(key)
            if steps >= budget:
                break
            prev = grid.copy()
            data = {'x': r["cx"], 'y': r["cy"]}
            try:
                fd = env.step(GameAction.ACTION6, data=data)
            except:
                continue
            if fd is None:
                return model, tracker, fd, grid, steps, level_ups
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(6, prev, grid, diff)
            changed = not np.array_equal(prev, grid)
            tracker.record_click(data['x'], data['y'], changed, int(np.sum(prev != grid)))
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                break

    return model, tracker, fd, grid, steps, level_ups


# ─── PARSE (same as v8) ───

def parse_actions(plan_text, tracker, available):
    name_map = {}
    for obj in tracker.objects.values():
        name_map[obj_name(obj).upper()] = obj
        cname = color_name(obj.color).upper()
        if cname not in name_map:
            name_map[cname] = []
        if isinstance(name_map.get(cname), list):
            name_map[cname].append(obj)

    actions = []
    for line in plan_text.split("\n"):
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        upper = line.upper().strip()
        if not upper:
            continue

        m = re.match(r'REPEAT\s+(\d+)\s+(.+)', upper)
        if m:
            n = min(int(m.group(1)), 30)
            sub = parse_actions(m.group(2), tracker, available)
            actions.extend(sub * n)
            continue

        m = re.match(r'CLICK\s+ALL\s+(\w+)', upper)
        if m and 6 in available:
            objs = name_map.get(m.group(1), [])
            if isinstance(objs, list):
                for o in objs:
                    actions.append((6, {'x': o.cx, 'y': o.cy}, obj_name(o)))
            continue

        if 'CLICK INTERACTIVE' in upper and 6 in available:
            for o in tracker.get_interactive_objects():
                actions.append((6, {'x': o.cx, 'y': o.cy}, obj_name(o)))
            continue

        m = re.match(r'CLICK\s+(\w+_\d+)', upper)
        if m and 6 in available:
            obj = name_map.get(m.group(1))
            if obj and not isinstance(obj, list):
                actions.append((6, {'x': obj.cx, 'y': obj.cy}, obj_name(obj)))
            continue

        m = re.match(r'CLICK\s+(\w+)', upper)
        if m and 6 in available:
            objs = name_map.get(m.group(1), [])
            if isinstance(objs, list) and objs:
                o = objs[0]
                actions.append((6, {'x': o.cx, 'y': o.cy}, obj_name(o)))
            continue

        for name, num in [("UP", 1), ("DOWN", 2), ("LEFT", 3), ("RIGHT", 4)]:
            if name in upper and num in available:
                actions.append((num, None, name))
                break
        else:
            if any(w in upper for w in ["SELECT", "SUBMIT"]) and 5 in available:
                actions.append((5, None, "SELECT"))
            elif "UNDO" in upper and 7 in available:
                actions.append((7, None, "UNDO"))

    return actions


# ─── CONTEXT ASSEMBLY + PLANNING ───

def assemble_context_and_plan(narrative, tracker, action_model, available,
                              levels, win_levels, ctx: ContextConstructor = None,
                              attempt_num=1, verbose=False, fleet_context: str = ""):
    """Assemble all 4 layers with budget tracking, ask LLM for next actions."""
    budget = ContextBudget()
    has_move = any(a in available for a in [1, 2, 3, 4])

    # Layer 4: Metacognitive principles (fixed, ~500 tokens)
    layer4 = METACOGNITIVE
    budget.record("L4_metacognitive", layer4)

    # Layer 3: Situation-relevant membot memories (adaptive, target ~2K)
    patterns = narrative.detect_patterns()
    interactive_names = [obj_name(o) for o in tracker.get_interactive_objects()]
    situation = f"Playing game level {levels}/{win_levels}. "
    if interactive_names:
        situation += f"Interactive objects: {', '.join(interactive_names)}. "
    if "STABLE" in patterns:
        situation += "Grid similarity stable — not progressing. "
    if "WARNING" in patterns:
        situation += "Repeated clicking without level-up. Need different approach. "
    layer3 = ctx.build_layer3(situation) if ctx else ""

    n_events = len(narrative.events)
    if n_events > 8 and not any(e.leveled_up for e in narrative.events):
        stuck_advice = ctx.on_stuck(n_events, interactive_names, patterns) if ctx else ""
        if stuck_advice:
            layer3 = layer3 + "\n\n" + stuck_advice if layer3 else stuck_advice
    budget.record("L3_membot", layer3)

    # Layer 2: Game KB — action model + curated object catalog (target ~3K)
    layer2_model = action_model.describe()
    layer2_catalog = obj_catalog(tracker)
    layer2 = f"ACTION MODEL:\n{layer2_model}\n\nOBJECTS:\n{layer2_catalog}"
    budget.record("L2_game_kb", layer2)

    # Layer 1: Session narrative — expanding with compression (target ~5K)
    layer1 = narrative.to_context()
    budget.record("L1_narrative", layer1)

    # Log budget periodically
    if verbose and (n_events % 10 == 0 or n_events <= 3):
        print(f"    {budget.report()}")

    # Fleet knowledge (federated, from all machines)
    if fleet_context:
        budget.record("L3_federation", fleet_context)

    prompt = f"""{layer4}

{layer3}

{fleet_context}

GAME STATE: Level {levels}/{win_levels}, Attempt {attempt_num}

{layer2}

{layer1}

{"MOVEMENT available: UP/DOWN/LEFT/RIGHT." if has_move else "Click-only game."}

Based on EVERYTHING above — your metacognitive principles, cross-game insights,
what you've learned about this game's objects, and your recent action history —
what are the NEXT 3 actions?

Use exact object names: "CLICK cyan_23" (not "CLICK cyan")
Use REPEAT for repetition: "REPEAT 5 CLICK cyan_23"
Use CLICK INTERACTIVE to click all known-working objects.

NEXT 3 ACTIONS:"""

    budget.record("prompt_framing", prompt[len(layer4)+len(layer3)+len(layer2)+len(layer1):])
    return ask_llm(prompt)


# ─── MAIN SOLVE LOOP WITH ATP + COHERENCE LOGGING ───

def solve_game(arcade, game_id, max_attempts=5, budget=300, verbose=False, log_atp=False):
    prefix = game_id.split("-")[0]
    best_levels = 0

    # Context constructor
    ctx = ContextConstructor(prefix)

    # Federation
    fed = FederatedKnowledge()
    fleet_context = fed.build_context(prefix)
    if verbose:
        print(f"  {fed.summary()}")
        if fleet_context:
            print(f"  Fleet context: {len(fleet_context)} chars")

    # Metabolic controller (Thor Session #60)
    metabolic = None
    atp_log = []
    if log_atp and METABOLIC_AVAILABLE:
        metabolic = MetabolicController(
            initial_atp=100.0,
            max_atp=100.0,
            simulation_mode=True,  # Use cycle counts not wall time
            enable_circadian=False  # Simpler for ARC-AGI
        )
        if verbose:
            print(f"  ATP logging enabled (initial: {metabolic.atp_current:.1f})")

    # ★ NEW: Coherence tracker (Thor Session #65)
    coherence_tracker = CoherenceTracker(window_size=10)
    if verbose:
        print(f"  Golden ratio tracking: φ - 1 = {PHI_MINUS_1:.4f}")

    for attempt in range(max_attempts):
        ctx.new_attempt()
        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)
        available = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
        total_steps = 0

        if verbose:
            print(f"\n  Attempt {attempt+1}/{max_attempts} | Actions: {available} | Levels: 0/{fd.win_levels}")

        # PROBE
        probe_budget = min(budget // 3, 25)
        action_model, tracker, fd, grid, probe_steps, probe_levels = \
            probe(env, fd, grid, available, budget=probe_budget)
        total_steps += probe_steps

        # Initialize narrative
        narrative = SessionNarrative(grid)
        interactive = tracker.get_interactive_objects()

        if verbose:
            print(f"  Probe: {probe_steps} steps | {len(interactive)} interactive")
            for o in interactive:
                print(f"    ✓ {obj_name(o)} at ({o.cx},{o.cy}) — {o.click_effectiveness:.0%}")

        # ★ NEW: Update coherence with probe discoveries
        probe_objects = [obj_name(o) for o in interactive]
        patterns = narrative.detect_patterns()
        coherence_tracker.update(
            pattern_detected=patterns[:50] if patterns else None,
            new_objects=probe_objects,
            action_effective=(probe_levels > 0),
            current_level=fd.levels_completed if fd else 0,
            planning_failed=False
        )

        # Layer 3: query membot
        game_type = action_model.infer_game_type()
        game_start_ctx = ctx.on_game_start(available, game_type)
        probe_ctx = ctx.on_probe_complete(
            [obj_name(o) for o in interactive],
            action_model.describe(), game_type)
        if verbose and (game_start_ctx or probe_ctx):
            print(f"  Membot: {len(game_start_ctx)} + {len(probe_ctx)} chars")

        if fd is None or fd.state.name in ("WON", "LOST", "GAME_OVER"):
            if fd and fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
            continue

        # INTERLEAVED LOOP
        remaining = budget - total_steps
        replan_count = 0

        while remaining > 0 and replan_count < 20 and fd.state.name not in ("WON", "LOST", "GAME_OVER"):
            replan_count += 1

            # Log ATP before planning (Session #62)
            atp_before_plan = None
            metabolic_state_before = None
            if metabolic:
                snapshot = metabolic.get_metabolic_snapshot()
                atp_before_plan = snapshot['atp_current']
                metabolic_state_before = snapshot['state']

            # ★ NEW: Get coherence before planning (Session #65)
            coherence_snapshot = coherence_tracker.get_snapshot()

            # ASSEMBLE CONTEXT + PLAN
            t0 = time.time()
            plan_text = assemble_context_and_plan(
                narrative, tracker, action_model, available,
                fd.levels_completed, fd.win_levels, ctx=ctx,
                attempt_num=attempt+1, verbose=verbose,
                fleet_context=fleet_context)
            plan_actions = parse_actions(plan_text, tracker, available)
            plan_time = time.time() - t0

            if not plan_actions:
                interactive = tracker.get_interactive_objects()
                if interactive:
                    o = random.choice(interactive)
                    plan_actions = [(6, {'x': o.cx, 'y': o.cy}, obj_name(o))]
                else:
                    a = random.choice(available)
                    plan_actions = [(a, None, ACTION_NAMES.get(a, f"A{a}"))]

            if verbose:
                names = [pa[2] if len(pa) > 2 else str(pa[0]) for pa in plan_actions[:5]]
                atp_str = f" | ATP: {atp_before_plan:.1f}" if atp_before_plan else ""
                # ★ NEW: Show C and golden zone status
                C = coherence_snapshot['coherence']
                zone_str = " φ!" if coherence_snapshot['in_golden_zone'] else ""
                c_str = f" | C: {C:.3f}{zone_str}"
                print(f"  [{total_steps}] Plan ({plan_time:.0f}s{atp_str}{c_str}): {' → '.join(names)}")

            # EXECUTE 2-3 actions
            exec_count = min(3, len(plan_actions), remaining)
            actions_caused_change = []
            actions_leveled_up = []

            for i in range(exec_count):
                action_int = plan_actions[i][0]
                data = plan_actions[i][1]
                target_name = plan_actions[i][2] if len(plan_actions[i]) > 2 else ACTION_NAMES.get(action_int, f"A{action_int}")

                prev_grid = grid.copy()
                prev_levels = fd.levels_completed

                try:
                    if data:
                        fd = env.step(INT_TO_GAME_ACTION[action_int], data=data)
                    else:
                        fd = env.step(INT_TO_GAME_ACTION[action_int])
                except Exception:
                    continue

                if fd is None:
                    break

                grid = get_frame(fd)
                total_steps += 1
                remaining -= 1

                # Update tracker
                diff = tracker.update(grid)
                if action_int == 6 and data:
                    changed = not np.array_equal(prev_grid, grid)
                    tracker.record_click(data['x'], data['y'], changed,
                                         int(np.sum(prev_grid != grid)))

                # Record in narrative
                event = narrative.record(
                    total_steps, action_int, target_name, data,
                    grid, prev_levels, fd.levels_completed)

                if verbose and event.changed:
                    print(f"    {event.observation}")

                # Track action effectiveness
                if event.changed:
                    actions_caused_change.append(target_name)
                if event.leveled_up:
                    actions_leveled_up.append(target_name)

                # Level up!
                if fd.levels_completed > prev_levels:
                    if verbose:
                        print(f"    ★ LEVEL {fd.levels_completed}/{fd.win_levels}!")

                    # Store winning sequence
                    winning_actions = [ev.target for ev in narrative.events[-10:] if ev.changed]
                    ctx.on_level_up(fd.levels_completed, winning_actions)

                    # Re-probe
                    if remaining > 10:
                        narrative = SessionNarrative(grid)
                        action_model, tracker, fd, grid, ps, _ = \
                            probe(env, fd, grid, available, budget=min(10, remaining))
                        total_steps += ps
                        remaining -= ps
                    break

                # Update available
                new_avail = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
                if new_avail:
                    available = new_avail

                if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                    break

            # Update ATP after actions (Session #58 formula)
            if metabolic:
                salience = 0.5  # baseline
                if actions_leveled_up:
                    salience = 0.9  # very high salience
                elif actions_caused_change:
                    salience = 0.7  # high salience
                elif len(plan_actions) == 0:
                    salience = 0.3  # low salience (random/fallback)

                # Update metabolic controller
                atp_consumed = 2.0 + salience * 3.0  # Session #58 formula
                metabolic.update({
                    'atp_consumed': atp_consumed,
                    'attention_load': len(interactive),
                    'max_salience': salience,
                    'crisis_detected': False
                })

            # ★ NEW: Update coherence after actions (Session #65)
            current_patterns = narrative.detect_patterns()
            new_interactive = [obj_name(o) for o in tracker.get_interactive_objects()]
            coherence_tracker.update(
                pattern_detected=current_patterns[:50] if current_patterns else None,
                new_objects=new_interactive,
                action_effective=(len(actions_caused_change) > 0),
                current_level=fd.levels_completed if fd else 0,
                planning_failed=(len(plan_actions) == 0)
            )

            # ★ NEW: Log ATP + Coherence coupling (Session #65 - Golden Ratio Test)
            if metabolic and log_atp:
                coherence_after = coherence_tracker.get_snapshot()
                atp_log.append({
                    'step': total_steps,
                    'atp_before': atp_before_plan,
                    'state_before': metabolic_state_before,
                    'atp_after': round(metabolic.atp_current, 2),
                    'state_after': metabolic.current_state.value,
                    'salience_estimated': salience,
                    'actions_planned': len(plan_actions),
                    'actions_executed': exec_count,
                    'actions_caused_change': len(actions_caused_change),
                    'leveled_up': len(actions_leveled_up) > 0,
                    'plan_time': round(plan_time, 2),
                    # ★ NEW: Coherence measurements
                    'coherence_before': coherence_snapshot,
                    'coherence_after': coherence_after,
                })

        if fd and fd.levels_completed > best_levels:
            best_levels = fd.levels_completed

        if verbose:
            print(f"  Result: {fd.levels_completed if fd else 0}/{fd.win_levels if fd else '?'} in {total_steps} steps")
            patterns = narrative.detect_patterns()
            if patterns:
                print(f"  Patterns: {patterns[:120]}")
            # ★ NEW: Show final coherence
            final_c = coherence_tracker.get_snapshot()
            print(f"  Final C: {final_c['coherence']:.3f} (dist from φ-1: {final_c['distance_from_phi']:.3f})")

        # Game-end learning
        ctx.on_game_end(
            fd.levels_completed if fd else 0,
            fd.win_levels if fd else 0,
            narrative.detect_patterns(),
            narrative.object_summary())

        # Federation: store discoveries
        patterns = narrative.detect_patterns()
        obj_summary = narrative.object_summary()
        interactive = tracker.get_interactive_objects()
        game_type = action_model.infer_game_type()

        if interactive:
            fed.add_discovery(prefix,
                f"Interactive objects: {', '.join(obj_name(o) for o in interactive[:5])}. "
                f"Game type: {game_type}. {obj_summary[:100]}")

        if "STABLE" in patterns:
            fed.add_failure(prefix,
                f"Clicking interactive objects repeatedly doesn't advance levels. "
                f"Grid similarity stays stable. Need correct sequence, not just correct objects.")

        if fd and fd.levels_completed > 0:
            winning = [ev.target for ev in narrative.events if ev.leveled_up]
            fed.add_discovery(prefix,
                f"SOLVED Level {fd.levels_completed}: winning actions included {', '.join(winning[-5:])}. "
                f"Game type: {game_type}.")
            fed.add_strategy(game_type,
                f"On {prefix}: level solved by targeting {', '.join(winning[-3:])}.")

        if attempt == max_attempts - 1:
            fed.add_meta_insight(
                f"Game {prefix} ({game_type}): {patterns[:150]}")
            fed.save()

            # Store in membot
            ctx.store(
                f"SAGE game experience summary ({prefix}): Played {max_attempts} attempts. "
                f"Best: {best_levels}/{fd.win_levels if fd else '?'} levels. "
                f"Key learning: {patterns[:150]}. "
                f"Objects learned: {obj_summary[:150]}."
            )

    # Save ATP + Coherence log
    if atp_log and log_atp:
        log_file = f"arc-agi-3/experiments/logs/{prefix}_atp_coherence_log.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # ★ NEW: Include coherence trajectory analysis
        coherence_history = coherence_tracker.coherence_history

        with open(log_file, 'w') as f:
            json.dump({
                'game_id': game_id,
                'best_levels': best_levels,
                'win_levels': fd.win_levels if fd else 0,
                'atp_coherence_log': atp_log,
                'coherence_trajectory': coherence_history,
                'phi_target': PHI_MINUS_1,
                'session': 'thor-65-golden-ratio-validation'
            }, f, indent=2)
        if verbose:
            print(f"  ATP + Coherence log saved: {log_file}")

    return {"game_id": game_id, "best_levels": best_levels,
            "win_levels": fd.win_levels if fd else 0, "atp_log": atp_log,
            "coherence_history": coherence_tracker.coherence_history}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Solver v10 — Golden Ratio Validation")
    parser.add_argument("--game", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--model", default=None, help="Ollama model override")
    parser.add_argument("--log-atp", action="store_true", help="Enable ATP + Coherence logging")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    global MODEL
    _detect_model()
    if args.model:
        MODEL = args.model

    print("=" * 60)
    print(f"SAGE Solver v10 — Golden Ratio Validation ({MODEL})")
    print(f"Thinking: {'native' if _is_thinking_model() else 'disabled'}")
    if args.log_atp:
        print(f"ATP + Coherence logging: ENABLED (Thor Session #65)")
        print(f"Golden ratio target: φ - 1 = {PHI_MINUS_1:.6f}")
    print("=" * 60)

    print("\nWarming up...", end=" ", flush=True)
    t = ask_llm("ready")
    print("OK" if "error" not in t.lower() else f"WARN: {t[:40]}")

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in e.game_id]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\n{len(targets)} games, {args.attempts} attempts, budget={args.budget}\n")

    total_levels = 0
    scored = {}

    for env_info in targets:
        game_id = env_info.game_id
        prefix = game_id.split("-")[0]
        try:
            result = solve_game(arcade, game_id, max_attempts=args.attempts,
                               budget=args.budget, verbose=args.verbose, log_atp=args.log_atp)
        except Exception as e:
            if args.verbose:
                print(f"  {prefix}: CRASHED ({e})")
            continue

        levels = result["best_levels"]
        total_levels += levels
        if levels > 0:
            scored[prefix] = levels
            print(f"  ★ {prefix:6s}  {levels}/{result['win_levels']}")
        else:
            print(f"    {prefix:6s}  0/{result['win_levels']}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_levels} levels across {len(scored)} games")
    if scored:
        for g, l in sorted(scored.items(), key=lambda x: -x[1]):
            print(f"  {g}: {l}")


if __name__ == "__main__":
    main()
