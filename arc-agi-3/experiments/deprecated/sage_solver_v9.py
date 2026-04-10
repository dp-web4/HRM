#!/usr/bin/env python3
"""
SAGE Puzzle Solver v9 — Multimodal Vision for Gemma 4.

Extends v7 with native multimodal vision support. Instead of text descriptions
of the grid, we send the actual grid IMAGE to gemma4:e4b's vision system.

The model SEES the puzzle state directly, allowing it to perceive spatial
relationships, colors, and patterns that text descriptions cannot capture.

Key changes from v7:
- grid_to_png_base64(): Converts 64x64 ARC grid → base64 PNG image
- ask_llm() accepts optional grid_image parameter for vision queries
- Planning calls pass current grid state as image, not text

Requires gemma4:e4b or other multimodal vision model.

Usage:
    cd ~/ai-workspace/SAGE
    .venv-arc/bin/python arc-agi-3/experiments/sage_solver_v9.py --game lp85 -v
    .venv-arc/bin/python arc-agi-3/experiments/sage_solver_v9.py --all --attempts 5
"""

import sys, os, time, json, re, random, base64, io
import numpy as np
import requests
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, background_color, color_name, find_color_regions
from arc_spatial import SpatialTracker
from arc_action_model import ActionEffectModel
from arc_narrative import SessionNarrative
from arc_context import ContextConstructor
from arc_federation import FederatedKnowledge
from arc_session_writer import SessionWriter

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


# Context budget targets (from context-budget-and-raising.md)
TARGET_BUDGET = {
    "layer4_metacognitive": 500,   # tokens — stable, rarely changes
    "layer3_membot": 2000,         # tokens — adaptive, more when uncertain
    "layer2_game_kb": 3000,        # tokens — growing, bounded by curation
    "layer1_narrative": 5000,      # tokens — expanding, compressed when large
    # Total: ~10.5K tokens. Leaves ~120K for reasoning.
}

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT", 6: "CLICK", 7: "UNDO"}


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


# ─── VISION: Grid → PNG Image ───

# ARC grid color palette (matches official ARC colors)
ARC_COLORS = np.array([
    [0, 0, 0],         # 0: black
    [0, 116, 217],     # 1: blue
    [255, 65, 54],     # 2: red
    [46, 204, 64],     # 3: green
    [255, 220, 0],     # 4: yellow
    [170, 170, 170],   # 5: gray
    [240, 18, 190],    # 6: magenta
    [255, 133, 27],    # 7: orange
    [127, 219, 255],   # 8: cyan
    [135, 12, 0],      # 9: brown
    [246, 164, 255],   # 10: pink
    [128, 0, 0],       # 11: maroon
    [128, 128, 0],     # 12: olive
    [0, 0, 128],       # 13: navy
    [0, 128, 128],     # 14: teal
    [255, 255, 255],   # 15: white
], dtype=np.uint8)

def grid_to_png_base64(grid: np.ndarray, scale: int = 8) -> str:
    """Convert ARC grid (64x64 int8) to base64-encoded PNG image.

    Args:
        grid: 64x64 numpy array with values 0-15 (ARC color indices)
        scale: Upscale factor (8 → 512x512 image for better vision)

    Returns:
        Base64-encoded PNG string suitable for Ollama vision API
    """
    if grid.ndim == 3:  # Handle (1, 64, 64) from SDK
        grid = grid[0]

    # Map grid indices → RGB colors
    rgb = ARC_COLORS[grid.astype(np.int32) % 16]

    # Upscale for better vision perception (64x64 → 512x512)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    # Convert to PIL Image and encode as PNG
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ─── LLM ───

def ask_llm(prompt: str, max_tokens: int = -1, grid_image: str = None) -> str:
    """Query LLM with optional grid image for multimodal vision.

    Args:
        prompt: Text prompt
        max_tokens: Max response tokens (-1 = model default)
        grid_image: Optional base64-encoded PNG for vision models

    Returns:
        LLM response text
    """
    opts = {"temperature": 0.3}
    if max_tokens == -1 and any(s in MODEL for s in ["0.8b", "0.5b", "1b", "2b", "3b"]):
        opts["num_predict"] = 300
    elif max_tokens > 0:
        opts["num_predict"] = max_tokens

    message = {"role": "user", "content": prompt}
    if grid_image:
        message["images"] = [grid_image]

    payload = {
        "model": MODEL, "stream": False,
        "messages": [message],
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


# ─── LAYER 3: Membot via ContextConstructor (see arc_context.py) ───
# Replaced static membot_recall/store with adaptive ContextConstructor


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


# ─── PROBE (same as v5) ───

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


# ─── PARSE ───

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

    # Layer 2: Game KB — scene description + action model + object catalog (target ~3K)
    # Scene description gives spatial understanding (what the screen looks like)
    layer2_scene = ""
    try:
        from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation
        gv = GridVisionIRP.__new__(GridVisionIRP)  # lightweight — just need describe_scene
        gv._buffer = []
        gv._frame_count = 0
        gv._prev_frame = None
        # Build observation from tracker's current state
        grid = tracker.prev_grid if hasattr(tracker, 'prev_grid') and tracker.prev_grid is not None else None
        if grid is not None:
            objects_list = [{"id": o.id, "color": o.color,
                           "bbox": [o.y, o.x, o.y + o.h, o.x + o.w],
                           "centroid": [o.cy, o.cx], "size": o.size}
                          for o in tracker.objects.values()]
            obs = GridObservation(frame_raw=grid, objects=objects_list,
                                changes=[], moved=[],
                                step_number=0, level_id="")
            layer2_scene = "SCENE (what the screen looks like):\n" + gv.describe_scene(obs)
    except Exception:
        pass  # graceful fallback — scene description is additive, not required

    layer2_model = action_model.describe()
    layer2_catalog = obj_catalog(tracker)
    layer2 = f"{layer2_scene}\n\nACTION MODEL:\n{layer2_model}\n\nOBJECTS:\n{layer2_catalog}" if layer2_scene else f"ACTION MODEL:\n{layer2_model}\n\nOBJECTS:\n{layer2_catalog}"
    budget.record("L2_game_kb", layer2)

    # Layer 1: Session narrative — expanding with compression (target ~5K)
    layer1 = narrative.to_context()
    # If Layer 1 exceeds budget, the narrative's internal compression handles it
    budget.record("L1_narrative", layer1)

    # Log budget periodically
    if verbose and (n_events % 10 == 0 or n_events <= 3):
        print(f"    {budget.report()}")

    # Fleet knowledge (federated, from all machines)
    if fleet_context:
        budget.record("L3_federation", fleet_context)

    # Generate grid image for multimodal vision
    grid_image = None
    if tracker.prev_grid is not None:
        try:
            grid_image = grid_to_png_base64(tracker.prev_grid)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not generate grid image: {e}")

    prompt = f"""{layer4}

{layer3}

{fleet_context}

GAME STATE: Level {levels}/{win_levels}, Attempt {attempt_num}

{layer2}

{layer1}

{"MOVEMENT available: UP/DOWN/LEFT/RIGHT." if has_move else "Click-only game."}

{"[IMAGE ATTACHED: Current game grid state]" if grid_image else ""}

Based on EVERYTHING above — your metacognitive principles, cross-game insights,
what you've learned about this game's objects, and your recent action history —
{"and LOOKING AT THE IMAGE showing the current game state — " if grid_image else ""}what are the NEXT 3 actions?

Use exact object names: "CLICK cyan_23" (not "CLICK cyan")
Use REPEAT for repetition: "REPEAT 5 CLICK cyan_23"
Use CLICK INTERACTIVE to click all known-working objects.

NEXT 3 ACTIONS:"""

    budget.record("prompt_framing", prompt[len(layer4)+len(layer3)+len(layer2)+len(layer1):])
    return ask_llm(prompt, grid_image=grid_image)


# ─── MAIN SOLVE LOOP ───

def solve_game(arcade, game_id, max_attempts=5, budget=300, verbose=False):
    prefix = game_id.split("-")[0]
    best_levels = 0

    # Context constructor — adaptive membot queries throughout the game
    ctx = ContextConstructor(prefix)

    # Federation — load knowledge from all fleet machines
    fed = FederatedKnowledge()
    fleet_context = fed.build_context(prefix)
    if verbose:
        print(f"  {fed.summary()}")
        if fleet_context:
            print(f"  Fleet context: {len(fleet_context)} chars")

    # Session writer for live visualization
    session_writer = None

    for attempt in range(max_attempts):
        ctx.new_attempt()  # Bust cache — fresh membot queries for new attempt
        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)
        available = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
        total_steps = 0

        # Initialize session writer on first attempt, update on subsequent
        if session_writer is None:
            try:
                env_info = [e for e in arcade.get_environments() if e.game_id == game_id][0]
                baseline = sum(env_info.baseline_actions) if hasattr(env_info, 'baseline_actions') else 0
            except:
                baseline = 0
            session_writer = SessionWriter(game_id, fd.win_levels, available,
                                          baseline=baseline, player=f"gemma4:{MODEL}")
            session_writer.save_grid(grid, "current")
            session_writer.save_grid(grid, "previous")
        else:
            session_writer.new_attempt(attempt + 1)
            session_writer.save_grid(grid, "current")

        if verbose:
            print(f"\n  Attempt {attempt+1}/{max_attempts} | Actions: {available} | Levels: 0/{fd.win_levels}")

        # PROBE
        probe_budget = min(budget // 3, 25)
        action_model, tracker, fd, grid, probe_steps, probe_levels = \
            probe(env, fd, grid, available, budget=probe_budget)
        total_steps += probe_steps

        # Initialize narrative from probe
        narrative = SessionNarrative(grid)

        interactive = tracker.get_interactive_objects()

        if verbose:
            print(f"  Probe: {probe_steps} steps | {len(interactive)} interactive")
            for o in interactive:
                print(f"    ✓ {obj_name(o)} at ({o.cx},{o.cy}) — {o.click_effectiveness:.0%}")

        # Layer 3: query membot with probe results
        game_type = action_model.infer_game_type()
        game_start_ctx = ctx.on_game_start(available, game_type)
        probe_ctx = ctx.on_probe_complete(
            [obj_name(o) for o in interactive],
            action_model.describe(), game_type)
        if verbose and (game_start_ctx or probe_ctx):
            print(f"  Membot: {len(game_start_ctx)} + {len(probe_ctx)} chars of relevant context")

        if fd is None or fd.state.name in ("WON", "LOST", "GAME_OVER"):
            if fd and fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
            continue

        # INTERLEAVED LOOP
        remaining = budget - total_steps
        replan_count = 0

        while remaining > 0 and replan_count < 20 and fd.state.name not in ("WON", "LOST", "GAME_OVER"):
            replan_count += 1

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
                print(f"  [{total_steps}] Plan ({plan_time:.0f}s): {' → '.join(names)}")

            # EXECUTE 2-3 actions
            exec_count = min(3, len(plan_actions), remaining)
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

                # Record in narrative (Layer 1)
                event = narrative.record(
                    total_steps, action_int, target_name, data,
                    grid, prev_levels, fd.levels_completed)

                # Record in session writer for viewer
                if session_writer:
                    x = data.get('x') if data else None
                    y = data.get('y') if data else None
                    session_writer.record_action(action_int, x, y,
                                                 observation=event.observation if event else "",
                                                 grid=grid)

                if verbose and event.changed:
                    print(f"    {event.observation}")

                # Level up!
                if fd.levels_completed > prev_levels:
                    if verbose:
                        print(f"    ★ LEVEL {fd.levels_completed}/{fd.win_levels}!")

                    # Store the winning sequence via context constructor
                    winning_actions = [ev.target for ev in narrative.events[-10:] if ev.changed]
                    ctx.on_level_up(fd.levels_completed, winning_actions)

                    # Record level-up in session writer
                    if session_writer:
                        session_writer.record_level_up(fd.levels_completed, winning_actions)

                    # Re-probe for new level
                    if remaining > 10:
                        narrative = SessionNarrative(grid)  # fresh narrative for new level
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
                    if session_writer:
                        session_writer.record_game_end(fd.state.name)
                    break

        if fd and fd.levels_completed > best_levels:
            best_levels = fd.levels_completed

        if verbose:
            print(f"  Result: {fd.levels_completed if fd else 0}/{fd.win_levels if fd else '?'} in {total_steps} steps")
            patterns = narrative.detect_patterns()
            if patterns:
                print(f"  Patterns: {patterns[:120]}")

        # Game-end learning: abstract patterns for future sessions
        ctx.on_game_end(
            fd.levels_completed if fd else 0,
            fd.win_levels if fd else 0,
            narrative.detect_patterns(),
            narrative.object_summary())

        # Federation: store discoveries for other machines
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
            fed.save()  # Write to git-tracked file

            # Also store in membot for raising bridge
            ctx.store(
                f"SAGE game experience summary ({prefix}): Played {max_attempts} attempts. "
                f"Best: {best_levels}/{fd.win_levels if fd else '?'} levels. "
                f"Key learning: {patterns[:150]}. "
                f"Objects learned: {obj_summary[:150]}."
            )

    return {"game_id": game_id, "best_levels": best_levels,
            "win_levels": fd.win_levels if fd else 0}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Solver v6 — Context-Shaped Intelligence")
    parser.add_argument("--game", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--model", default=None, help="Ollama model override")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    global MODEL
    _detect_model()
    if args.model:
        MODEL = args.model

    print("=" * 60)
    print(f"SAGE Solver v9 — Multimodal Vision + Viewer ({MODEL})")
    print(f"Thinking: {'native' if _is_thinking_model() else 'disabled'}")
    print("Vision-enabled with grid PNGs + Live visualization via SessionWriter")
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
                               budget=args.budget, verbose=args.verbose)
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
