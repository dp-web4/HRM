#!/usr/bin/env python3
"""llm_dispatch — the router's invoke head meets an actual LLM.

At each game step:
  1. NN observes state, runs v4 adapter → (decision, play_action, hint_ranking)
  2. If decision == "play" → env.step(play_action, center-click if CLICK)
  3. If decision == "invoke" → render prev + curr frames as PNG, build a
     context+hint prompt, POST to a local vision LLM (Ollama by default),
     parse the response for ACTION=<n> [X=<x> Y=<y>], env.step that

This is the first time the invoke head's output actually gets consumed by
an LLM. The LLM sees the frames, considers the NN's hint, and either
confirms or overrides. Its response is logged for later analysis.

Backend: Ollama HTTP API at localhost:11434 by default, configurable via
SAGE_LLM_MODEL and SAGE_LLM_BASE_URL env vars. Default model
llama3.2-vision:11b — US-origin, vision-capable, fits in 8GB VRAM.

Spec: shared-context/arc-agi-3/phase2/brain-arch/phase-2-world-model-embedding-sprint.md
      (this is the Phase-3 follow-on: router + LLM consuming its hint)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sage.cognition.router.data import RouterDatasetWriter
from sage.cognition.router.record import RouterRecord
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.thalamic_router.frame_router import (
    FrameRouter, FrameRouterConfig, load_frame_router,
    onehot_frame, build_scalar_vector,
    N_ACTIONS, ACTION_NAMES, FRAME_H, FRAME_W, N_COLORS, RECENT_ACTIONS_K,
)
from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, _discover_trace, ARC_AGI_EXPERIMENTS,
)


# Default thresholds for the dispatch decision
INVOKE_THRESHOLD = 0.5
PLAY_CONFIDENCE_THRESHOLD = 0.55
PLAY_MARGIN_THRESHOLD = 0.05
STUCK_WINDOW = 5
STUCK_FRAME_EPS = 1e-3
PROGRESS_STALL_WINDOW = 30

# Canonical ARC-AGI-3 palette for rendering frames to PNG for the vision LLM.
# 16 distinct colors indexed 0-15.
_PALETTE_RGB = [
    (0, 0, 0),          # 0 black
    (0, 116, 217),      # 1 blue
    (255, 65, 54),      # 2 red
    (46, 204, 64),      # 3 green
    (255, 220, 0),      # 4 yellow
    (170, 170, 170),    # 5 grey
    (240, 18, 190),     # 6 magenta
    (255, 133, 27),     # 7 orange
    (127, 219, 255),    # 8 cyan
    (139, 69, 19),      # 9 brown
    (128, 0, 128),      # 10 purple
    (0, 128, 128),      # 11 teal
    (255, 192, 203),    # 12 pink
    (0, 255, 0),        # 13 lime
    (100, 100, 100),    # 14 dark grey
    (255, 255, 255),    # 15 white
]


def _frame_int_from_onehot(oh: np.ndarray) -> np.ndarray:
    """Inverse of onehot_frame — recover the integer grid (H, W)."""
    return oh.argmax(axis=0).astype(np.int32)


def _frame_to_rgb(frame_oh_or_raw: Any) -> np.ndarray:
    arr = np.asarray(frame_oh_or_raw)
    if arr.ndim == 3 and arr.shape[0] == N_COLORS:
        grid = _frame_int_from_onehot(arr.astype(np.float32))
    elif arr.ndim == 3:
        grid = arr[-1].astype(np.int32)
    elif arr.ndim == 2:
        grid = arr.astype(np.int32)
    else:
        grid = np.zeros((FRAME_H, FRAME_W), dtype=np.int32)
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(_PALETTE_RGB):
        mask = (grid == i)
        rgb[mask] = (r, g, b)
    return rgb


def _upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def render_frame_png(frame_oh_or_raw: Any, scale: int = 4) -> bytes:
    """Render a single frame to PNG bytes (one-hot or raw int grid)."""
    rgb = _upscale(_frame_to_rgb(frame_oh_or_raw), scale)
    try:
        from PIL import Image
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode()
        return header + rgb.tobytes()


def render_frame_pair_png(
    prev_frame: Any, curr_frame: Any, scale: int = 4, gap: int = 8,
) -> bytes:
    """Stitch prev and curr into one side-by-side PNG with a black gap between
    them. Many vision LLMs only accept one image per turn; this gets both
    frames to the model in a single attachment.

    Layout: [ PREV frame | gap | CURR frame ]
    """
    prev_rgb = _upscale(_frame_to_rgb(prev_frame), scale)
    curr_rgb = _upscale(_frame_to_rgb(curr_frame), scale)
    h = max(prev_rgb.shape[0], curr_rgb.shape[0])
    # Pad shorter to match height
    def _pad(rgb, target_h):
        if rgb.shape[0] == target_h:
            return rgb
        pad = np.zeros((target_h - rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
        return np.vstack([rgb, pad])
    prev_rgb = _pad(prev_rgb, h)
    curr_rgb = _pad(curr_rgb, h)
    # Black gap column + simple label strip on top with text would be nice but
    # PIL text rendering adds a dep we don't strictly need. The gap itself
    # plus the prompt's explicit "LEFT=PREV RIGHT=CURR" is enough.
    gap_col = np.zeros((h, gap, 3), dtype=np.uint8)
    stitched = np.hstack([prev_rgb, gap_col, curr_rgb])
    try:
        from PIL import Image
        img = Image.fromarray(stitched, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        header = f"P6\n{stitched.shape[1]} {stitched.shape[0]}\n255\n".encode()
        return header + stitched.tobytes()


# ───────────────────────────────────────────────────────────────────
# LLM client — Ollama HTTP API
# ───────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2-vision:11b",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self, prompt: str, images_png: Optional[List[bytes]] = None,
        max_tokens: int = 300,
    ) -> str:
        """POST to /api/chat with optional image attachments.
        Returns the assistant's response text."""
        import urllib.request, urllib.error
        message = {"role": "user", "content": prompt}
        if images_png:
            message["images"] = [base64.b64encode(b).decode("ascii") for b in images_png]
        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return f"ERROR: HTTP {e.code} — {e.read().decode('utf-8', 'ignore')[:200]}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        msg = body.get("message") or {}
        return str(msg.get("content", "") or "")


# ───────────────────────────────────────────────────────────────────
# Prompt + response parsing
# ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the deliberation tier of SAGE, a cognition kernel playing \
ARC-AGI-3 grid-based puzzle games. A small neural router has flagged this state as \
requiring deliberation. You see two images: the previous game frame and the current \
game frame. Cells are 16 colors. The grid is 64x64.

The neural network provides hints. Your job: pick the best next action to make \
progress toward winning the level."""


def build_prompt(
    game: str, level: int, step_index: int,
    play_action_idx: int, play_confidence: float,
    action_ranking: List[Tuple[int, float]],
    recent_actions: List[int],
    invoke_reasons: List[str],
) -> str:
    """Compose the invoke-time LLM prompt.

    Keeps it tight — the LLM is on a budget, don't waste its context.
    """
    top5 = action_ranking[:5]
    top_str = ", ".join(
        f"{ACTION_NAMES[a]}({p:.2f})" for a, p in top5
    )
    recent_names = " → ".join(
        ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)
        for a in recent_actions[-5:]
    ) or "(none yet)"

    return f"""{SYSTEM_PROMPT}

The image contains TWO frames side by side: LEFT is the PREVIOUS frame, \
RIGHT is the CURRENT frame (separated by a black gap). Compare them to \
understand what just changed.

Game: {game}  Level: {level}  Step: {step_index}
Invoke triggers: {', '.join(invoke_reasons) if invoke_reasons else 'manual'}
Recent actions: {recent_names}

NN's best-action ranking (top 5): {top_str}
NN's top pick: {ACTION_NAMES[play_action_idx]} (confidence {play_confidence:.2f})

Actions: A0=0 UP=1 DOWN=2 LEFT=3 RIGHT=4 SEL=5 CLICK=6

Respond with exactly this format on the first line:
ACTION=<0-6>[ X=<0-63> Y=<0-63>]
<one-sentence rationale>

If you choose CLICK (6), you MUST provide X and Y pixel coordinates on the 64×64 grid.
"""


# ACTION=<n> — required. Coords parsed independently (tolerates brackets,
# extra whitespace, commas, etc. that real LLMs produce).
_ACTION_RE = re.compile(r"ACTION\s*=\s*(\d+)", re.IGNORECASE)
_X_RE = re.compile(r"X\s*=\s*(-?\d+)", re.IGNORECASE)
_Y_RE = re.compile(r"Y\s*=\s*(-?\d+)", re.IGNORECASE)


def parse_llm_response(
    text: str, fallback_action: int, fallback_coords: Optional[Dict[str, int]] = None,
) -> Tuple[int, Optional[Dict[str, int]], str]:
    """Extract (action, coords_or_None, rationale) from the LLM's response.
    Falls back to the NN's hint if the response is malformed.
    """
    m = _ACTION_RE.search(text)
    if not m:
        return fallback_action, fallback_coords, f"parse_failed: {text[:120]}"
    try:
        action = int(m.group(1))
    except Exception:
        return fallback_action, fallback_coords, f"bad_action_int: {text[:120]}"
    if not (0 <= action < N_ACTIONS):
        return fallback_action, fallback_coords, f"action_out_of_range: {action}"

    coords: Optional[Dict[str, int]] = None
    if action == 6:
        xm = _X_RE.search(text); ym = _Y_RE.search(text)
        if xm and ym:
            try:
                x = max(0, min(FRAME_W - 1, int(xm.group(1))))
                y = max(0, min(FRAME_H - 1, int(ym.group(1))))
                coords = {"x": x, "y": y}
            except Exception:
                coords = fallback_coords
        else:
            coords = fallback_coords

    # Rationale — everything after the action line (first 200 chars)
    lines = text.strip().splitlines()
    rationale = ""
    for line in lines:
        if "ACTION" in line.upper():
            continue
        rationale = line.strip()
        if rationale:
            break
    return action, coords, rationale[:200]


# ───────────────────────────────────────────────────────────────────
# Dispatch decision — same structure as choose_dispatch in world_model.py,
# adapted for FrameRouter
# ───────────────────────────────────────────────────────────────────

@dataclass
class DispatchDecision:
    decision: str                         # "play" | "invoke"
    invoke_reasons: List[str]
    invoke_prob: float
    play_action: int
    play_confidence: float
    play_margin: float
    action_ranking: List[Tuple[int, float]]


def choose_dispatch(
    model: FrameRouter, cfg: FrameRouterConfig,
    prev_frame_oh: np.ndarray, curr_frame_oh: np.ndarray,
    scalar: np.ndarray, device: str = "cpu",
    invoke_threshold: float = INVOKE_THRESHOLD,
    play_confidence_threshold: float = PLAY_CONFIDENCE_THRESHOLD,
    play_margin_threshold: float = PLAY_MARGIN_THRESHOLD,
) -> DispatchDecision:
    model.eval()
    with torch.no_grad():
        prev = torch.from_numpy(prev_frame_oh).unsqueeze(0).to(device)
        curr = torch.from_numpy(curr_frame_oh).unsqueeze(0).to(device)
        sc = torch.from_numpy(scalar).unsqueeze(0).to(device)
        out = model(prev, curr, sc)
        action_probs = F.softmax(out["action_logits"], dim=-1)[0]
        invoke_prob = float(torch.sigmoid(out["invoke_logit"]).item())

    sorted_idx = torch.argsort(action_probs, descending=True)
    top_action = int(sorted_idx[0].item())
    top_conf = float(action_probs[top_action].item())
    second_conf = float(action_probs[int(sorted_idx[1].item())].item())
    margin = top_conf - second_conf
    ranking = [
        (int(i.item()), float(action_probs[i].item())) for i in sorted_idx
    ]

    reasons: List[str] = []
    if invoke_prob > invoke_threshold:
        reasons.append("novelty")
    if top_conf < play_confidence_threshold:
        reasons.append("low_confidence")
    if margin < play_margin_threshold:
        reasons.append("tight_margin")

    return DispatchDecision(
        decision="invoke" if reasons else "play",
        invoke_reasons=reasons, invoke_prob=invoke_prob,
        play_action=top_action, play_confidence=top_conf, play_margin=margin,
        action_ranking=ranking,
    )


# ───────────────────────────────────────────────────────────────────
# Harness
# ───────────────────────────────────────────────────────────────────

@dataclass
class LlmDispatchResult:
    game: str
    game_id: str
    n_steps: int
    max_steps: int
    final_state: Optional[str]
    final_levels: Optional[int]
    outcome: str
    solver_steps: Optional[int] = None
    action_counts: Dict[str, int] = field(default_factory=dict)
    invoke_count: int = 0
    llm_calls: int = 0
    llm_parse_failures: int = 0
    stuck_count: int = 0
    llm_responses: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _make_env(game_family: str, game_id: str):
    if str(ARC_AGI_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(ARC_AGI_EXPERIMENTS))
    from arc_agi import Arcade
    arc = Arcade(operation_mode="offline")
    env = arc.make(game_id) or arc.make(game_family)
    if env is None:
        raise RuntimeError(f"arc.make None for {game_id}/{game_family}")
    fd = env.reset()
    return env, fd


def run_llm_dispatch(
    model: FrameRouter, cfg: FrameRouterConfig,
    game_family: str, game_id: str,
    writer: RouterDatasetWriter, machine: str,
    llm_client: OllamaClient,
    max_steps: int, trace_steps: Optional[int] = None,
    device: str = "cpu", log_every_llm: bool = True,
) -> LlmDispatchResult:
    errors: List[str] = []
    try:
        env, fd = _make_env(game_family, game_id)
    except Exception as e:
        return LlmDispatchResult(
            game=game_family, game_id=game_id, n_steps=0,
            max_steps=max_steps, final_state=None, final_levels=None,
            outcome="ENV_INIT_FAILED", errors=[f"env init: {e!r}"],
        )

    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}

    model = model.to(device)
    game_slugs = cfg.game_slugs
    game_idx = game_slugs.index(game_family) if game_family in game_slugs else 0

    action_counts: Dict[str, int] = {}
    invoke_count = 0
    llm_calls = 0
    llm_parse_failures = 0
    stuck_count = 0
    llm_responses: List[Dict[str, Any]] = []

    # Zero-frame convention at game start
    zero_frame_oh = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    prev_frame_oh = zero_frame_oh
    prev_frame_raw = None
    last_action: int = 0
    recent: deque = deque([0] * RECENT_ACTIONS_K, maxlen=RECENT_ACTIONS_K)

    # Stuck tracking
    recent_action_win: deque = deque(maxlen=STUCK_WINDOW)
    recent_frame_win: deque = deque(maxlen=STUCK_WINDOW)
    steps_since_progress = 0
    last_levels = 0

    step_idx = 0
    outcome_terminal = {"WIN", "GAME_OVER"}

    while step_idx < max_steps:
        curr_frame_raw = getattr(fd, "frame", None)
        curr_frame_oh = onehot_frame(curr_frame_raw) if curr_frame_raw is not None else zero_frame_oh
        curr_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
        curr_levels = getattr(fd, "levels_completed", 0) or 0

        level = curr_levels
        step_frac = min(1.0, (step_idx + 1) / max(1, trace_steps or 100))
        budget_remaining = max(0.0, 1.0 - step_frac)

        scalar = build_scalar_vector(
            game_idx=game_idx, n_games=cfg.n_games,
            level=level, n_levels=cfg.n_levels,
            step_frac=step_frac, budget_remaining=budget_remaining,
            last_action=last_action,
            recent_actions=list(recent),
            available_actions=[1] * N_ACTIONS,
            batch_state=[0.0, 0.0, 0.0],
            include_last_action=cfg.include_last_action,
        )
        scalar_arr = np.array(scalar, dtype=np.float32)

        dispatch = choose_dispatch(
            model, cfg, prev_frame_oh, curr_frame_oh, scalar_arr, device=device,
        )

        # Harness-level stuck detection
        stuck_triggered = False
        if len(recent_action_win) == STUCK_WINDOW:
            same_actions = len(set(recent_action_win)) == 1
            frames_same = all(
                np.array_equal(recent_frame_win[0], recent_frame_win[i])
                for i in range(1, len(recent_frame_win))
            )
            if same_actions and frames_same:
                stuck_triggered = True
                stuck_count += 1
        if steps_since_progress >= PROGRESS_STALL_WINDOW:
            stuck_triggered = True
            stuck_count += 1
        if stuck_triggered and dispatch.decision != "invoke":
            dispatch.decision = "invoke"
            dispatch.invoke_reasons.append("stuck")

        # Resolve action
        llm_info: Optional[Dict[str, Any]] = None
        if dispatch.decision == "invoke":
            invoke_count += 1
            llm_calls += 1
            # Stitch prev+curr into one image (llama3.2-vision accepts 1 image)
            pair_png = render_frame_pair_png(prev_frame_oh, curr_frame_oh)
            prompt = build_prompt(
                game=game_family, level=level, step_index=step_idx + 1,
                play_action_idx=dispatch.play_action,
                play_confidence=dispatch.play_confidence,
                action_ranking=dispatch.action_ranking,
                recent_actions=list(recent),
                invoke_reasons=dispatch.invoke_reasons,
            )
            t0 = time.time()
            response = llm_client.chat(prompt, images_png=[pair_png])
            llm_latency = time.time() - t0

            action, coords, rationale = parse_llm_response(
                response,
                fallback_action=dispatch.play_action,
                fallback_coords={"x": 32, "y": 32} if dispatch.play_action == 6 else None,
            )
            parse_ok = not rationale.startswith("parse_failed")
            if not parse_ok:
                llm_parse_failures += 1

            llm_info = {
                "prompt_preview": prompt[-300:],
                "response": response[:500],
                "rationale": rationale,
                "action": action, "coords": coords,
                "latency_s": llm_latency,
                "parse_ok": parse_ok,
            }
            if log_every_llm:
                llm_responses.append(llm_info)
        else:
            action = dispatch.play_action
            coords = {"x": 32, "y": 32} if action == 6 else None

        aname = ACTION_NAMES[action] if 0 <= action < N_ACTIONS else str(action)
        action_counts[aname] = action_counts.get(aname, 0) + 1

        # Advance env
        try:
            ga = int_to_action.get(action)
            if ga is None:
                errors.append(f"step {step_idx}: no enum for {action}")
                break
            new_fd = env.step(ga, data=coords) if coords else env.step(ga)
        except Exception as e:
            errors.append(f"step {step_idx}: env.step({aname}) failed: {e!r}")
            break

        new_state = getattr(getattr(new_fd, "state", None), "name", None) or "RUNNING"
        new_levels = getattr(new_fd, "levels_completed", 0) or 0

        metadata = {
            "source": "sage_plays_self",
            "game": game_family, "game_id": game_id,
            "step_index": step_idx + 1, "synthetic_kernel_state": True,
            "sage_plays_self": {
                "decision": dispatch.decision,
                "invoke_reasons": dispatch.invoke_reasons,
                "invoke_prob": dispatch.invoke_prob,
                "play_action": dispatch.play_action,
                "play_confidence": dispatch.play_confidence,
                "action_ranking": dispatch.action_ranking[:5],
                "applied_action": action,
                "applied_coords": coords,
                "llm_invoked": llm_info is not None,
                "llm_action": llm_info["action"] if llm_info else None,
                "llm_rationale": llm_info["rationale"] if llm_info else None,
                "llm_latency_s": llm_info["latency_s"] if llm_info else None,
                "state_before": curr_state, "state_after": new_state,
                "levels_before": curr_levels, "levels_after": new_levels,
                "stuck_triggered": stuck_triggered,
                "planner": "llm_dispatch_v4",
            },
        }
        # Bypass RouterRecord — we don't have a full RouterInput to attach
        # for this stream. Write a minimal dict directly.
        rec_dict = {
            "record_id": f"llm_{machine}_{int(time.time()*1000):013d}_{step_idx:05d}",
            "schema_version": "v0.2.0",
            "timestamp": time.time(),
            "machine": machine,
            "router_input": {
                "tick": step_idx + 1, "game_family": game_family,
                "level": curr_levels, "step_index": step_idx + 1,
            },
            "router_output": {"action": "noop", "rationale_code": "llm_dispatch"},
            "outcome": None,
            "metadata": metadata,
        }
        try:
            writer.append(rec_dict)
        except Exception as e:
            errors.append(f"step {step_idx}: write failed: {e!r}")

        # Update state
        recent.append(action)
        last_action = int(action)
        recent_action_win.append(action)
        recent_frame_win.append(_frame_int_from_onehot(curr_frame_oh))
        if new_levels > last_levels:
            steps_since_progress = 0; last_levels = new_levels
        else:
            steps_since_progress += 1

        fd = new_fd
        prev_frame_oh = curr_frame_oh
        prev_frame_raw = curr_frame_raw
        step_idx += 1

        if new_state in outcome_terminal:
            return LlmDispatchResult(
                game=game_family, game_id=game_id, n_steps=step_idx,
                max_steps=max_steps, final_state=new_state,
                final_levels=new_levels, outcome=new_state,
                solver_steps=trace_steps, action_counts=action_counts,
                invoke_count=invoke_count, llm_calls=llm_calls,
                llm_parse_failures=llm_parse_failures, stuck_count=stuck_count,
                llm_responses=llm_responses, errors=errors,
            )

    final_state = getattr(getattr(fd, "state", None), "name", None) or "RUNNING"
    final_levels = getattr(fd, "levels_completed", 0) or 0
    return LlmDispatchResult(
        game=game_family, game_id=game_id, n_steps=step_idx,
        max_steps=max_steps, final_state=final_state,
        final_levels=final_levels, outcome="MAX_STEPS",
        solver_steps=trace_steps, action_counts=action_counts,
        invoke_count=invoke_count, llm_calls=llm_calls,
        llm_parse_failures=llm_parse_failures, stuck_count=stuck_count,
        llm_responses=llm_responses, errors=errors,
    )


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="FrameRouter adapter base path.")
    p.add_argument("--game", required=True)
    p.add_argument("--game-id", default=None)
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"))
    p.add_argument("--data-dir", default=os.environ.get(
        "SAGE_ROUTER_DATA_DIR",
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--llm-model", default=os.environ.get("SAGE_LLM_MODEL", "llama3.2-vision:11b"))
    p.add_argument("--llm-url", default=os.environ.get("SAGE_LLM_BASE_URL", "http://localhost:11434"))
    p.add_argument("--llm-timeout", type=float, default=60.0)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    # Resolve game_id
    game_id = args.game_id
    if not game_id:
        for root in [Path(os.environ.get("ARC_SAGE_DIR", "")),
                     Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE"),
                     Path.home() / "ai-workspace" / "ARC-SAGE",
                     Path.home() / "repos" / "ARC-SAGE"]:
            coord = root / "knowledge" / "game_coordination.json"
            if coord.exists():
                try:
                    for g in json.loads(coord.read_text()).get("games", []):
                        if g.get("family") == args.game:
                            game_id = g.get("id"); break
                except Exception:
                    pass
                if game_id: break
    if not game_id:
        print(f"ERROR: could not resolve game_id for {args.game}")
        return 1

    # Trace for step budget
    trace_steps = None
    try:
        tp = _discover_trace(args.game, game_id)
        if tp and tp.exists():
            tr = load_trace(tp, args.game, game_id)
            trace_steps = len(tr.steps)
    except Exception:
        pass
    max_steps = args.max_steps or (min(2 * trace_steps, 500) if trace_steps else 200)

    model, cfg = load_frame_router(Path(args.adapter))
    llm = OllamaClient(base_url=args.llm_url, model=args.llm_model, timeout=args.llm_timeout)

    # Probe LLM availability
    probe = llm.chat("Reply with just OK.", max_tokens=10)
    if probe.startswith("ERROR"):
        print(f"LLM probe failed: {probe}")
        print(f"  model: {args.llm_model} at {args.llm_url}")
        return 1
    print(f"LLM online: {args.llm_model} (probe: {probe[:60]!r})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = RouterDatasetWriter(
        base_dir=Path(args.data_dir), machine=args.machine,
        compress=True, subdir="llm_dispatch",
    )
    try:
        result = run_llm_dispatch(
            model, cfg, args.game, game_id, writer, args.machine, llm,
            max_steps=max_steps, trace_steps=trace_steps, device=device,
        )
    finally:
        writer.close()

    print("=" * 60)
    print(f"llm_dispatch — {args.game} — {args.machine}")
    print(f"  LLM model: {args.llm_model}")
    print("=" * 60)
    print(f"  Solver ref : {trace_steps} steps")
    print(f"  SAGE took  : {result.n_steps} / {max_steps}")
    print(f"  Final      : {result.final_state}  levels={result.final_levels}")
    print(f"  Outcome    : {result.outcome}")
    print(f"  Actions    : {result.action_counts}")
    print(f"  Invokes    : {result.invoke_count}/{result.n_steps} "
          f"({100*result.invoke_count/max(result.n_steps,1):.1f}%)")
    print(f"  LLM calls  : {result.llm_calls}  (parse_fail={result.llm_parse_failures})")
    print(f"  Stuck hits : {result.stuck_count}")
    if result.llm_responses:
        print(f"\n  First LLM response:")
        r = result.llm_responses[0]
        print(f"    rationale: {r.get('rationale','')}")
        print(f"    action:    {r.get('action')}  coords: {r.get('coords')}")
        print(f"    latency:   {r.get('latency_s', 0):.2f}s")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nWrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
