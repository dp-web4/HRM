"""
The being's tool-use agent loop (gateway-member, PRD_FLEET F2).

Model-agnostic: `generate(messages) -> {"content": str, "intents": [BeingIntent]}`
is supplied by the caller (the raising runner wraps its Ollama call + tool grammar).
Every intent the being emits is routed through a BeingGateClient — gated by the real
hestia law, dispatched by F1a — and its ResultEnvelope is re-injected before the
being speaks again.

This is what closes the Scenario-3 gap observed in the tool probe: the model could
select a tool and fill arguments, but after a tool result it fell back to narrating
a placeholder answer instead of *acting*. The loop keeps it in tool-space — result
in, decide again — until it produces a spoken turn with no further intents (or a
step cap forces a close). "Respond" becomes an act, not a text turn.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sage.gateway.being_gate_client import BeingGateClient, BeingIntent, ResultEnvelope

# generate(messages) -> {"content": str, "intents": list[BeingIntent]}
GenerateFn = Callable[[List[Dict[str, Any]]], Dict[str, Any]]


@dataclass
class ToolTurnResult:
    reply: str                                             # the being's final spoken words
    trace: List[Tuple[BeingIntent, ResultEnvelope]] = field(default_factory=list)
    steps: int = 0                                         # tool rounds taken
    capped: bool = False                                   # hit max_steps still wanting tools
    thinking: List[str] = field(default_factory=list)      # the model's think block per generate, if any
    salvaged: List[dict] = field(default_factory=list)     # calls lifted from the text channel: {step, effector, form}
    generates: List[dict] = field(default_factory=list)    # per generate, from Ollama's reply: {done_reason, prompt_eval_count, eval_count, retried}

    @property
    def acted(self) -> bool:
        return any(env.ok for _, env in self.trace)

    @property
    def refused(self) -> List[Tuple[BeingIntent, ResultEnvelope]]:
        return [(i, e) for i, e in self.trace if e.refused]


def run_tool_turn(client: BeingGateClient, generate: GenerateFn,
                  messages: List[Dict[str, Any]], max_steps: int = 3) -> ToolTurnResult:
    """Run one being turn that may reach for tools, gated end to end.

    Loop invariant: the being never sees a fabricated result — each tool message is a
    real ResultEnvelope (executed, refused, or honestly `pending` until F1a exists).
    """
    convo = list(messages)
    trace: List[Tuple[BeingIntent, ResultEnvelope]] = []

    for step in range(max_steps):
        out = generate(convo)
        content = out.get("content") or ""
        intents = out.get("intents") or []

        if not intents:                                    # a spoken turn — the being is done
            return ToolTurnResult(reply=content, trace=trace, steps=step)

        convo.append({"role": "assistant", "content": content, "intents": intents})
        for intent in intents:
            env = client.dispatch(intent)                  # gate + F1a dispatch + consume
            trace.append((intent, env))
            convo.append({"role": "tool", "effector": intent.effector,
                          "content": env.to_tool_message()})

    # Cap reached with tools still pending: force one final spoken close — we take its
    # words even if it wants more tools, so the being always ends its turn in language.
    out = generate(convo)
    return ToolTurnResult(reply=out.get("content") or "", trace=trace,
                          steps=max_steps, capped=True)


_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", re.S)


_NAME_KEYS = ("name", "tool", "action", "function", "tool_name")
_ARGS_KEYS = ("arguments", "parameters", "args", "input", "params")


def _json_calls(text: str, names) -> List[dict]:
    """`names`: the offered tool names (set) or {name: [param, ...]} (dict) for flat-form
    argument filtering."""
    known = names if isinstance(names, dict) else {n: [] for n in names}
    out, dec, i = [], json.JSONDecoder(), 0
    while True:
        starts = [k for k in (text.find("{", i), text.find("[", i)) if k >= 0]
        if not starts:
            return out
        j = min(starts)
        try:
            obj, end = dec.raw_decode(text, j)
        except ValueError:
            i = j + 1
            continue
        for o in (obj if isinstance(obj, list) else [obj]):
            if not isinstance(o, dict):
                continue
            # The name key varies by beat: {"name"}, {"tool"}, {"action"}, {"function"}
            # (Sprout beat 29, 2026-09-05: "action": "peer_ask" and a list of {"tool":
            # "memory_write", "path": ..., "content": ...} — 3 of 3 turns, 0 lifted).
            name = next((o[k] for k in _NAME_KEYS if isinstance(o.get(k), str)), None)
            args = next((o[k] for k in _ARGS_KEYS if isinstance(o.get(k), dict)), None)
            if name not in known and isinstance(args, dict):
                # {"name": "tool", "arguments": {"type": "recall", ...}}: the tool named
                # inside the arguments (Sprout beat 30, 2026-09-05)
                inner = next((args[k] for k in ("type", "tool", "name", "action") if isinstance(args.get(k), str)), None)
                if inner in known:
                    name = inner
                    args = {k: v for k, v in args.items() if k not in ("type", "tool", "name", "action")}
            if name not in known:
                continue
            if args is None:
                # flat form: the arguments sit beside the name key; keep only schema params
                # when the schema is known, so stray keys ("timestamp", "status") never
                # become arguments of a call.
                allowed = known.get(name) or []
                args = {k: v for k, v in o.items()
                        if k not in _NAME_KEYS and k not in _ARGS_KEYS and (not allowed or k in allowed)}
            out.append({"function": {"name": name, "arguments": dict(args)}, "_salvaged": "json"})
        i = max(end, j + 1)


def _python_calls(text: str, names: Dict[str, List[str]]) -> List[dict]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    consts: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)):
            consts[node.targets[0].id] = node.value.value
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
        if name not in names or len(node.args) > len(names[name]):
            continue
        args, ok = {}, True
        # positional arguments map onto the tool's parameters in schema order:
        # memory_write("journal.md", journal_entry) is (path, content) (beat 7, Sprout)
        for param, v in zip(names[name], node.args):
            if isinstance(v, ast.Constant):
                args[param] = v.value
            elif isinstance(v, ast.Name) and v.id in consts:
                args[param] = consts[v.id]
            else:
                ok = False
                break
        for kw in (node.keywords if ok else []):
            v = kw.value
            if kw.arg and isinstance(v, ast.Constant):
                args[kw.arg] = v.value
            elif kw.arg and isinstance(v, ast.Name) and v.id in consts:
                args[kw.arg] = consts[v.id]
            else:
                ok = False
                break
        if ok:
            out.append({"function": {"name": name, "arguments": args}, "_salvaged": "python"})
    return out


def salvage_tool_calls(content: str, tools: Iterable[dict]) -> List[dict]:
    """Lift well-formed tool calls that a model put in the TEXT channel, in Ollama's
    tool_calls shape (plus `_salvaged`: "json" | "python"). Accepted: a JSON object or
    array of {"name", "arguments"} (fenced or bare), or fenced Python `name(k="v", ...)`
    with literal or locally-assigned arguments, positional ones mapped in schema order.
    `tools` is what was offered this turn (Ollama tool specs); only those names count,
    so prose that mentions a tool is never a call.

    Measured 2026-09-05, same full-beat prompt: qwen2.5:1.5b emits bare JSON (Legion);
    qwen3.8-distill:2b emits fenced JSON in one beat and fenced Python in the next
    (Sprout, beats 5 to 7) while its think block says it decided to act. A salvaged call
    is gated like a native one and recorded as salvaged, so the record still shows which
    channel the being used."""
    params = {t["function"]["name"]: list((t["function"].get("parameters") or {}).get("properties") or {})
              for t in tools}
    if not content or not params:
        return []
    blocks = _FENCE.findall(content)
    found: List[dict] = []
    for text in (blocks or [content]):
        found.extend(_json_calls(text, params))
        found.extend(_python_calls(text, params))
    if blocks and not found:                    # fenced prose, bare call outside the fence
        found.extend(_json_calls(content, params))
    return found


def _think_budget(llm, floor: int = 6000) -> int:
    """The model config's think-on num_predict (variants[size].num_predict_think) when
    declared, else 6000, the value that recovered empty turns on Sprout and Legion
    (2026-09-03/04). With thinking on this is also the FIRST attempt's budget (OllamaIRP
    resolves it from the same config), so it is the retry's floor, not its value."""
    try:
        b = llm._adapter.capabilities.resolve_num_predict(llm.model_name, True, None)
        return max(int(b or 0), floor)
    except Exception:
        return floor


RETRY_MARGIN = 128   # tokens kept back from the window on a retry (template, tool-call framing)


def _retry_budget(llm, raw: Optional[dict] = None) -> int:
    """The once-retry budget for a length-stopped or truncated turn: everything the
    window has left after this prompt, never less than the think budget.

    Why not max(think_budget, max_response_tokens), which this was: with thinking on
    OllamaIRP already sends num_predict_think on the first attempt, and its
    resolve_num_predict ignores max_response_tokens whenever the variant declares a
    value. So the old retry re-sent the same 6000 and was a re-roll at temperature,
    not room to finish (Legion 18:33Z 2026-09-05: first explore turn eval 6000 empty,
    retry at 6000 stood by chance). num_ctx - prompt_eval_count is the most the model
    can be given; past that Ollama stops at the wall regardless (beat 46)."""
    floor = _think_budget(llm)
    try:
        num_ctx = int(getattr(llm, "num_ctx", None) or 0)
        prompt = int((raw or {}).get("prompt_eval_count") or 0)
    except (TypeError, ValueError):
        return floor
    if num_ctx <= 0 or prompt <= 0:
        return floor
    return max(floor, num_ctx - prompt - RETRY_MARGIN)


class _retry_room:
    """Set llm.num_predict_override for one retry, restore it after. Falls back to
    max_response_tokens for llm objects without the override (older adapters)."""
    def __init__(self, llm, budget: int):
        self.llm, self.budget = llm, budget

    def __enter__(self):
        llm = self.llm
        if hasattr(llm, "num_predict_override"):
            self.keep = ("override", llm.num_predict_override)
            llm.num_predict_override = self.budget
        else:
            self.keep = ("max", getattr(llm, "max_response_tokens", None))
            llm.max_response_tokens = max(self.budget, self.keep[1] or 0)
        return self.budget

    def __exit__(self, *exc):
        kind, val = self.keep
        if kind == "override":
            self.llm.num_predict_override = val
        elif val is not None:
            self.llm.max_response_tokens = val
        return False


def run_ollama_tool_turn(client: BeingGateClient, llm, seed_messages: List[Dict[str, Any]],
                         max_steps: int = 2, tools: Optional[List[dict]] = None) -> ToolTurnResult:
    """Run a gated tool turn using an OllamaIRP-like `llm` exposing
    get_chat_response(messages, tools=...) -> {"content", "tool_calls"}.

    Wraps the model + the bounded native-tool registry into the loop's generate()
    contract, so callers (the raising runner) need only supply the seed messages.
    `tools` narrows what is offered for this turn (default: the whole registry).
    """
    from sage.gateway.being_gate_client import ollama_tools, parse_tool_calls
    tools = tools if tools is not None else ollama_tools()
    # Keep the think block per generate: when a small model narrates instead of acting,
    # whether it decided not to call or failed to format the call is only visible here.
    thoughts: List[str] = []
    salvaged: List[dict] = []
    generates: List[dict] = []

    def generate(convo: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Flatten the loop's convo (carries extra keys) to chat messages. An assistant
        # turn that emitted intents MUST keep them as tool_calls: Qwen's chat template
        # raises on a tool message that follows an assistant message without tool_calls,
        # and Ollama surfaces that as HTTP 500 (measured: every post-tool turn 500'd).
        msgs = []
        for m in convo:
            out = {"role": m.get("role", "user"), "content": m.get("content", "")}
            if m.get("role") == "assistant" and m.get("intents"):
                out["tool_calls"] = [{"function": {"name": i.effector, "arguments": dict(i.args or {})}}
                                     for i in m["intents"]]
            msgs.append(out)
        retried = 0
        resp = llm.get_chat_response(msgs, tools=tools)
        content = resp.get("content", "") or ""
        calls = resp.get("tool_calls", []) or []
        if content.startswith("[OllamaIRP:") and not calls:
            # a transport failure is not the being's turn: retry once with a bigger budget,
            # then let the failure through as the visible reply rather than pretending.
            # The common 500 here is "invalid tool call arguments ... unexpected end of JSON
            # input": a long memory_write body cut off by num_predict (measured 2026-09-04).
            import sys as _sys
            print(f"[tool-loop] transport error, retrying once: {content[:200]}", file=_sys.stderr)
            # no raw reply here, so no prompt_eval_count: the retry gets the think budget
            # (for a no-think model that is still more than its variant num_predict)
            with _retry_room(llm, _retry_budget(llm, None)):
                resp = llm.get_chat_response(msgs, tools=tools)
                retried += 1
            content = resp.get("content", "") or ""
            calls = resp.get("tool_calls", []) or []
        if not content and not calls:
            # An empty turn is a harness signal, not a being's choice: say why on stderr
            # (budget exhausted in deliberation, adapter stripped everything, ...).
            import sys as _sys
            raw = resp.get("raw") or {}
            msg = raw.get("message") or {}
            print(f"[tool-loop] EMPTY turn: done_reason={raw.get('done_reason')} "
                  f"prompt_eval={raw.get('prompt_eval_count')} eval={raw.get('eval_count')} "
                  f"raw_content={str(msg.get('content', ''))[:200]!r} "
                  f"thinking={str(msg.get('thinking', ''))[:200]!r}", file=_sys.stderr)
            # Qwen3.8 (heretic) sometimes re-opens a think block even with think=false and
            # spends the whole budget there (measured 5/10 turns, 2026-09-03). Give it room
            # ONCE to finish and act, rather than recording silence as the being's choice.
            # Room = what the window has left after this prompt (_retry_budget), sent as an
            # override so the config's first-attempt budget cannot silently re-apply.
            if raw.get("done_reason") == "length" and (hasattr(llm, "max_response_tokens")
                                                       or hasattr(llm, "num_predict_override")):
                with _retry_room(llm, _retry_budget(llm, raw)) as budget:
                    print(f"[tool-loop] retrying once with num_predict={budget} "
                          f"(num_ctx={getattr(llm, 'num_ctx', None)} prompt_eval={raw.get('prompt_eval_count')})",
                          file=_sys.stderr)
                    resp = llm.get_chat_response(msgs, tools=tools)
                    retried += 1
                    content = resp.get("content", "") or ""
                    calls = resp.get("tool_calls", []) or []
        thoughts.append(str(((resp.get("raw") or {}).get("message") or {}).get("thinking") or ""))
        # What the window did this generate, from the reply that stood (after any retry):
        # prompt_eval_count + eval_count == num_ctx with done_reason "length" is the wall
        # beat 46 hit; it was only on stderr then and had to be reconstructed by hand.
        raw = resp.get("raw") or {}
        generates.append({"done_reason": raw.get("done_reason"), "prompt_eval_count": raw.get("prompt_eval_count"),
                          "eval_count": raw.get("eval_count"), "retried": retried})
        if not calls and content:
            # the call in the wrong channel: lift it, gate it as normal, record that it was lifted
            calls = salvage_tool_calls(content, tools)
            salvaged.extend({"step": len(thoughts) - 1, "effector": c["function"]["name"],
                             "form": c["_salvaged"]} for c in calls)
        return {"content": content, "intents": parse_tool_calls(calls)}

    result = run_tool_turn(client, generate, seed_messages, max_steps=max_steps)
    result.thinking = thoughts
    result.salvaged = salvaged
    result.generates = generates
    return result
