"""
One governed tool turn for a SAGE being, from the command line.

This is the harness entry the seat uses to hand a being a task and let it ACT under
hestia governance: the model runs on the local Ollama substrate, every intent it emits
is normalized and judged by the shared hestia gate law (fail-closed), and an allowed
intent is executed and witnessed by the F1a dispatcher against the running daemon.
The being never holds a tool; the seat never fabricates a result.

    python3 -m sage.gateway.governed_turn \
        --member legion-being --model qwen38-heretic:q3km \
        --instance sage/instances/legion-gemma3-12b \
        --task-file task.md [--system-file system.md] [--max-steps 2] [--out trace.json]

Output: one JSON document on stdout (and appended to <instance>/tool_turns.jsonl):
the reply, and for every intent the gate verdict, the result envelope and the witness
id the daemon returned. A refused act is a first-class outcome, not an error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


def _plain(obj):
    """Render a verdict/dataclass/whatever as plain JSON-able data."""
    import dataclasses
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    return str(obj)


def _read(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def build_client(member: str, instance: Path, model: str, workspace: str,
                 forum_dir: str | None, host_session_id: str, temperature: float,
                 max_tokens: int):
    from sage.gateway.being_gate_client import BeingGateClient
    from sage.gateway.hestia_dispatch import HestiaF1aDispatcher, make_forum_publisher
    from sage.irp.plugins.ollama_irp import OllamaIRP

    publish_fn = None
    if forum_dir and os.path.isdir(forum_dir):
        publish_fn = make_forum_publisher(forum_dir, member)
    dispatcher = HestiaF1aDispatcher(member, memory_root=str(instance),
                                     publish_fn=publish_fn,
                                     host_session_id=host_session_id)
    client = BeingGateClient(member_id=member,
                             identity_path=str(instance / "identity.json"),
                             workspace=workspace, dispatcher=dispatcher,
                             host_session_id=host_session_id)
    llm = OllamaIRP({"model_name": model, "temperature": temperature,
                     "max_response_tokens": max_tokens, "timeout_seconds": 600})
    return client, llm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--member", required=True, help="hestia member id, e.g. legion-being")
    ap.add_argument("--model", required=True, help="ollama model tag")
    ap.add_argument("--instance", required=True, help="the being's instance dir")
    ap.add_argument("--task-file", required=True, help="the user turn (the task)")
    ap.add_argument("--system-file", help="system turn; default is the gateway seed")
    ap.add_argument("--workspace", default=None, help="gate workspace root (default: repo root)")
    ap.add_argument("--forum-dir", default=os.path.expanduser("~/ai-workspace/shared-context/forum"))
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--out", help="also write the trace JSON here")
    args = ap.parse_args(argv)

    instance = Path(args.instance).resolve()
    if not (instance / "identity.json").exists():
        print(f"no identity.json under {instance}", file=sys.stderr)
        return 2
    workspace = args.workspace or str(Path(__file__).resolve().parents[2])
    host_session_id = f"governed-turn-{uuid.uuid4().hex[:12]}"

    client, llm = build_client(args.member, instance, args.model, workspace,
                               args.forum_dir, host_session_id, args.temperature,
                               args.max_tokens)

    system = _read(args.system_file) or (
        "You have a small set of real tools you may use through the hub: peer_ask, mesh, "
        "witness, memory_read, memory_write. Anything you do is governed by hestia and may "
        "be refused; a refusal is recorded, not hidden. Act when acting is the right "
        "response; otherwise say what you would do.")
    task = _read(args.task_file)
    seed = [{"role": "system", "content": system}, {"role": "user", "content": task}]

    from sage.gateway.being_tool_loop import run_ollama_tool_turn
    t0 = time.time()
    result = run_ollama_tool_turn(client, llm, seed, max_steps=args.max_steps)
    elapsed = round(time.time() - t0, 1)

    trace = []
    for intent, env in result.trace:
        trace.append({
            "effector": intent.effector,
            "args": dict(intent.args or {}),
            "ok": env.ok,
            "refused": env.refused,
            "verdict": _plain(getattr(env, "verdict", None)),
            "pending": getattr(env, "pending", None),
            "error": env.error,
            "witness_id": env.witness_id,
            "result": env.result if isinstance(env.result, (str, int, float, dict, list, type(None)))
            else str(env.result),
            "note": getattr(env, "note", None),
        })
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "member": args.member, "model": args.model, "instance": str(instance),
        "host_session_id": host_session_id, "elapsed_s": elapsed,
        "steps": result.steps, "capped": result.capped, "acted": result.acted,
        "reply": result.reply, "trace": trace,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(instance / "tool_turns.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
