"""Offline comparison of a recorded failed edit with a caller-chosen grounded edit.

All dispatch targets are disposable copies of a memory_read WINDOW, never the
being's files. This is a mechanism/oracle test, not model success, task completion,
full-file reconstruction, or approval to apply the suggested replacement.
No recorded text, path, replacement or error prose is emitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def observed_window(read):
    if read.get("effector") != "memory_read" or read.get("ok") is not True:
        raise ValueError("requires a successful memory_read")
    text = read.get("result")
    if not isinstance(text, str):
        raise ValueError("read result is not text")
    rows = text.splitlines(keepends=True)
    match = re.fullmatch(r"\[lines (\d+)-(\d+) of (\d+) in '.+'\]\n", rows[0] if rows else "")
    if not match:
        raise ValueError("requires a numbered, bounded read window")
    start, end, total = map(int, match.groups())
    if not 1 <= start <= end <= total:
        raise ValueError("invalid window bounds")
    size = end - start + 1
    body = rows[1:1 + size]
    suffix = "".join(rows[1 + size:])
    marker = f"\n[end of the range you asked for: lines {start}-{end} of {total}."
    if len(body) != size or not suffix.startswith(marker):
        raise ValueError("window is truncated or lacks its boundary marker")
    return "".join(body), start


def probe(read, edit, relative_line, new_line):
    """Replay only text editing, on three isolated copies. Never execute read content."""
    from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict
    from sage.gateway.reference_f1a import ReferenceF1aDispatcher

    text, start = observed_window(read)
    if edit.get("effector") != "memory_edit" or edit.get("ok") is not False:
        raise ValueError("requires a recorded failed memory_edit")
    args = edit.get("args") or {}
    if args.get("path") != (read.get("args") or {}).get("path"):
        raise ValueError("read and edit must name the same file")
    if not isinstance(args.get("old"), str) or not isinstance(args.get("new"), str):
        raise ValueError("requires canonical old/new text arguments")
    if any(k in args for k in ("start_line", "end_line", "delete_lines", "old_line")):
        raise ValueError("original proposal must be an exact-text edit, not a range")
    lines = text.splitlines(keepends=True)
    if not 1 <= relative_line <= len(lines) or "\n" in new_line or "\r" in new_line:
        raise ValueError("choose one observed line and a single-line replacement")
    old = lines[relative_line - 1]
    new = new_line + ("\n" if old.endswith("\n") else "")
    if new == old:
        raise ValueError("oracle replacement must change the selected line")
    checked = {"start_line": relative_line, "end_line": relative_line, "old": old, "new": new}
    changed = list(lines)
    changed[relative_line - 1] = new
    expected = "".join(changed)
    shifted = "[stale-read control: inserted line]\n" + text
    out = {}
    with tempfile.TemporaryDirectory(prefix="sage-edit-probe-") as root:
        dispatcher = ReferenceF1aDispatcher(memory_root=root)
        for label, initial, proposal, wanted in (
            ("recorded_proposal", text, {"old": args["old"], "new": args["new"]}, None),
            ("grounded_oracle", text, checked, expected),
            ("stale_read_control", shifted, checked, None),
        ):
            path = Path(root) / (label + ".md")
            path.write_text(initial, encoding="utf-8")
            # Never forward the recorded path or arbitrary recorded argument fields.
            result = dispatcher(BeingIntent("memory_edit", {**proposal, "path": path.name}),
                                GatewayVerdict("allow"))
            final = path.read_text(encoding="utf-8")
            out[label] = {"reported_ok": result.ok, "changed": final != initial,
                          "matches_oracle": final == wanted if wanted is not None else None,
                          "before_sha256": _sha(initial), "after_sha256": _sha(final)}
    return {"schema": "edit-grounding-probe/v1", "window_sha256": _sha(text),
            "recorded_edit_sha256": _sha(json.dumps({"old": args["old"], "new": args["new"]}, sort_keys=True)),
            "oracle_line_sha256": _sha(new),
            "observed_start_line": start, "selected_line": start + relative_line - 1,
            "arms": out, "limit": "Window-only, caller-chosen replacement; no model or task-success claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--beat", required=True, help="Exact recorded UTC timestamp")
    parser.add_argument("--phase", choices=("explore", "posture", "reflect", "answer", "act_after_answer"), required=True)
    parser.add_argument("--read-index", type=int, required=True)
    parser.add_argument("--edit-index", type=int, required=True)
    parser.add_argument("--relative-line", type=int, required=True)
    parser.add_argument("--new-line", required=True, help="Explicit oracle text; never generated or applied live")
    args = parser.parse_args(argv)
    if not 0 <= args.read_index < args.edit_index:
        parser.error("read index must precede edit index")
    try:
        matches = []
        with open(args.log, "rb") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("ts") == args.beat:
                    matches.append((row, hashlib.sha256(line).hexdigest()))
                    if len(matches) > 1:
                        raise ValueError("duplicate beat timestamp")
        if len(matches) != 1:
            raise ValueError("beat timestamp must identify exactly one record")
        trace = matches[0][0][args.phase]["trace"]
        result = probe(trace[args.read_index], trace[args.edit_index], args.relative_line, args.new_line)
        result["beat_record_sha256"] = matches[0][1]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        parser.error("input does not satisfy the recorded-window replay contract")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
