#!/usr/bin/env python3
"""consolidation — the being's self-account organ: replay -> distill -> write, with provenance.

WHY THIS FILE EXISTS (Legion seat, S2 of PRD_ONE_BEING_ONE_EXPERIENCE.md, 2026-09-05)
-------------------------------------------------------------------------------------
The being's record is three loops that never meet: raising sessions, heartbeats, and the
journal/todo/cartridge it writes by hand. Nothing replays them into one account. This is the
scheduled process that does, in the shape dev-sage's consolidation loop settled on
(cited by principle and commit `36a36172` only; nothing of its corpus crosses):

  1. REPLAY. `name_sources()` names every source under an instance dir EXACTLY (path,
     sha256, byte count, record count) — including the ones that are ABSENT. An absent
     journal is a reading (legion-being has no journal.md because it holds no write grant),
     not an error, and the graft says so.

  2. DISTILL. `distill()` is MECHANICAL: counts, histograms, spans, and a content-addressed
     table of the being's OWN words. No model is called and no summary is written. The
     seat's voice does not enter the being's account (the membot-cartridge lesson of 09-03:
     four seat memories came back as the being's own recall). The July own-account carry
     ("ask, do not offer"; dev-sage `1ee1479c`) applies at the organ: keep the being's text
     verbatim, name where it came from, drop nothing into a paraphrase.

  3. WRITE, WITH PROVENANCE. `consolidate()` is the cycle:
       - hash the named source set; if the latest graft for this member already names that
         hash AND was written by this same instrument, the cycle is a NO-OP and the no-op is
         LOGGED (a scheduled process that leaves no record of having run is indistinguishable
         from one that did not run). A changed instrument over unchanged sources is a new
         version (`reason: instrument_changed`): the graft is a function of sources AND
         instrument, so a fixed reading re-lands (Sprout's #42 review, finding 2);
       - otherwise distill and write `self-account-<member>.vN.json`, N monotonic,
         `supersedes` pointing at vN-1, older versions NEVER deleted or rewritten;
       - the graft names its training data structurally: every source's path, sha256,
         bytes and count, plus this instrument's own source hash.

GRAFTS SHIP DARK. Nothing live reads a graft; `read_graft()` returns None unless
SAGE_GRAFT_SELF_ACCOUNT=on. A consumer (the heartbeat digest, the raising prompt) opts in by
explicit switch, and the PRD's JOIN/ACCOUNT instruments say whether it changed anything.

READ ONCE. The bytes that are hashed into `training_data` are the bytes that are distilled
(`_read_sources()` returns both). A beat appended between naming and distilling cannot make
the graft name a sha the table does not reflect (#42 finding 1). Paths in the graft are
instance-relative (or cwd-/home-relative for the cartridge) so a graft is portable across
checkout roots (#42 finding 4). `error` counts NON-refusal errors only: a refusal carries an
error string too, and counting it twice made 135 refusals read as 136 errors (#42 finding 3).

PRE-REGISTERED PROPERTIES (checked by sage/memory/tests/test_consolidation.py):
  F1 idempotence: an unchanged source set under an unchanged instrument consolidates to a
     logged no-op, never a new version; a changed instrument yields vN+1 over the same sources.
  F2 determinism: the same instance consolidated into a fresh grafts dir yields a
     byte-identical table and training-data list.
  F3 ground truth: a toy instance with known contents distills to the known counts, and an
     absent source is named as absent.
  F4 hygiene: this file calls no model and carries no seat voice — no LLM client, no
     subprocess, no network, no summariser; the test greps this source to keep it so.
  F5 verbatim: every carried line is a byte-exact substring of the source it names.

This file instantiates nothing about a particular being; the instance dir is the profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_GRAFT = "self-account/v1"
SCHEMA_INDEX = "graft-index/v1"
VERSION = "consolidation/v1.1"
ENV_SWITCH = "SAGE_GRAFT_SELF_ACCOUNT"

# The being's own words leave the harness through these effectors (heartbeat.py trace).
OWN_WORD_EFFECTORS = ("memory_write", "remember", "witness")
_DATE_LINE = re.compile(r"^\s*(#+\s*)?(\*\*)?\d{4}-\d{2}-\d{2}")


# ----------------------------------------------------------------------------- hashing

def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_text(s: str) -> str:
    return _sha_bytes(s.encode("utf-8"))


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _self_sha() -> str:
    return _sha_file(Path(__file__))


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _portable(p: Path, instance: Path | None = None) -> str:
    """A path as the graft records it: instance-relative when under the instance, else relative
    to the cwd (the checkout) or to $HOME as `~/...`, else absolute. Machine-specific roots
    stay out of the graft so F2 survives a different checkout root."""
    p = Path(p)
    for root, prefix in ((instance, ""), (Path.cwd(), ""), (Path.home(), "~/")):
        if root is None:
            continue
        try:
            rel = p.resolve().relative_to(Path(root).resolve())
        except ValueError:
            continue
        return prefix + rel.as_posix() if str(rel) != "." else (prefix.rstrip("/") or ".")
    return str(p)


# ----------------------------------------------------------------------------- sources

def _jsonl(p: Path) -> list:
    return _jsonl_bytes(p.read_bytes())


def _journal_entries(text: str) -> list[str]:
    """Entries begin at a line that starts with a date (heartbeat.py's REFLECT asks for exactly
    that). Text before the first date line is one entry of its own, so nothing is dropped."""
    entries: list[str] = []
    cur: list[str] = []
    for line in text.splitlines(keepends=True):
        if _DATE_LINE.match(line) and cur:
            entries.append("".join(cur))
            cur = []
        cur.append(line)
    if cur:
        entries.append("".join(cur))
    return [e for e in entries if e.strip()]


def _jsonl_bytes(b: bytes) -> list:
    rows = []
    for line in b.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_unparsed": line})
    return rows


def _read_sources(instance: Path, cartridge_manifest: Path | None = None) -> tuple[list[dict], dict]:
    """Name every source exactly AND return the very bytes that were named, parsed, so
    distill() reads what training_data hashes and nothing later. Absent sources are listed
    with present=False, count 0. Order is fixed so the source-set hash is stable."""
    instance = Path(instance)
    out: list[dict] = []
    contents: dict[str, Any] = {"beats": [], "journal": "", "todo": "", "experience": [], "sessions": []}

    def one(name: str, p: Path, kind: str) -> bytes | None:
        if p.exists() and p.is_file():
            b = p.read_bytes()
            out.append({"source": name, "kind": kind, "path": _portable(p, instance), "present": True,
                        "sha256": _sha_bytes(b), "bytes": len(b), "count": 0})
            return b
        out.append({"source": name, "kind": kind, "path": _portable(p, instance), "present": False,
                    "sha256": None, "bytes": 0, "count": 0})
        return None

    b = one("beats", instance / "heartbeats.jsonl", "jsonl")
    if b is not None:
        contents["beats"] = _jsonl_bytes(b)
        out[-1]["count"] = len(contents["beats"])
    b = one("journal", instance / "journal.md", "markdown")
    if b is not None:
        contents["journal"] = b.decode("utf-8", errors="replace")
        out[-1]["count"] = len(_journal_entries(contents["journal"]))
    b = one("todo", instance / "todo.md", "markdown")
    if b is not None:
        contents["todo"] = b.decode("utf-8", errors="replace")
        out[-1]["count"] = len([l for l in contents["todo"].splitlines() if l.strip()])
    b = one("account", instance / "account.json", "json")
    if b is not None:
        out[-1]["count"] = 1
    b = one("experience", instance / "experience_buffer.json", "json")
    if b is not None:
        try:
            buf = json.loads(b.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            buf = []
        contents["experience"] = buf if isinstance(buf, list) else []
        out[-1]["count"] = len(buf) if isinstance(buf, (list, dict)) else 0

    sess_dir = instance / "sessions"
    files = sorted(p for p in sess_dir.glob("session_*.json")) if sess_dir.is_dir() else []
    if files:
        h = hashlib.sha256()
        total = 0
        for p in files:
            fb = p.read_bytes()
            h.update(p.name.encode()); h.update(_sha_bytes(fb).encode())
            total += len(fb)
            try:
                d = json.loads(fb.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(d, dict):
                contents["sessions"].append((p, d))
        out.append({"source": "sessions", "kind": "json-dir", "path": _portable(sess_dir, instance), "present": True,
                    "sha256": h.hexdigest(), "bytes": total, "count": len(files)})
    else:
        out.append({"source": "sessions", "kind": "json-dir", "path": _portable(sess_dir, instance), "present": False,
                    "sha256": None, "bytes": 0, "count": 0})

    # The long-term memory cartridge is named by its manifest (count, fingerprint) and the
    # sha of its store. Its text is not read here: it needs numpy and the being's words that
    # went into it already appear verbatim in the beats' `remember` args.
    if cartridge_manifest is not None:
        cm = Path(cartridge_manifest)
        if cm.exists():
            try:
                m = json.loads(cm.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                m = {}
            store = cm.with_name(cm.name.replace("_manifest.json", ".npz"))
            out.append({"source": "cartridge", "kind": "membot-cartridge", "path": _portable(cm), "present": True,
                        "sha256": _sha_file(cm), "bytes": cm.stat().st_size,
                        "count": int(m.get("count", 0) or 0),
                        "fingerprint": m.get("fingerprint"), "manifest_timestamp": m.get("timestamp"),
                        "store_path": _portable(store) if store.exists() else None,
                        "store_sha256": _sha_file(store) if store.exists() else None})
        else:
            out.append({"source": "cartridge", "kind": "membot-cartridge", "path": _portable(cm), "present": False,
                        "sha256": None, "bytes": 0, "count": 0})
    return out, contents


def name_sources(instance: Path, cartridge_manifest: Path | None = None) -> list[dict]:
    """Name every source exactly (the training_data list). A view over _read_sources()."""
    return _read_sources(instance, cartridge_manifest)[0]


def source_set_sha(sources: list[dict]) -> str:
    return _sha_text(_canon([(s["source"], s["sha256"], s["count"]) for s in sources]))


# ----------------------------------------------------------------------------- distill

def _own_line(text: str, source: str, ref: str, ts: str | None) -> dict:
    return {"sha256": _sha_text(text), "source": source, "ref": ref, "ts": ts, "chars": len(text)}


TUTOR_SPEAKERS = frozenset({"Claude", "Human", "User", "Teacher", "dp", "Dennis", "system", "System"})


def _being_speaker(sessions: list[tuple[Path, dict]]) -> tuple[str | None, dict[str, int]]:
    """The being is the speaker in the transcripts that is not the tutor. identity.json's name is
    not it (legion-gemma3-12b: identity says "legion", every session says "SAGE"; measured 09-05)."""
    speakers: dict[str, int] = {}
    for _p, d in sessions:
        for turn in (d.get("conversation") or []) if isinstance(d, dict) else []:
            if isinstance(turn, dict) and turn.get("speaker"):
                sp = str(turn["speaker"]); speakers[sp] = speakers.get(sp, 0) + 1
    cands = sorted(((n, sp) for sp, n in speakers.items() if sp not in TUTOR_SPEAKERS), reverse=True)
    return (cands[0][1] if cands else None), dict(sorted(speakers.items()))


def _turn_trace(turn: dict | None) -> list[dict]:
    if not isinstance(turn, dict):
        return []
    tr = turn.get("trace") or []
    return [e for e in tr if isinstance(e, dict)]


def distill(instance: Path, sources: list[dict], carry: int = 12, contents: dict | None = None) -> dict:
    """Mechanical. Counts, histograms, spans, and the content-addressed table of the being's
    own words. Every text kept verbatim (`carry`) names the source it is a substring of.
    `contents` is what _read_sources() parsed from the bytes it hashed; without it the sources
    are re-read here, which is the name-then-reread race this argument exists to close."""
    instance = Path(instance)
    if contents is None:
        sources, contents = _read_sources(instance)
    by = {s["source"]: s for s in sources}
    table: dict[str, Any] = {"schema": SCHEMA_GRAFT}
    own: list[dict] = []          # every own-word line: sha + pointer, never the text
    carried: list[dict] = []      # the last `carry` lines from beats/journal, verbatim

    # --- beats -------------------------------------------------------------------------
    beats = contents["beats"] if by["beats"]["present"] else []
    eff: dict[str, dict[str, int]] = {}
    rules: dict[str, int] = {}
    models: dict[str, int] = {}
    empty = act_first = salvaged_beats = salvaged_calls = thinking_beats = 0
    ts_list: list[str] = []
    # S1/S3 instruments as the record carries them (heartbeat.py on main after 0d87bb577):
    # `interventions` [{kind, suppressed, ...}], `account` {present, sha256}, `join` {session, presence}, `wake` {by}
    iv_kinds: dict[str, int] = {}
    iv_suppressed: dict[str, int] = {}
    account_present = 0
    account_shas: set[str] = set()
    join_session = join_presence = 0
    wake_by: dict[str, int] = {}
    beats_with_ledger = 0
    for i, r in enumerate(beats):
        if "_unparsed" in r:
            continue
        ts = r.get("ts")
        if ts:
            ts_list.append(ts)
        models[str(r.get("model"))] = models.get(str(r.get("model")), 0) + 1
        if r.get("act_first"):
            act_first += 1
        turns = [r.get(k) for k in ("explore", "posture", "reflect")]
        any_trace = any(_turn_trace(t) for t in turns)
        any_reply = any(isinstance(t, dict) and (t.get("reply") or "").strip() for t in turns)
        if not any_trace and not any_reply:
            empty += 1
        s_calls = sum(len(t.get("salvaged") or []) for t in turns if isinstance(t, dict))
        if s_calls:
            salvaged_beats += 1; salvaged_calls += s_calls
        if any(isinstance(t, dict) and t.get("thinking") for t in turns):
            thinking_beats += 1
        if isinstance(r.get("interventions"), list):
            beats_with_ledger += 1
            for iv in r["interventions"]:
                if isinstance(iv, dict):
                    k = str(iv.get("kind")); iv_kinds[k] = iv_kinds.get(k, 0) + 1
                    sp = str(iv.get("suppressed")); iv_suppressed[sp] = iv_suppressed.get(sp, 0) + 1
        acc = r.get("account")
        if isinstance(acc, dict) and acc.get("present"):
            account_present += 1
            if acc.get("sha256"):
                account_shas.add(str(acc["sha256"]))
        jn = r.get("join")
        if isinstance(jn, dict):
            if jn.get("session"):
                join_session += 1
            if jn.get("presence"):
                join_presence += 1
        wk = r.get("wake")
        if isinstance(wk, dict) and wk.get("by"):
            wake_by[str(wk["by"])] = wake_by.get(str(wk["by"]), 0) + 1
        ref = str(r.get("host_session_id") or f"beat:{i}")
        for tname, t in zip(("explore", "posture", "reflect"), turns):
            if not isinstance(t, dict):
                continue
            reply = (t.get("reply") or "")
            if reply.strip():
                own.append(_own_line(reply, "beats", f"{ref}/{tname}.reply", ts))
                if tname == "reflect":
                    carried.append({"text": reply, "source": "beats", "ref": f"{ref}/{tname}.reply", "ts": ts})
            for e in _turn_trace(t):
                name = str(e.get("effector"))
                d = eff.setdefault(name, {"trials": 0, "ok": 0, "refused": 0, "pending": 0, "error": 0})
                d["trials"] += 1
                if e.get("ok"):
                    d["ok"] += 1
                if e.get("refused"):
                    d["refused"] += 1
                    rule = str(e.get("rule") or "unstated")
                    rules[rule] = rules.get(rule, 0) + 1
                if e.get("pending"):
                    d["pending"] += 1
                if e.get("error") and not e.get("refused"):   # a refusal carries an error string too
                    d["error"] += 1
                if name in OWN_WORD_EFFECTORS:
                    args = e.get("args") or {}
                    content = args.get("content") or args.get("text") or args.get("note")
                    if isinstance(content, str) and content.strip():
                        own.append(_own_line(content, "beats", f"{ref}/{tname}.{name}", ts))
                        if name == "memory_write" and str(args.get("path", "")).endswith("journal.md"):
                            carried.append({"text": content, "source": "beats", "ref": f"{ref}/{tname}.{name}", "ts": ts})
    table["beats"] = {
        "n": len(beats), "empty": empty, "act_first": act_first,
        "span": [min(ts_list), max(ts_list)] if ts_list else None,
        "models": dict(sorted(models.items())),
        "effectors": dict(sorted(eff.items())),
        "refusal_rules": dict(sorted(rules.items())),
        # the guards ledger: what the record carries per turn (salvaged, thinking) and, on
        # records written after S3, the explicit `interventions` list with what each suppressed
        "interventions": {"salvaged_beats": salvaged_beats, "salvaged_calls": salvaged_calls,
                          "thinking_beats": thinking_beats, "beats_with_ledger": beats_with_ledger,
                          "kinds": dict(sorted(iv_kinds.items())),
                          "suppressed": dict(sorted(iv_suppressed.items()))},
        # S1/S3 instruments, beat side, as counts over this record (the PRD's JOIN and ACCOUNT)
        "join": {"session_attributed": join_session, "presence_attributed": join_presence},
        "account": {"present": account_present, "distinct_sha256": len(account_shas)},
        "wake_by": dict(sorted(wake_by.items())),
    }

    # --- journal / todo -------------------------------------------------------------------
    entries = _journal_entries(contents["journal"]) if by["journal"]["present"] else []
    for k, e in enumerate(entries):
        own.append(_own_line(e, "journal", f"entry:{k}", None))
    for k, e in enumerate(entries[-carry:]):
        carried.append({"text": e, "source": "journal", "ref": f"entry:{len(entries) - len(entries[-carry:]) + k}", "ts": None})
    table["journal"] = {"present": by["journal"]["present"], "entries": len(entries)}
    table["todo"] = {"present": by["todo"]["present"], "lines": by["todo"]["count"]}

    # --- sessions ----------------------------------------------------------------------
    sessions = contents["sessions"] if by["sessions"]["present"] else []
    being, speakers = _being_speaker(sessions)
    phases: dict[str, int] = {}
    smodels: dict[str, int] = {}
    being_turns = 0
    starts: list[str] = []
    for p, d in sessions:
        phases[str(d.get("phase"))] = phases.get(str(d.get("phase")), 0) + 1
        smodels[str(d.get("model"))] = smodels.get(str(d.get("model")), 0) + 1
        if d.get("start"):
            starts.append(str(d["start"]))
        n = str(d.get("session") or p.stem.split("_")[-1])
        for j, turn in enumerate(d.get("conversation") or []):
            if isinstance(turn, dict) and turn.get("speaker") == being and isinstance(turn.get("text"), str) and turn["text"].strip():
                being_turns += 1
                own.append(_own_line(turn["text"], "sessions", f"session:{n}/turn:{j}", d.get("start")))
    table["sessions"] = {"n": by["sessions"]["count"], "being": being, "speakers": speakers, "being_turns": being_turns,
                         # what the derivation left out, so a reader sees the assumption, not the code
                         "tutor_speakers": sorted(TUTOR_SPEAKERS),
                         "span": [min(starts), max(starts)] if starts else None,
                         "phases": dict(sorted(phases.items())), "models": dict(sorted(smodels.items()))}

    # --- experience buffer -------------------------------------------------------------
    exp_n = 0
    prompts: dict[str, int] = {}
    if by["experience"]["present"]:
        buf = contents["experience"]
        if isinstance(buf, list):
            for x in buf:
                if not isinstance(x, dict):
                    continue
                exp_n += 1
                pr = str(x.get("prompt") or "")[:80]
                prompts[pr] = prompts.get(pr, 0) + 1
                resp = x.get("response")
                if isinstance(resp, str) and resp.strip():
                    own.append(_own_line(resp, "experience", f"id:{x.get('id')}", x.get("timestamp") or x.get("ts")))
    top = sorted(prompts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    table["experience"] = {"n": exp_n, "top_prompts": [{"prompt": p, "n": c} for p, c in top]}

    # --- cartridge (named only) ----------------------------------------------------------
    if "cartridge" in by:
        c = by["cartridge"]
        table["cartridge"] = {"present": c["present"], "count": c["count"], "fingerprint": c.get("fingerprint")}

    # --- the being's own words ---------------------------------------------------------------
    per_source: dict[str, int] = {}
    for line in own:
        per_source[line["source"]] = per_source.get(line["source"], 0) + 1
    table["own_account"] = {
        "lines": len(own), "distinct": len({l["sha256"] for l in own}),
        "chars": sum(l["chars"] for l in own),
        "per_source": dict(sorted(per_source.items())),
        "carry": carried[-carry:],          # verbatim, the most recent; ask, do not offer
    }
    # sha + pointer for every own-word line; the text stays at the source. Written as a sidecar
    # by consolidate() and named there by path + sha256 + count, like the training data.
    table["_own_index"] = own
    return table


# ----------------------------------------------------------------------------- write

def _log(grafts_dir: Path, event: dict) -> None:
    grafts_dir.mkdir(parents=True, exist_ok=True)
    with open(grafts_dir / "consolidation_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), **event}, ensure_ascii=False) + "\n")


def _load_index(grafts_dir: Path) -> dict:
    p = grafts_dir / "index.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"schema": SCHEMA_INDEX, "members": {}}


def _latest_instrument_sha(grafts_dir: Path, fname: str) -> str | None:
    """Index entries written by consolidation/v1 carry no instrument sha; the graft file does."""
    try:
        return json.loads((grafts_dir / fname).read_text(encoding="utf-8")).get("instrument", {}).get("sha256")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _member_id(instance: Path, member: str | None) -> str:
    if member:
        return member
    hb = instance / "heartbeats.jsonl"
    if hb.exists():
        for r in reversed(_jsonl(hb)):
            if isinstance(r, dict) and r.get("member"):
                return str(r["member"])
    return instance.name


def consolidate(instance: Path, grafts_dir: Path | None = None, *, seat: str = "unknown",
                member: str | None = None, cartridge_manifest: Path | None = None,
                carry: int = 12) -> dict:
    """One scheduled cycle for one instance. Idempotent (F1): an unchanged source set is a
    LOGGED no-op, not a new version. Never overwrites, never deletes."""
    instance = Path(instance)
    grafts_dir = Path(grafts_dir) if grafts_dir else instance / "grafts"
    mid = _member_id(instance, member)
    sources, contents = _read_sources(instance, cartridge_manifest)
    sset = source_set_sha(sources)
    self_sha = _self_sha()
    index = _load_index(grafts_dir)
    entry = index["members"].setdefault(mid, {"latest": None, "versions": []})
    reason = "first"
    if entry["versions"]:
        last = entry["versions"][-1]
        last_instr = last.get("instrument_sha256") or _latest_instrument_sha(grafts_dir, last["file"])
        if last["source_set_sha256"] == sset and last_instr == self_sha:
            ev = {"event": "noop", "member": mid, "source_set_sha256": sset,
                  "latest": entry["latest"], "seat": seat, "instrument_sha256": self_sha}
            _log(grafts_dir, ev)
            return ev
        reason = "sources_changed" if last["source_set_sha256"] != sset else "instrument_changed"
    table = distill(instance, sources, carry=carry, contents=contents)
    version = len(entry["versions"]) + 1
    supersedes = entry["versions"][-1]["file"] if entry["versions"] else None
    fname = f"self-account-{mid}.v{version}.json"
    own_index = table.pop("_own_index")
    iname = f"self-account-{mid}.v{version}.own-index.jsonl"
    ibytes = ("".join(_canon(l) + "\n" for l in own_index)).encode("utf-8")
    table["own_account"]["index"] = {"file": iname, "sha256": _sha_bytes(ibytes), "lines": len(own_index)}
    graft = {
        "schema": SCHEMA_GRAFT, "instrument": {"version": VERSION, "file": os.path.basename(__file__),
                                              "sha256": self_sha},
        "member": mid, "instance": _portable(instance), "version": version, "supersedes": supersedes,
        "reason": reason, "written_at": _now(), "seat": seat,
        "source_set_sha256": sset, "training_data": sources,
        "table": table,
    }
    grafts_dir.mkdir(parents=True, exist_ok=True)
    target = grafts_dir / fname
    if target.exists() or (grafts_dir / iname).exists():  # never overwrite: a version is written once
        raise FileExistsError(str(target))
    (grafts_dir / iname).write_bytes(ibytes)
    target.write_text(json.dumps(graft, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    entry["versions"].append({"version": version, "file": fname, "source_set_sha256": sset,
                              "instrument_sha256": self_sha, "reason": reason,
                              "written_at": graft["written_at"], "supersedes": supersedes})
    entry["latest"] = fname
    (grafts_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    ev = {"event": "graft", "member": mid, "version": version, "file": fname, "supersedes": supersedes,
          "reason": reason, "source_set_sha256": sset, "seat": seat, "instrument_sha256": graft["instrument"]["sha256"],
          "own_lines": table["own_account"]["lines"], "beats": table["beats"]["n"],
          "sessions": table["sessions"]["n"], "journal_entries": table["journal"]["entries"]}
    _log(grafts_dir, ev)
    return ev


def read_graft(instance: Path, grafts_dir: Path | None = None, member: str | None = None) -> dict | None:
    """Dark by default. Returns the latest graft only when SAGE_GRAFT_SELF_ACCOUNT=on."""
    if os.environ.get(ENV_SWITCH, "").lower() != "on":
        return None
    instance = Path(instance)
    grafts_dir = Path(grafts_dir) if grafts_dir else instance / "grafts"
    mid = _member_id(instance, member)
    entry = _load_index(grafts_dir)["members"].get(mid)
    if not entry or not entry.get("latest"):
        return None
    return json.loads((grafts_dir / entry["latest"]).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="consolidate a being's record into a versioned self-account graft (dark)")
    ap.add_argument("--instance", required=True, help="sage/instances/<dir>")
    ap.add_argument("--grafts", default=None, help="grafts dir (default <instance>/grafts)")
    ap.add_argument("--seat", default=os.environ.get("SAGE_SEAT", "unknown"))
    ap.add_argument("--member", default=None, help="member id (default: from the last beat, else the dir name)")
    ap.add_argument("--cartridge-manifest", default=None, help="membot <mount>.cart_manifest.json to name as a source")
    ap.add_argument("--carry", type=int, default=12, help="how many recent own-word lines to carry verbatim")
    a = ap.parse_args(argv)
    ev = consolidate(Path(a.instance), Path(a.grafts) if a.grafts else None, seat=a.seat, member=a.member,
                     cartridge_manifest=Path(a.cartridge_manifest) if a.cartridge_manifest else None,
                     carry=a.carry)
    print(json.dumps(ev, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
