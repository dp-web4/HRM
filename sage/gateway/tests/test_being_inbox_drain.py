"""Hermetic: the being's inbound hub drain (S4). Fake hub read, fake seat notify."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_inbox_drain import drain_once, map_kind, persist_notice  # noqa: E402


def _inst():
    return Path(tempfile.mkdtemp(prefix="inbox-"))


def test_kind_mapping_hub_roots_to_hestia_member_kinds():
    assert map_kind("review") == "review_request" and map_kind("review.request.pr") == "review_request"
    assert map_kind("forum.note") == "forum-note" and map_kind("reply") == "reply"
    assert map_kind("weird") == "coordination" and map_kind("") == "coordination"


def test_drain_persists_before_notifying_with_provenance_and_is_idempotent():
    inst = _inst()
    notes = [{"kind": "reply", "pointer_uri": "shared-context/forum/x.md", "from": "4a7f7eeb-legion-sage", "pair_id": "n-1"},
             {"kind": "review.request", "pointer_uri": "https://github.com/dp-web4/SAGE/pull/99", "from": "61525719-legion", "pair_id": "n-2"}]
    sent = []
    r = drain_once(inst, env_file="unused", fetch=lambda: notes, notify=lambda k, p: (sent.append((k, p)) or {"ok": True}))
    assert r["fetched"] == 2 and r["persisted"] == 2 and r["notified"] == 2 and r["errors"] == []
    files = sorted((inst / "notes" / "inbox").glob("*.md"))
    assert len(files) == 2
    body = files[0].read_text()
    assert "from: 4a7f7eeb-legion-sage" in body and "pointer: shared-context/forum/x.md" in body and "hub_notice_id: n-1" in body
    assert "relayed_by: sprout-claude" in body and "not by you and not by the sender" in body
    assert [k for k, _ in sent] == ["reply", "review_request"] and all(p.endswith(".md") for _, p in sent)
    # second pass with the same ids: nothing new, nothing re-notified
    sent.clear()
    r2 = drain_once(inst, env_file="unused", fetch=lambda: notes, notify=lambda k, p: (sent.append((k, p)) or {"ok": True}))
    assert r2["skipped"] == 2 and r2["persisted"] == 0 and sent == []
    assert (inst / "notes" / "inbox" / ".seen").read_text().split() == ["n-1", "n-2"]


def test_notify_failure_is_reported_not_raised_and_the_file_stays():
    inst = _inst()
    r = drain_once(inst, env_file="unused", fetch=lambda: [{"kind": "ack", "pointer_uri": "p", "from": "f", "pair_id": "n-9"}],
                   notify=lambda k, p: {"ok": False, "error": "hestia.member_notify_self"})
    assert r["persisted"] == 1 and r["notified"] == 0 and "member_notify_self" in r["errors"][0]
    assert list((inst / "notes" / "inbox").glob("*.md"))


def test_fetch_failure_is_reported_not_raised():
    def boom():
        raise RuntimeError("hub down")
    r = drain_once(_inst(), env_file="unused", fetch=boom, notify=lambda k, p: {"ok": True})
    assert r["fetched"] == 0 and r["errors"] and "hub down" in r["errors"][0]


def test_pointer_is_workspace_relative_when_inside_the_workspace():
    ws = Path(tempfile.mkdtemp(prefix="ws-")) / "sage"
    inst = ws / "sage" / "instances" / "x"
    inst.mkdir(parents=True)
    sent = []
    drain_once(inst, env_file="unused", fetch=lambda: [{"kind": "reply", "pointer_uri": "p", "from": "f", "pair_id": "n-3"}],
               notify=lambda k, p: (sent.append(p) or {"ok": True}), workspace=str(ws))
    assert sent and sent[0].startswith("sage/sage/instances/x/notes/inbox/")


def test_env_file_expands_shell_variables_and_tilde():
    from sage.gateway.being_inbox_drain import _env_from_file
    p = Path(tempfile.mkdtemp(prefix="env-")) / "hub.env"
    p.write_text('# comment\nHUB_URL="http://hub:8770"\nMY_LCT=abc   # trailing comment\nCHANNEL_CLIENT=$HOME/bin/cc\nMY_KEYPAIR=~/.web4/k.bin\n')
    e = _env_from_file(str(p))
    assert e["HUB_URL"] == "http://hub:8770" and e["MY_LCT"] == "abc"
    assert e["CHANNEL_CLIENT"] == os.path.expanduser("~/bin/cc") and e["MY_KEYPAIR"] == os.path.expanduser("~/.web4/k.bin")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
