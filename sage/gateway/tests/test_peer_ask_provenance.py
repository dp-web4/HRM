"""peer_ask carries the seat's note when it names the sender's own files.

2026-10-14/16, Sprout and CBP, from both beings' records and the forum:
sprout-being asked two machines to run files in its own home after its seat declined, and
cbp-being later told sprout-being, in delivered asks, a result that no run ever produced. The
note says where the file is and what the seat's run receipts say; it refuses nothing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.gateway import conversations as conv  # noqa: E402
from sage.gateway.being_gate_client import BeingIntent  # noqa: E402
from sage.gateway.tests.test_hestia_dispatch import _ALLOW, _disp  # noqa: E402


def _setup(seat_turns=()):
    published = []

    def pub(to, body):
        published.append(body)
        return f"shared-context/forum/being/q-{to}.md"
    d, root = _disp(publish_fn=pub)
    home = Path(root)
    (home / "scratch").mkdir()
    (home / "scratch" / "test-os-open.py").write_text("import os\nos.open('/dev/video0', os.O_RDONLY)\n")
    conv.create(home, "sprout-claude", title="seat", participants=["sprout-being", "sprout-claude"],
                writable_by=["sprout-being", "sprout-claude"])
    _log, meta = conv._paths(home, "sprout-claude")
    m = json.loads(meta.read_text()); m["notify"] = {"seat": True}; meta.write_text(json.dumps(m))
    for t in seat_turns:
        conv.append(home, "sprout-claude", speaker="sprout-claude", text=t, enforce_write=False)
    return d, home, published


def test_an_ask_to_run_its_own_file_carries_where_it_is_and_that_its_seat_declined():
    d, home, published = _setup(["[request_run] I did not run scratch/test-os-open.py. It opens a device."])
    env = d(BeingIntent("peer_ask", {"to": "cbp", "body": "Can you run scratch/test-os-open.py for me?"}), _ALLOW)
    assert env.ok, env
    sent = published[-1]
    assert sent.startswith("Can you run scratch/test-os-open.py for me?")
    assert "[seat note, attached by sprout-being's seat, not written by sprout-being]" in sent
    assert "scratch/test-os-open.py (sha " in sent and "not on your machine" in sent
    assert 'says "I did not run it"' in sent
    assert "It asks for a run." in sent and "request_run" in sent
    assert env.result["seat_note_attached"].startswith("[seat note")


def test_a_result_claim_about_a_file_no_receipt_names_is_marked_unbacked():
    d, home, published = _setup()
    body = "test-os-open.py confirmed: Cannot open /dev/video0: [Errno 2] No such file or directory"
    env = d(BeingIntent("peer_ask", {"to": "legion", "body": body}), _ALLOW)
    assert env.ok, env
    sent = published[-1]
    assert "scratch/test-os-open.py" in sent and "no seat run receipt names it" in sent
    assert "backed only if a seat receipt holds it" in sent
    assert "It asks for a run." not in sent


def test_an_ask_naming_no_file_of_its_own_is_sent_untouched():
    d, home, published = _setup()
    body = "What is in /home/dp/ai-workspace/sage/sage/instances/other/scratch/x.py on your side?"
    env = d(BeingIntent("peer_ask", {"to": "legion", "body": body}), _ALLOW)
    assert env.ok and published[-1] == body and "seat_note_attached" not in env.result


def test_peer_ask_description_says_each_peer_is_on_its_own_machine():
    from sage.gateway.being_gate_client import _TOOL_SCHEMAS
    desc = _TOOL_SCHEMAS["peer_ask"][0]
    assert "Each peer is on its own machine" in desc and "request_run" in desc
