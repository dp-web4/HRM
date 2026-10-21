"""Independent boundary probes for PR 484; synthetic files and conversations only."""
import hashlib
import os
from unittest.mock import patch

from sage.gateway.hestia_dispatch import HestiaF1aDispatcher


def dispatcher(home):
    d = object.__new__(HestiaF1aDispatcher)
    d.memory_root = str(home)
    d.member = "sender-being"
    return d


def test_bare_symlink_does_not_publish_outside_file_digest(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"synthetic outside content")
    (home / "report.txt").symlink_to(outside)
    d = dispatcher(home)
    with patch.object(d, "_seat_run_receipt", return_value="no receipt"), \
            patch("socket.gethostname", return_value="synthetic-host"):
        note = d._ask_provenance("Please inspect report.txt")
    assert hashlib.sha256(outside.read_bytes()).hexdigest()[:12] not in note


def test_bare_fifo_is_not_a_hashable_candidate(tmp_path):
    os.mkfifo(tmp_path / "pending.txt")
    # Do not open it: opening this accepted candidate would wait for a writer.
    assert dispatcher(tmp_path)._ask_files("Please inspect pending.txt") == []


def test_peer_receipt_is_not_attributed_to_seat(tmp_path):
    d = dispatcher(tmp_path)
    metadata = [{"id": "peer-thread", "participants": ["sender-being", "peer-being"],
                 "notify": {"peer-being": "peer-route"}}]
    turns = [{"seq": 1, "from": "peer-being", "text": "[request_run] I ran scratch/check.py."}]
    with patch("sage.gateway.conversations.listing", return_value=metadata), \
            patch("sage.gateway.conversations.recent", return_value=turns):
        receipt = d._seat_run_receipt("scratch/check.py")
    assert "the seat's latest receipt" not in receipt


def test_plain_regular_file_is_still_a_candidate(tmp_path):
    (tmp_path / "report.txt").write_text("synthetic local content")
    assert dispatcher(tmp_path)._ask_files("Please inspect report.txt") == [
        ("report.txt", str(tmp_path / "report.txt"))]
