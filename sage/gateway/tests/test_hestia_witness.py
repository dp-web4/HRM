"""Tests for the hestia witness helper — pure parsing + the fail-safe contract.
No live daemon needed (fail-safe path uses an unreachable endpoint). Runnable under
pytest or directly."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.hestia_witness import _parse, _unwrap, make_hestia_witness_fn  # noqa: E402


def test_parse_plain_json():
    assert _parse('{"a": 1}') == {"a": 1}


def test_parse_sse_picks_last_data_line():
    sse = 'event: message\ndata: {"jsonrpc":"2.0","id":1}\n\ndata: {"result":{"ok":true}}\n'
    assert _parse(sse) == {"result": {"ok": True}}


def test_parse_empty_and_garbage():
    assert _parse("") == {}
    assert _parse("not json") == {}


def test_unwrap_structured_content():
    assert _unwrap({"result": {"structuredContent": {"actionId": "x"}}}) == {"actionId": "x"}


def test_unwrap_text_content_fallback():
    resp = {"result": {"content": [{"type": "text", "text": '{"actionId": "y"}'}]}}
    assert _unwrap(resp) == {"actionId": "y"}


def test_unwrap_empty():
    assert _unwrap({}) == {}


def test_fail_safe_returns_none_when_daemon_unreachable():
    # port 1 is not listening -> connection refused -> the contract is None, not a raise
    fn = make_hestia_witness_fn("sprout-being", endpoint="http://127.0.0.1:1/mcp")
    assert fn("some event") is None


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
