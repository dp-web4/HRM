
# tests/test_api.py
import os, time, requests

BASE = os.environ.get("BASE", "http://localhost:8080")

def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    assert r.json().get("status") == "ok"

def test_read_seeded():
    r = requests.get(f"{BASE}/totality/read", params={"activation_min": 0.0}, timeout=5)
    r.raise_for_status()
    d = r.json()
    assert len(d.get("schemes", [])) >= 1
    assert len(d.get("canvases", [])) >= 1

def test_imagine_and_write():
    r = requests.get(f"{BASE}/totality/read", params={"activation_min": 0.0}, timeout=5)
    d = r.json()
    sid = d["schemes"][0]["id"]
    r2 = requests.post(f"{BASE}/totality/imagine", json={
        "seed_scheme_ids": [sid],
        "ops": [{"op": "value_perm"}],
        "count": 2
    }, timeout=5)
    r2.raise_for_status()
    created = r2.json().get("created", [])
    assert len(created) == 2

    r3 = requests.post(f"{BASE}/totality/write", json={
        "schemes": [{"label":"manipulate(item)","slots":[{"name":"item","value":"cube"}],"activation":0.5}],
        "mode": "merge"
    }, timeout=5)
    r3.raise_for_status()
    assert r3.json().get("status") == "ok"
