
# tests/run_tests.py
import os, sys, json, time, requests

BASE = os.environ.get("BASE", "http://localhost:8080")

def pretty(j):
    print(json.dumps(j, indent=2))

def main():
    print("== Health ==")
    r = requests.get(f"{BASE}/health", timeout=5); r.raise_for_status(); pretty(r.json())

    print("\n== Read ==")
    r = requests.get(f"{BASE}/totality/read", params={"activation_min":0.0}, timeout=5); r.raise_for_status()
    d = r.json(); pretty({"schemes":len(d.get("schemes",[])), "canvases":len(d.get("canvases",[]))})
    sid = d["schemes"][0]["id"]

    print("\n== Imagine 2 ==")
    r = requests.post(f"{BASE}/totality/imagine", json={
        "seed_scheme_ids":[sid],
        "ops":[{"op":"value_perm"}],
        "count":2
    }, timeout=5); r.raise_for_status(); pretty(r.json())

    print("\n== Activate +0.2 ==")
    r = requests.post(f"{BASE}/totality/activate", json={
        "targets":[{"id":sid,"delta":0.2}]
    }, timeout=5); r.raise_for_status(); pretty(r.json())

    print("\n== Write abstraction ==")
    r = requests.post(f"{BASE}/totality/write", json={
        "schemes":[{"label":"manipulate(item)","slots":[{"name":"item","value":"cube"}],"activation":0.6}],
        "mode":"merge"
    }, timeout=5); r.raise_for_status(); pretty(r.json())

    print("\n== Snapshot ==")
    r = requests.get(f"{BASE}/totality/snapshot", timeout=5); r.raise_for_status(); pretty({"snapshot_id":r.json().get("snapshot_id")})

if __name__ == "__main__":
    main()
