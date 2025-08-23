#!/usr/bin/env python3
import sys, json
from statistics import mean

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def main(p):
    E, steps = [], []
    for rec in load_jsonl(p):
        e = rec.get('E')
        if isinstance(e, (int, float)):
            E.append(float(e))
        s = rec.get('step_idx')
        if isinstance(s, int):
            steps.append(s)
    if not E:
        print('no energy values found'); return
    dE = [E[i]-E[i-1] for i in range(1, len(E))]
    neg = sum(1 for v in dE if v < 0)
    mono_ratio = (neg / len(dE)) if dE else float('nan')
    print(f'file: {p}')
    print(f'num_steps: {len(E)}  minE: {min(E):.6f}  finalE: {E[-1]:.6f}  meanE: {mean(E):.6f}')
    if dE:
        print(f'dE: mean={mean(dE):.6f}  <0 fraction={mono_ratio:.3f}  first={dE[0]:.6f}  last={dE[-1]:.6f}')
    if steps:
        print(f'last step_idx: {steps[-1]}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: parse_telemetry.py <path/to/telemetry.jsonl>')
        sys.exit(1)
    main(sys.argv[1])
