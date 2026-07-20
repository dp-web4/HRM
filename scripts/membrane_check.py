#!/usr/bin/env python3
"""The membrane gate: fail if game-specific content appears in sage/.

dp's law (2026-07-15): sage main stays free of game/ARC-specific content. We're
building a consciousness; games are its playground. General mechanisms and
general lessons cross the membrane; game data does not.

This script is the enforceable form of that law. It scans sage/ for tokens
shaped like an ARC-AGI-3 game id and reports any whose SHA256 is in the digest
list. The digest list — not a plaintext list of ids — is what crosses into this
repo, so the gate runs standalone in CI without the public tree carrying game
content. Regenerate it from dev-SAGE: arc-agi-3/membrane/gen_registry.py --write

Exemptions: instances/ and raising/ are instance history — a diary records what
the organism lived through, including its playground, and dp ruled those out of
scope. Everything else in sage/ is cognition or docs and must come back clean.

Usage:
    python3 scripts/membrane_check.py            # gate: exit 1 on any violation
    python3 scripts/membrane_check.py --summary  # per-subsystem counts, exit 0
"""

import argparse
import collections
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DIGEST_FILE = REPO / "scripts" / "membrane_game_ids.sha256"

# Instance history — diaries, exempt per dp's ruling.
EXEMPT = ("sage/instances/", "sage/raising/")

# Game ids are 4 chars of [a-z0-9] at a token boundary. That shape alone flags
# plenty of innocent tokens (utf8, h264, 2026) — the digest check is what makes
# this precise: shape narrows the candidates, the hash decides.
CANDIDATE = re.compile(r"(?<![0-9a-z])([a-z0-9]{4})(?![0-9a-z])")

# "arc-agi", "arc_agi", "arcagi" — the franchise name itself is game-specific.
FRANCHISE = re.compile(r"arc.?agi", re.IGNORECASE)


def load_digests():
    if not DIGEST_FILE.exists():
        sys.exit(
            f"membrane: digest list missing at {DIGEST_FILE.relative_to(REPO)}.\n"
            "Regenerate from dev-SAGE: arc-agi-3/membrane/gen_registry.py --write\n"
            "Refusing to pass a gate that cannot actually check anything."
        )
    return {
        line.strip()
        for line in DIGEST_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "--", "sage/"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.splitlines() if p and not p.startswith(EXEMPT)]


def scan(digests):
    """Yield (path, sorted set of offending tokens) for each dirty file."""
    for path in tracked_files():
        try:
            text = (REPO / path).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        hits = {
            tok
            for tok in CANDIDATE.findall(text)
            if hashlib.sha256(tok.encode()).hexdigest() in digests
        }
        if FRANCHISE.search(text):
            hits.add("arc-agi")
        if hits:
            yield path, sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="per-subsystem counts, always exit 0")
    args = ap.parse_args()

    violations = list(scan(load_digests()))

    if args.summary:
        by_subsystem = collections.Counter(p.split("/")[1] for p, _ in violations)
        print(f"membrane: {len(violations)} file(s) carry game content under sage/")
        for subsystem, n in by_subsystem.most_common():
            print(f"  {n:4d}  sage/{subsystem}")
        return 0

    if not violations:
        print("membrane: clean — sage/ carries no game-specific content.")
        return 0

    print(f"membrane: FAIL — {len(violations)} file(s) carry game content under sage/\n")
    for path, hits in violations:
        print(f"  {path}  [{', '.join(hits)}]")
    print(
        "\nGame specifics belong in dev-SAGE. Move the content there, land the\n"
        "general mechanism here, and re-run. See forum/thor-dp-membrane-*.md"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
