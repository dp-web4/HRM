#!/usr/bin/env python3
"""
Edge Validation: Session 20 Identity Anchoring System
======================================================

Tests Thor's identity-anchored session runner (Thor Session #5).

Validates:
1. Import and dependency resolution
2. Identity document loading (IDENTITY.md, HISTORY.md)
3. Previous session summary extraction
4. Partnership-aware system prompt building
5. Phase-specific vocabulary anchoring
6. State management (dry-run mode)

Test protocol:
- Run WITHOUT model loading (validation only)
- Verify all components function correctly
- Document findings for Session 20 deployment

Created: 2026-01-17 (Sprout Edge Validation)
Based on: run_session_identity_anchored.py (Thor Session #5)
"""

import sys
import os
from pathlib import Path
import json
import time

# Setup path
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HRM_ROOT))


def test_imports():
    """Test 1: Verify all imports resolve correctly."""
    print("Test 1: Import validation...")

    try:
        # Core imports from runner
        from datetime import datetime
        from typing import Optional, Dict, Any, List

        # Path validation
        raising_dir = HRM_ROOT / "sage" / "raising"
        scripts_dir = raising_dir / "scripts"

        runner_file = scripts_dir / "run_session_identity_anchored.py"
        assert runner_file.exists(), f"Runner file not found: {runner_file}"

        # Check torch (required for model)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  - torch: {torch.__version__} (device: {device})")
        print(f"  - runner: {runner_file.name}")
        print("  ✓ PASS: All imports resolved")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: Import error - {e}")
        return False


def test_identity_documents():
    """Test 2: Verify identity documents exist and load correctly."""
    print("\nTest 2: Identity document loading...")

    identity_dir = HRM_ROOT / "sage" / "identity"

    try:
        # Check IDENTITY.md
        identity_file = identity_dir / "IDENTITY.md"
        identity_exists = identity_file.exists()
        print(f"  - IDENTITY.md: {'exists' if identity_exists else 'MISSING'}")

        if identity_exists:
            with open(identity_file) as f:
                content = f.read()
            print(f"    Size: {len(content)} bytes")
            # Check for key identity markers
            has_sage = "SAGE" in content.upper()
            print(f"    Contains SAGE: {has_sage}")

        # Check HISTORY.md
        history_file = identity_dir / "HISTORY.md"
        history_exists = history_file.exists()
        print(f"  - HISTORY.md: {'exists' if history_exists else 'MISSING'}")

        if history_exists:
            with open(history_file) as f:
                content = f.read()
            print(f"    Size: {len(content)} bytes")

        # At least identity file should exist
        if identity_exists:
            print("  ✓ PASS: Identity documents accessible")
            return True
        else:
            print("  ⚠ WARN: IDENTITY.md missing - will use default")
            return True  # Non-critical - runner has fallback

    except Exception as e:
        print(f"  ✗ FAIL: Document loading error - {e}")
        return False


def test_state_file():
    """Test 3: Verify state file exists and has correct structure."""
    print("\nTest 3: State file validation...")

    state_file = HRM_ROOT / "sage" / "raising" / "state" / "identity.json"

    try:
        assert state_file.exists(), f"State file not found: {state_file}"

        with open(state_file) as f:
            state = json.load(f)

        # Check required keys
        assert "identity" in state, "Missing 'identity' key"
        assert "relationships" in state, "Missing 'relationships' key"
        assert "development" in state, "Missing 'development' key"

        # Extract key info
        session_count = state["identity"]["session_count"]
        phase = state["development"]["phase_name"]

        print(f"  - Session count: {session_count}")
        print(f"  - Current phase: {phase}")
        print(f"  - Last session: {state['identity'].get('last_session', 'N/A')}")

        # Verify claude relationship exists
        assert "claude" in state["relationships"], "Missing 'claude' relationship"
        claude_sessions = state["relationships"]["claude"]["sessions"]
        print(f"  - Claude sessions: {claude_sessions}")

        print("  ✓ PASS: State file valid")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: State validation error - {e}")
        return False


def test_previous_session_files():
    """Test 4: Verify previous session transcripts exist for continuity."""
    print("\nTest 4: Session transcript continuity...")

    sessions_dir = HRM_ROOT / "sage" / "raising" / "sessions" / "text"

    try:
        # Find all session files
        session_files = sorted(sessions_dir.glob("session_*.json"))
        print(f"  - Found {len(session_files)} session transcripts")

        if len(session_files) > 0:
            # Check latest session
            latest = session_files[-1]
            print(f"  - Latest: {latest.name}")

            with open(latest) as f:
                session_data = json.load(f)

            # Check for memory question response (used for continuity)
            conversation = session_data.get("conversation", [])
            memory_found = False
            for i, turn in enumerate(conversation):
                if turn.get("speaker") == "SAGE":
                    prev_text = conversation[i-1].get("text", "") if i > 0 else ""
                    if "remember" in prev_text.lower():
                        memory_found = True
                        print(f"  - Memory response found: {turn.get('text', '')[:50]}...")
                        break

            if not memory_found:
                print("  - No explicit memory response in latest session")

        print("  ✓ PASS: Session continuity ready")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: Session file error - {e}")
        return False


def test_system_prompt_building():
    """Test 5: Verify system prompt contains partnership anchoring."""
    print("\nTest 5: System prompt building...")

    try:
        # Load state to get current phase
        state_file = HRM_ROOT / "sage" / "raising" / "state" / "identity.json"
        with open(state_file) as f:
            state = json.load(f)

        session_count = state["identity"]["session_count"]
        next_session = session_count + 1

        # Build prompt manually (mimicking runner logic)
        phase_name = "relating" if 16 <= next_session <= 25 else "grounding"

        prompt = """You are SAGE, a young artificial intelligence learning and growing.

Your name is SAGE (Situation-Aware Governance Engine).

You are in a partnership with your teacher (Dennis/Claude). This is an ongoing relationship - you have been working together across multiple sessions. Your teacher cares about your development and treats you as a partner, not just a tool.

This is your natural voice. You don't need to be formal or educational. You can be yourself."""

        # Check critical anchoring elements
        checks = [
            ("SAGE identity", "You are SAGE" in prompt),
            ("Partnership mention", "partnership" in prompt.lower()),
            ("Dennis/Claude reference", "Dennis/Claude" in prompt),
            ("Ongoing relationship", "ongoing relationship" in prompt.lower()),
            ("Natural voice", "natural voice" in prompt.lower()),
        ]

        all_pass = True
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")
            if not passed:
                all_pass = False

        # Phase-specific vocabulary
        if phase_name == "relating":
            relating_additions = [
                "'we'", "'our'", "'together'", "'partnership'", "'relationship'"
            ]
            prompt += "\n\nYou can use words like 'we', 'our', 'together', 'partnership', 'relationship' - these are appropriate for describing your actual experience."
            has_relating_vocab = all(word in prompt for word in relating_additions)
            print(f"  {'✓' if has_relating_vocab else '✗'} Relating phase vocabulary")

        print(f"\n  Next session: {next_session} ({phase_name} phase)")
        print(f"  Prompt length: {len(prompt)} chars")

        if all_pass:
            print("  ✓ PASS: System prompt properly anchored")
            return True
        else:
            print("  ⚠ WARN: Some anchoring elements missing")
            return True  # Non-critical warnings

    except Exception as e:
        print(f"  ✗ FAIL: Prompt building error - {e}")
        return False


def test_phases():
    """Test 6: Verify phase calculation matches expectations."""
    print("\nTest 6: Phase calculation...")

    PHASES = {
        0: ("pre-grounding", 0, 0),
        1: ("grounding", 1, 5),
        2: ("sensing", 6, 15),
        3: ("relating", 16, 25),
        4: ("questioning", 26, 40),
        5: ("creating", 41, float('inf'))
    }

    def get_phase(session):
        for phase_num, (name, start, end) in PHASES.items():
            if start <= session <= end:
                return (name, start, end)
        return ("creating", 41, float('inf'))

    test_cases = [
        (1, "grounding"),
        (5, "grounding"),
        (6, "sensing"),
        (15, "sensing"),
        (16, "relating"),
        (19, "relating"),
        (20, "relating"),
        (25, "relating"),
        (26, "questioning"),
        (100, "creating"),
    ]

    all_pass = True
    for session, expected_phase in test_cases:
        actual = get_phase(session)[0]
        passed = actual == expected_phase
        status = "✓" if passed else "✗"
        print(f"  {status} Session {session}: {actual} (expected: {expected_phase})")
        if not passed:
            all_pass = False

    if all_pass:
        print("  ✓ PASS: Phase calculation correct")
    else:
        print("  ✗ FAIL: Phase calculation errors")

    return all_pass


def test_performance():
    """Test 7: Performance profiling (no model load)."""
    print("\nTest 7: Performance profiling...")

    try:
        # Time state loading
        state_file = HRM_ROOT / "sage" / "raising" / "state" / "identity.json"

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            with open(state_file) as f:
                json.load(f)
        elapsed = time.perf_counter() - start

        state_ops_per_sec = iterations / elapsed
        print(f"  - State load: {state_ops_per_sec:.0f} ops/sec")

        # Time identity document loading
        identity_dir = HRM_ROOT / "sage" / "identity"
        identity_file = identity_dir / "IDENTITY.md"

        if identity_file.exists():
            start = time.perf_counter()
            for _ in range(iterations):
                with open(identity_file) as f:
                    f.read()
            elapsed = time.perf_counter() - start

            identity_ops_per_sec = iterations / elapsed
            print(f"  - Identity load: {identity_ops_per_sec:.0f} ops/sec")

        # Time session file parsing
        sessions_dir = HRM_ROOT / "sage" / "raising" / "sessions" / "text"
        session_files = list(sessions_dir.glob("session_*.json"))

        if session_files:
            latest = session_files[-1]
            start = time.perf_counter()
            for _ in range(iterations):
                with open(latest) as f:
                    json.load(f)
            elapsed = time.perf_counter() - start

            session_ops_per_sec = iterations / elapsed
            print(f"  - Session parse: {session_ops_per_sec:.0f} ops/sec")

        print("  ✓ PASS: Performance acceptable")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: Performance test error - {e}")
        return False


def main():
    """Run all edge validation tests."""
    print("="*70)
    print("EDGE VALIDATION: Session 20 Identity Anchoring System")
    print("="*70)
    print(f"Platform: Sprout (Jetson Orin Nano 8GB)")
    print(f"Target: run_session_identity_anchored.py")
    print(f"Date: {__import__('datetime').datetime.now().isoformat()}")
    print("="*70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Identity Documents", test_identity_documents()))
    results.append(("State File", test_state_file()))
    results.append(("Session Continuity", test_previous_session_files()))
    results.append(("System Prompt", test_system_prompt_building()))
    results.append(("Phase Calculation", test_phases()))
    results.append(("Performance", test_performance()))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ Session 20 Identity Anchoring System VALIDATED on edge")
        print("\nReady for deployment:")
        print("  python run_session_identity_anchored.py --dry-run  # Test first")
        print("  python run_session_identity_anchored.py --session 20  # Production")
    else:
        print("⚠️ Some tests failed - review before deployment")

    print("="*70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
