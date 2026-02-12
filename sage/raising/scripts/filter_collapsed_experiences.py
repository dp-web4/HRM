#!/usr/bin/env python3
"""
Filter collapsed experiences from buffer.

Created: 2026-02-12
Purpose: Remove S68 experiences which contain question loop collapse patterns.

These experiences have high salience (0.57-0.83) but contain:
- Repetitive question patterns ("What's the next masterpiece?" x30)
- Task-switching collapse (code generation instead of conversation)
- Loop patterns that would contaminate LoRA training

Usage:
    python filter_collapsed_experiences.py --dry-run  # Preview changes
    python filter_collapsed_experiences.py            # Apply filter
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil


def load_buffer(path: Path) -> list:
    """Load experience buffer as list of experiences."""
    with open(path) as f:
        return json.load(f)


def save_buffer(experiences: list, path: Path):
    """Save filtered buffer."""
    with open(path, 'w') as f:
        json.dump(experiences, f, indent=2)


def detect_collapse_patterns(response: str) -> tuple[bool, str]:
    """
    Detect collapse patterns in response text.

    Returns:
        (is_collapsed, reason)
    """
    # Pattern 1: Repetitive phrases (same phrase 3+ times)
    words = response.lower().split()
    if len(words) > 20:
        # Check for phrase repetition
        phrase_len = 5
        phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words) - phrase_len)]
        phrase_counts = {}
        for p in phrases:
            phrase_counts[p] = phrase_counts.get(p, 0) + 1

        max_repeat = max(phrase_counts.values()) if phrase_counts else 0
        if max_repeat >= 3:
            return True, f"repetitive_phrase (repeated {max_repeat}x)"

    # Pattern 2: Question flood (too many questions)
    question_count = response.count('?')
    if question_count > 10:
        return True, f"question_flood ({question_count} questions)"

    # Pattern 3: Task switch (code generation in conversation)
    code_markers = ['def ', 'function ', 'class ', 'import ', '```python', '```javascript']
    if any(marker in response.lower() for marker in code_markers):
        # Check if this is supposed to be conversation
        if 'Write a' in response or 'Create a' in response:
            return True, "task_switch_to_code"

    return False, ""


def filter_session(experiences: list, session_id: int, dry_run: bool = True) -> tuple[list, list]:
    """
    Filter out experiences from a specific session.

    Returns:
        (filtered_experiences, removed_experiences)
    """
    filtered = []
    removed = []

    for exp in experiences:
        if exp.get('session') == session_id:
            removed.append(exp)
        else:
            filtered.append(exp)

    return filtered, removed


def filter_collapsed(experiences: list, dry_run: bool = True) -> tuple[list, list]:
    """
    Filter experiences with collapse patterns.

    Returns:
        (filtered_experiences, removed_experiences)
    """
    filtered = []
    removed = []

    for exp in experiences:
        is_collapsed, reason = detect_collapse_patterns(exp.get('response', ''))
        if is_collapsed:
            exp['filter_reason'] = reason
            removed.append(exp)
        else:
            filtered.append(exp)

    return filtered, removed


def main():
    parser = argparse.ArgumentParser(description='Filter collapsed experiences from buffer')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--session', type=int, help='Filter specific session ID')
    parser.add_argument('--pattern-detect', action='store_true', help='Use pattern detection instead of session ID')
    args = parser.parse_args()

    buffer_path = Path('/home/sprout/ai-workspace/HRM/sage/raising/state/experience_buffer.json')

    if not buffer_path.exists():
        print(f"Buffer not found: {buffer_path}")
        return

    experiences = load_buffer(buffer_path)
    print(f"Loaded {len(experiences)} experiences")

    if args.session:
        filtered, removed = filter_session(experiences, args.session, args.dry_run)
        filter_desc = f"session {args.session}"
    elif args.pattern_detect:
        filtered, removed = filter_collapsed(experiences, args.dry_run)
        filter_desc = "collapse patterns"
    else:
        # Default: filter S68
        filtered, removed = filter_session(experiences, 68, args.dry_run)
        filter_desc = "session 68 (default)"

    print(f"\nFiltering {filter_desc}:")
    print(f"  - Original: {len(experiences)} experiences")
    print(f"  - Removed: {len(removed)} experiences")
    print(f"  - Remaining: {len(filtered)} experiences")

    if removed:
        print(f"\n  Removed experiences:")
        for exp in removed:
            session = exp.get('session', '?')
            turn = exp.get('metadata', {}).get('turn', '?')
            salience = exp.get('salience', {}).get('total', '?')
            response_preview = exp.get('response', '')[:80] + '...'
            reason = exp.get('filter_reason', 'session_match')
            print(f"    - S{session}/T{turn}: salience={salience:.2f}, reason={reason}")
            print(f"      {response_preview}")

    if args.dry_run:
        print(f"\n  [DRY RUN] No changes made")
    else:
        # Backup original
        backup_path = buffer_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        shutil.copy(buffer_path, backup_path)
        print(f"\n  Backup saved to: {backup_path}")

        # Save filtered
        save_buffer(filtered, buffer_path)
        print(f"  Filtered buffer saved: {len(filtered)} experiences")


if __name__ == '__main__':
    main()
