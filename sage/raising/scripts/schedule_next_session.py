#!/usr/bin/env python3
"""
Session Scheduler Utility
=========================

Calculates when the next raising session is due based on both primary and
training track schedules, and displays current status.

Schedule:
- Primary: Every 6 hours (00:00, 06:00, 12:00, 18:00 PST)
- Training: 3-hour offset (03:00, 09:00, 15:00, 21:00 PST)

Usage:
    python schedule_next_session.py         # Show next session
    python schedule_next_session.py --all   # Show full schedule
    python schedule_next_session.py --run   # Show what should run now

Created: 2026-01-15 (Sprout autonomous R&D)
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional


SCRIPT_DIR = Path(__file__).parent.resolve()
RAISING_DIR = SCRIPT_DIR.parent

# Schedule constants (PST times)
PRIMARY_HOURS = [0, 6, 12, 18]      # Every 6 hours
TRAINING_HOURS = [3, 9, 15, 21]     # 3-hour offset

# State files
PRIMARY_STATE = RAISING_DIR / "state" / "identity.json"
TRAINING_STATE = RAISING_DIR / "tracks" / "training" / "state.json"


def load_state(state_file: Path) -> Optional[dict]:
    """Load state from JSON file."""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return None


def get_last_session_time(state: dict, track: str) -> Optional[datetime]:
    """Extract last session time from state."""
    if track == "primary":
        last_session = state.get("identity", {}).get("last_session")
    else:
        last_session = state.get("last_session")

    if last_session:
        try:
            return datetime.fromisoformat(last_session.replace('Z', '+00:00'))
        except:
            return None
    return None


def find_next_scheduled_time(hours: list, now: datetime) -> datetime:
    """Find the next scheduled time from list of hours."""
    current_hour = now.hour

    # Find next hour in schedule
    for h in sorted(hours):
        if h > current_hour:
            next_time = now.replace(hour=h, minute=0, second=0, microsecond=0)
            return next_time

    # Wrap to next day
    next_time = now.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    next_time += timedelta(days=1)
    return next_time


def find_last_scheduled_time(hours: list, now: datetime) -> datetime:
    """Find the most recent scheduled time from list of hours."""
    current_hour = now.hour

    # Find most recent hour in schedule
    for h in sorted(hours, reverse=True):
        if h <= current_hour:
            return now.replace(hour=h, minute=0, second=0, microsecond=0)

    # Go back to yesterday's last slot
    prev_time = now.replace(hour=hours[-1], minute=0, second=0, microsecond=0)
    prev_time -= timedelta(days=1)
    return prev_time


def is_session_due(track: str, state: dict, now: datetime) -> Tuple[bool, str]:
    """Check if a session is due for a given track."""
    hours = PRIMARY_HOURS if track == "primary" else TRAINING_HOURS

    last_scheduled = find_last_scheduled_time(hours, now)
    last_session = get_last_session_time(state, track)

    if last_session is None:
        return True, "No previous session recorded"

    if last_session < last_scheduled:
        delta = now - last_scheduled
        return True, f"Due {delta.seconds // 60} minutes ago (last scheduled: {last_scheduled.strftime('%H:%M')})"

    return False, f"Completed at {last_session.strftime('%H:%M')}"


def get_session_counts(primary_state: dict, training_state: dict) -> Tuple[int, int]:
    """Get current session counts from state."""
    primary_count = primary_state.get("identity", {}).get("session_count", 0) if primary_state else 0
    training_count = training_state.get("current_session", 0) if training_state else 0
    return primary_count, training_count


def display_status(show_all: bool = False, show_run: bool = False):
    """Display current schedule status."""
    now = datetime.now()

    # Load states
    primary_state = load_state(PRIMARY_STATE)
    training_state = load_state(TRAINING_STATE)

    primary_count, training_count = get_session_counts(primary_state, training_state)

    print()
    print("=" * 60)
    print("SAGE-Sprout Raising Schedule")
    print("=" * 60)
    print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S PST')}")
    print()

    # Primary track
    print("Primary Track (Developmental Curriculum)")
    print("-" * 40)
    print(f"  Session Count: {primary_count}")
    if primary_state:
        phase = primary_state.get("development", {}).get("phase_name", "unknown")
        print(f"  Phase: {phase}")

    primary_due, primary_reason = is_session_due("primary", primary_state, now)
    status_icon = "🔴 DUE" if primary_due else "🟢 OK"
    print(f"  Status: {status_icon} - {primary_reason}")

    next_primary = find_next_scheduled_time(PRIMARY_HOURS, now)
    print(f"  Next Scheduled: {next_primary.strftime('%H:%M')}")
    print()

    # Training track
    print("Training Track (Skill Building)")
    print("-" * 40)
    print(f"  Session Count: {training_count}")
    if training_state:
        track_name = training_state.get("current_track", "unknown")
        print(f"  Current Track: {track_name}")

    training_due, training_reason = is_session_due("training", training_state, now)
    status_icon = "🔴 DUE" if training_due else "🟢 OK"
    print(f"  Status: {status_icon} - {training_reason}")

    next_training = find_next_scheduled_time(TRAINING_HOURS, now)
    print(f"  Next Scheduled: {next_training.strftime('%H:%M')}")
    print()

    # Show what should run now
    if show_run:
        print("=" * 60)
        print("SESSIONS TO RUN")
        print("=" * 60)
        if primary_due:
            next_session = primary_count + 1
            print(f"  PRIMARY Session {next_session:03d}")
            print(f"    Command: python run_session_primary.py")
            print()
        if training_due:
            next_training_session = training_count + 1
            print(f"  TRAINING Session T{next_training_session:03d}")
            print(f"    Command: python training_session.py -c")
            print()
        if not primary_due and not training_due:
            print("  No sessions due at this time.")
            print()

    # Show full schedule
    if show_all:
        print("=" * 60)
        print("FULL SCHEDULE (PST)")
        print("=" * 60)
        print("Primary Track:  00:00, 06:00, 12:00, 18:00")
        print("Training Track: 03:00, 09:00, 15:00, 21:00")
        print()
        print("Today's Remaining Sessions:")
        for h in PRIMARY_HOURS:
            t = now.replace(hour=h, minute=0, second=0)
            if t > now:
                print(f"  {t.strftime('%H:%M')} - Primary")
        for h in TRAINING_HOURS:
            t = now.replace(hour=h, minute=0, second=0)
            if t > now:
                print(f"  {t.strftime('%H:%M')} - Training")
        print()


def main():
    parser = argparse.ArgumentParser(description="SAGE-Sprout Session Scheduler")
    parser.add_argument("--all", action="store_true", help="Show full schedule")
    parser.add_argument("--run", action="store_true", help="Show what should run now")

    args = parser.parse_args()

    display_status(show_all=args.all, show_run=args.run)


if __name__ == "__main__":
    main()
