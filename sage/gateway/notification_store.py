"""
Notification store — JSONL-backed persistence for human-directed message notifications.

Thread-safe append/read/acknowledge following the chat_history pattern
in gateway_server.py. Auto-prunes to the most recent 200 entries on write.
"""

import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

_lock = threading.Lock()
_MAX_ENTRIES = 200


def append_notification(instance_paths, entry: Dict[str, Any]):
    """Append a notification entry to the instance's notifications.jsonl.

    entry should contain: id, timestamp, source, source_detail,
    text_snippet, patterns_matched, acknowledged.
    """
    path: Path = instance_paths.notifications
    with _lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
            # Auto-prune if over budget
            if path.exists() and path.stat().st_size > 0:
                _prune_if_needed(path)
        except Exception as e:
            print(f"[Notifications] write error: {e}")


def read_notifications(instance_paths, unread_only: bool = True,
                       limit: int = 50) -> List[Dict[str, Any]]:
    """Read notifications from JSONL, optionally filtering to unread only."""
    path: Path = instance_paths.notifications
    if not path.exists():
        return []
    entries = []
    with _lock:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if unread_only and entry.get('acknowledged', False):
                        continue
                    entries.append(entry)
        except Exception as e:
            print(f"[Notifications] read error: {e}")
    # Most recent first, capped at limit
    entries.reverse()
    return entries[:limit]


def acknowledge_notification(instance_paths, notification_id: str):
    """Mark a notification as acknowledged by rewriting the JSONL."""
    path: Path = instance_paths.notifications
    if not path.exists():
        return
    with _lock:
        try:
            entries = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get('id') == notification_id:
                        entry['acknowledged'] = True
                    entries.append(entry)
            with open(path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Notifications] acknowledge error: {e}")


def get_unread_count(instance_paths) -> int:
    """Fast count of unacknowledged notifications."""
    path: Path = instance_paths.notifications
    if not path.exists():
        return 0
    count = 0
    with _lock:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not entry.get('acknowledged', False):
                        count += 1
        except Exception as e:
            print(f"[Notifications] count error: {e}")
    return count


def _prune_if_needed(path: Path):
    """Keep only the most recent _MAX_ENTRIES entries. Caller holds _lock."""
    try:
        entries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]
            with open(path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Notifications] prune error: {e}")
