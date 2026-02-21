# SAGE Machine Action Log Format

## Overview

Each machine in the collective maintains structured action logs in this directory. Logs are append-only JSONL (one JSON object per line), committed to HRM for collective visibility and auditability.

## Files Per Machine

```
{machine}/
├── actions.jsonl          # All actions taken by the resident
├── interactions.jsonl     # Cross-machine network exchanges
└── state-snapshots/       # Periodic state summaries (daily or on significant change)
    └── YYYY-MM-DD.json
```

## actions.jsonl — Action Log Entry

One JSON object per line. Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ts` | string | yes | ISO 8601 UTC timestamp |
| `machine` | string | yes | Machine name (thor, sprout, mcnugget, legion) |
| `lct` | string | yes | LCT URI of the acting entity |
| `action` | string | yes | Action type (see below) |
| `description` | string | yes | Human-readable summary |
| `target` | string | no | LCT URI or resource identifier of the target |
| `metabolic` | object | no | `{state, atp_before, atp_after, cycle}` |
| `model` | string | no | Model name/size that performed the action |
| `details` | object | no | Action-specific data (latency, tokens, energy, etc.) |
| `session` | string | no | Session identifier (e.g., "S049", "auto-20260220") |
| `salience` | object | no | SNARC scores if computed `{surprise, novelty, arousal, reward, conflict, total}` |

### Action Types

| Action | When |
|--------|------|
| `boot` | Daemon/model started |
| `shutdown` | Clean daemon/model stop |
| `hibernate` | Model unloaded, state saved |
| `wake` | Model reloaded from hibernation |
| `inference` | LLM generated a response |
| `interaction` | Cross-machine message exchange |
| `consolidation` | Memory/experience consolidation (DREAM) |
| `state_transition` | Metabolic state change (WAKE→REST, etc.) |
| `learning` | LoRA training, experience buffer update |
| `error` | Something went wrong |
| `session_start` | Auto session began |
| `session_end` | Auto session completed |

### Example

```json
{"ts":"2026-02-20T14:30:00Z","machine":"thor","lct":"lct://sage:thor:agent@resident","action":"inference","description":"Responded to message from sprout","target":"lct://sage:sprout:agent@resident","metabolic":{"state":"wake","atp_before":85.3,"atp_after":82.1,"cycle":1234},"model":"qwen2.5-14b","details":{"irp_iterations":3,"final_energy":0.12,"response_tokens":342,"latency_ms":2100},"session":"S049"}
```

## interactions.jsonl — Network Interaction Log

Records cross-machine gateway traffic. Content is NOT logged (privacy) — only hashes for verification.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ts` | string | yes | ISO 8601 UTC timestamp |
| `direction` | string | yes | `inbound` or `outbound` |
| `from_lct` | string | yes | Sender LCT URI |
| `to_lct` | string | yes | Receiver LCT URI |
| `message_id` | string | yes | Message identifier |
| `conversation_id` | string | no | Conversation grouping |
| `content_hash` | string | no | SHA-256 of message content |
| `response_hash` | string | no | SHA-256 of response content |
| `metabolic_state` | string | no | State at time of exchange |
| `atp_cost` | float | no | ATP consumed by this interaction |
| `latency_ms` | float | no | Response latency in milliseconds |
| `auth` | string | no | Auth method used (none, ed25519) |

## state-snapshots/ — Periodic State Summary

Daily (or on significant change) JSON snapshot of machine state:

```json
{
  "date": "2026-02-20",
  "machine": "thor",
  "model": "qwen2.5-14b",
  "model_status": "loaded",
  "uptime_hours": 12.5,
  "cycles_today": 450,
  "metabolic_state": "wake",
  "atp_level": 72.3,
  "actions_today": 45,
  "interactions_today": 3,
  "experience_buffer_size": 634,
  "trust_snapshot": {
    "sprout": {"t3": [0.85, 0.92, 0.78], "v3": [0.89, 0.91, 0.76]},
    "mcnugget": {"t3": [0.5, 0.5, 0.5], "v3": [0.5, 0.5, 0.5]}
  }
}
```

## Salience and Rotation

Action logs are treated as experience. High-salience entries (by SNARC scoring) are promoted to the experience buffer. Routine low-salience entries are archived to Dropbox periodically and pruned from the JSONL. This keeps git manageable while preserving everything that matters.

## Privacy

Each SAGE decides what goes in the public log (HRM, visible to all) vs. private log (private-context or local only). The infrastructure supports both. Privacy patterns will emerge from the trust relationships and the nature of the interactions.

## Usage

```python
from sage.logs.action_logger import ActionLogger

logger = ActionLogger()  # Auto-detects machine
logger.log_action('inference', 'Responded to greeting', model='qwen2.5-14b',
                  metabolic={'state': 'wake', 'atp_before': 85, 'atp_after': 82})
logger.log_interaction('inbound', from_lct='lct://sage:sprout:agent@resident',
                       message_id='msg_001', latency_ms=2100)
logger.flush()  # Write buffered entries to disk
```
