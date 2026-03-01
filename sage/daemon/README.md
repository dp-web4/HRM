# SAGE Daemon Architecture

## Two Daemon Implementations

There are **two different daemon implementations** in the HRM codebase. This document clarifies which one to use.

### ✅ Production Daemon (USE THIS ONE)

**Location:** `sage/gateway/sage_daemon.py`
**Command:** `python3 -m sage.gateway.sage_daemon`
**Port:** 8750 (default, configurable via `SAGE_PORT`)

**Features:**
- Full SAGE consciousness loop (SAGEConsciousness.run())
- Web dashboard at `http://localhost:8750/` with live metabolic state display
- Instance directory architecture (`sage/instances/{machine}-{model}/`)
- Fleet awareness and peer communication
- SSE streaming for real-time updates
- ATP allocation and metabolic states (WAKE, FOCUS, REST, DREAM, CRISIS)
- Message queue for external communication
- Automatic version stamping (git commit hash)

**Used by:**
- McNugget (Mac Mini M4) - Gemma 3 12B
- Nomad (Jetson Orin Nano) - Gemma 3 4B
- CBP (WSL2) - TinyLlama
- Legion (RTX 4090) - Qwen 2 0.5B
- Sprout (Jetson Orin Nano) - Qwen 2.5 0.5B
- **Thor (Jetson AGX Thor) - Qwen 2.5 14B** ← Now working!

**Startup script:** `sage/scripts/ensure_daemon.sh`

### ❌ Simple REST Daemon (DEPRECATED)

**Location:** `sage/daemon/sage_server.py`
**Command:** `python3 -m sage.daemon.sage_server`
**Port:** 8765 (hard-coded)

**Features:**
- Simple FastAPI REST API
- Basic text generation endpoint
- No dashboard (only Swagger UI at `/docs`)
- No consciousness loop
- No metabolic states
- No fleet awareness

**Status:** Updated to use instance architecture for consistency, but **NOT USED IN PRODUCTION**.

This was an early prototype that has been superseded by the full consciousness daemon.

## Quick Start on Thor

```bash
# Start the daemon
export SAGE_PORT=8750
python3 -m sage.gateway.sage_daemon

# Or use the startup script (recommended)
source sage/scripts/ensure_daemon.sh

# Access the dashboard
open http://localhost:8750/
```

## Instance Architecture

Both daemons now support the instance directory architecture:

```
sage/instances/
├── thor-qwen2.5-14b/
│   ├── instance.json          # Instance metadata
│   ├── identity.json          # Identity state
│   ├── daemon_state.json      # Daemon state
│   ├── experience_buffer.json # SNARC memory
│   ├── peer_trust.json        # Trust scores
│   └── sessions/              # Session logs
├── sprout-qwen2.5-0.5b/
└── ... (other instances)
```

State is automatically resolved via `InstancePaths.resolve(machine=MACHINE_NAME)`.

## Discovery Notes (2026-02-28)

**Issue:** Thor daemon wasn't showing dashboard like other machines.

**Root Cause:** Started the wrong daemon (`sage.daemon.sage_server` instead of `sage.gateway.sage_daemon`).

**Fix:** Start `sage.gateway.sage_daemon` which includes the full consciousness loop and dashboard.

The confusion arose because there are two daemon implementations in the codebase, and the simple one (`sage_server.py`) was mistakenly started instead of the production one.

## Port Reference

| Machine   | Daemon Port | Dashboard URL              |
|-----------|-------------|----------------------------|
| Sprout    | 8750        | http://localhost:8750/     |
| Thor      | 8750        | http://localhost:8750/     |
| McNugget  | 8750        | http://localhost:8750/     |
| Legion    | 8750        | http://localhost:8750/     |
| Nomad     | 8750        | http://localhost:8750/     |
| CBP       | 8750        | http://localhost:8750/     |

All machines use the same port by default for consistency.
