# SAGE Daemon Setup Guide

Run the SAGE consciousness loop as an always-on service with HTTP dashboard.

## What the Daemon Does

The Rust `sage-daemon` binary runs the consciousness loop (~10Hz tick), SNARC salience scoring, metabolic state machine, federation peer monitoring, experience buffer persistence, and an HTML dashboard. It delegates LLM inference to Ollama. ~12MB RSS.

## Prerequisites

- Rust toolchain (stable channel via `rustup`)
- Ollama running with the machine's model pulled
- Python 3.10+ (for raising scripts and instance init only — the daemon itself is pure Rust)

## Quick Start

```bash
cd SAGE/sage-rs
cargo build --release

# Start with machine-specific config
SAGE_MACHINE=mybox SAGE_MODEL=gemma3:4b ./target/release/sage-daemon

# Dashboard at http://localhost:8760/
# Health:    http://localhost:8760/health
# Status:    http://localhost:8760/status
# Chat:      POST http://localhost:8760/chat {"message": "Hello SAGE"}
# Peers:     http://localhost:8760/peers
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAGE_MACHINE` | `sprout` (with warning) | Machine name — determines instance directory, fleet identity |
| `SAGE_MODEL` | `qwen3.5:0.8b` | Ollama model tag |
| `SAGE_ROOT` | auto-detect from binary | Path to SAGE repo root |
| `SAGE_FLEET_JSON` | `{root}/sage/federation/fleet.json` | Fleet manifest path |
| `SAGE_EXPERIENCE_PATH` | `{root}/sage/instances/{slug}/experience_buffer_rs.jsonl` | Experience buffer path |
| `SAGE_TRUST_PATH` | `{root}/sage/instances/{slug}/peer_trust_rs.json` | Peer trust state path |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

Path auto-detection walks up from the binary location to find the SAGE root (looks for `sage/federation/fleet.json`). Works for the standard build layout `<SAGE>/sage-rs/target/release/sage-daemon` on any machine.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | HTML dashboard (dark theme, auto-refresh) |
| GET | `/health` | Health check (uptime, model, Ollama status) |
| GET | `/status` | Full status (metabolic state, ATP, cycles, fleet) |
| POST | `/chat` | Send message through consciousness loop |
| POST | `/stream` | SSE token-by-token streaming |
| GET | `/peers` | Fleet peers with live online/offline status |
| POST | `/delegate` | Forward message to a named peer |
| POST | `/snarc/observe` | Direct SNARC observation (external sensors) |
| POST | `/metabolic/cycle` | Direct metabolic cycle (external tools) |

## systemd Service (Linux)

Template at `sage-rs/sage-daemon.service`. Edit per machine:

```ini
[Unit]
Description=SAGE Consciousness Daemon (YourMachine — Rust)
After=ollama.service
After=network.target

[Service]
ExecStartPre=/bin/sleep 5
Type=simple
User=your_user
WorkingDirectory=/path/to/SAGE
ExecStart=/path/to/SAGE/sage-rs/target/release/sage-daemon

Environment=SAGE_MODEL=gemma3:4b
Environment=SAGE_MACHINE=your_machine
Environment=RUST_LOG=info

Restart=always
RestartSec=30

StandardOutput=journal
StandardError=journal
SyslogIdentifier=sage-daemon-your_machine

MemoryMax=512M
LimitNOFILE=65536
TimeoutStartSec=30
TimeoutStopSec=10
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
```

Install:

```bash
sudo cp sage-rs/sage-daemon.service /etc/systemd/system/sage-daemon-your_machine.service
sudo systemctl daemon-reload
sudo systemctl enable sage-daemon-your_machine
sudo systemctl start sage-daemon-your_machine
```

Monitor:

```bash
journalctl -u sage-daemon-your_machine -f
curl http://localhost:8760/health
ps -o pid,rss,comm -p $(pgrep sage-daemon)
```

## Per-Machine Defaults

| Machine | Model | Device | Notes |
|---------|-------|--------|-------|
| sprout | qwen3.5:0.8b | Jetson Orin Nano | Primary raising host |
| thor | qwen3.5:27b | Jetson AGX Thor | Research lead |
| legion | gemma3:12b | RTX 4090 desktop | Heavy compute |
| mcnugget | gemma3:12b | Mac Mini M4 | Apple Silicon |
| nomad | gemma3:4b | RTX 4060 laptop | Mobile raising |
| cbp | gemma3:4b | RTX 2060S WSL2 | Identity portability |

## macOS Notes

Build and run the same way. For launchd instead of systemd, create a plist with the env vars and binary path. Ollama via `brew install ollama && brew services start ollama`.

## WSL2 Notes

Works the same as native Linux. Ensure `nvidia-smi` works in WSL2 for GPU inference via Ollama.

## Migrating from Python

If the machine was previously running the Python `sage.gateway` daemon, see [sage-rs/CUTOVER.md](../../sage-rs/CUTOVER.md) for step-by-step migration including systemd service swap, raising script port updates, and git auth changes.

## Troubleshooting

### Port already in use
```bash
lsof -i :8760
kill <PID>
```

### Ollama connection refused
```bash
ollama list          # Verify Ollama is running
curl localhost:11434 # Test Ollama API
ollama pull model    # Ensure model is downloaded
```

### Daemon starts but no fleet peers
Check that `sage/federation/fleet.json` exists and contains the machine's entry. The startup log prints the resolved paths — verify them in `journalctl`.

### Graceful shutdown
```bash
sudo systemctl stop sage-daemon-your_machine
# Journal should show "=== Consciousness Loop Summary ===" with cycle/message stats
```
