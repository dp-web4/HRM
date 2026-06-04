# SAGE Rust Migration Plan

**Goal**: Replace the Python SAGE daemon (~580MB RSS) with a Rust binary (~20-30MB) on the Jetson Orin Nano. Memory savings of ~550MB on an 8GB machine.

**Strategy**: Incremental migration. Python and Rust coexist. Rust daemon runs on port 8760 during development while Python stays on 8750. Sprint by sprint, the Rust binary takes over responsibilities until it's a drop-in replacement.

**What stays in Python**: Raising scripts (~50K lines), IRP ML plugins (~22K lines), cognition research code (~43K lines). These are not on the hot path and communicate via JSON files + HTTP.

## Workspace Structure

```
sage-rs/
  Cargo.toml                (workspace: sage-lib + sage-daemon)
  sage-lib/                  (library: domain logic, testable without async/HTTP)
    src/
      snarc/                 (5 detectors + temporal math)
      metabolic/             (state machine + circadian clock)
      identity/              (3-layer identity, HMAC signing)
      federation/            (fleet registry, peer trust)
      consciousness/         (12-step loop, observation, salience)
      experience/            (buffer, JSONL persistence)
  sage-daemon/               (binary: HTTP server, Ollama client, daemon lifecycle)
    src/
      server/                (axum routes: health, chat, stream, dashboard, peers)
      ollama/                (reqwest client to localhost:11434)
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| tokio | Async runtime (timers, channels, signals) |
| axum | HTTP server |
| reqwest (rustls-tls) | HTTP client (Ollama, peers) |
| serde + serde_json | All serialization |
| chrono | Timestamps, circadian |
| hmac + sha2 | Identity signing |
| tracing | Structured logging |
| uuid | Message IDs |

No ML crates needed — Ollama is accessed via HTTP.

## Sprint Plan

### Sprint 0: SNARC Math Foundation (3-4 days)

Port the temporal math and Surprise detector. Pure Rust, no async, no HTTP.

- `temporal.rs`: TimestampedDeque, time_decay_weight, weighted_percentile_of
- `surprise.rs`: SimplePredictorEMA, SurpriseDetector
- Unit tests with parity checks against Python reference values
- `cargo test` passes, `cargo build --release` works on aarch64

### Sprint 1: SNARC Complete + Metabolic (4-5 days)

Port remaining 4 SNARC detectors and the metabolic state machine.

- `novelty.rs`, `arousal.rs`, `reward.rs`, `conflict.rs`
- `controller.rs`: MetabolicState enum, state transitions, ATP management
- `circadian.rs`: CircadianPhase, sinusoidal bias
- CLI simulation: `sage-daemon --simulate 1000` prints state transitions
- 1000 cycles < 1 second on Sprout

### Sprint 2: Identity + Federation (3-4 days)

Port identity and fleet infrastructure. Reads actual instance files.

- `provider.rs`: load/save identity.json, unseal identity.sealed
- `signing.rs`: HMAC-SHA256 signing context
- `fleet.rs`: load fleet.json, peer lookup
- `peer_trust.rs`: T3 EMA tracking, JSON persistence
- Integration test loads real identity from disk

### Sprint 3: HTTP Gateway (5-7 days)

Build axum server with core endpoints. First actually useful binary.

- Routes: `/health`, `/status`, `/chat`, `/stream` (SSE), `/peers`
- Ollama client: `POST /api/generate` and `/api/chat`
- Message queue via tokio channels
- Runs alongside Python daemon on port 8760
- Memory < 30MB RSS

### Sprint 4: Consciousness Loop (5-7 days)

Wire everything together. The loop ticks at ~1Hz.

- 12-step cycle: sense → salience → metabolize → ... → act
- Messages flow: HTTP → queue → loop → Ollama → response → SSE
- Experience buffer (JSONL append)
- Graceful SIGTERM shutdown
- systemd service file
- Full daemon replaces Python for basic operation

### Sprint 5: Federation Networking (3-4 days)

Peer communication for fleet participation.

- PeerMonitor: health polling every 30s
- PeerClient: send/receive messages to/from peers
- Trust tracker updates on success/failure
- Cross-machine test: Sprout Rust → Thor Python

### Sprint 6: Dashboard + Cutover (3-4 days)

Final sprint. Dashboard, remaining endpoints, production switch.

- Serve HTML dashboard (embedded via include_str!)
- Chat history, notifications endpoints
- systemd service cutover on Sprout
- 24h stability test
- Verify raising scripts still work against Rust daemon

## Build

Native compilation on Jetson (aarch64). Debug builds during dev, release for deployment.

```bash
cd sage-rs
cargo build           # debug, fast iteration
cargo test            # run all tests
cargo build --release # deployment binary
```

## Estimated Total: 26-35 days (5-7 weeks)
