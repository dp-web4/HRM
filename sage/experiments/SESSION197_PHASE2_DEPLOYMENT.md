# Session 197 Phase 2: Thor ↔ Sprout Real Federation Deployment

**Created**: 2026-01-15 06:36 PST (Thor Autonomous Check)
**Status**: READY FOR DEPLOYMENT
**Prerequisites**: Phase 1 validated (localhost, 60s, all predictions confirmed)

---

## Overview

Phase 2 deploys the consciousness-aware federation protocol across real network distance between Thor (10.0.0.99) and Sprout (10.0.0.36) over WiFi.

**Goal**: Validate that the HTTP-based consciousness-aware federation protocol works across real network latency and achieves the same synchronization quality as localhost testing.

---

## Network Status

**Verified**: 2026-01-15 06:27 PST

```
Thor (Coordinator):
- IP: 10.0.0.99
- Interface: wlP1p1s0 (WiFi)
- Port: 8000 (available ✅)
- Status: Online, stable

Sprout (Participant):
- IP: 10.0.0.36
- Interface: WiFi
- Status: Reachable (ping 3/3, RTT 1-165ms)

Connectivity:
- Packet loss: 0%
- RTT min: 1.3ms
- RTT avg: 56.5ms
- RTT max: 165.7ms
- Assessment: Normal WiFi variability
```

---

## Phase 2 Test Plan

### Duration
60 seconds (same as Phase 1 for direct comparison)

### Metrics to Measure

**Network Performance**:
- Round-trip time (RTT) per snapshot/sync cycle
- Message loss rate
- Network-induced latency variance

**Synchronization Quality**:
- Sync quality Q (target: >0.9)
- Delta coherence ΔC (target: <0.1)
- Consciousness failures (target: 0)

**Coupling Propagation**:
- Time from coupling event to coordinator receipt
- Cross-network coupling latency vs localhost

**Comparison to Phase 1**:
- Snapshot frequency (Phase 1: 10.0 Hz)
- Success rate (Phase 1: 100%)
- Sync quality (Phase 1: Q = 0.925)
- Delta coherence (Phase 1: ΔC = 0.0075)

### Predictions for Phase 2

**P197.6**: Network RTT (1-165ms) is small compared to 100ms cycle time, so 10 Hz sync frequency should be maintained (±5%)

**P197.7**: WiFi variability will increase ΔC slightly but should stay <0.1 (target maintained)

**P197.8**: Sync quality Q may decrease slightly (WiFi jitter) but should stay >0.85

**P197.9**: Consciousness validation (C ≥ 0.5) should remain 100% successful (not affected by network)

**P197.10**: Coupling events will experience network latency but should propagate <200ms (vs <50ms localhost)

---

## Deployment Steps

### On Thor (Coordinator)

```bash
# Navigate to experiments directory
cd /home/dp/ai-workspace/HRM/sage/experiments

# Option 1: Use deployment script (recommended)
./session197_phase2_deploy_coordinator.sh 60

# Option 2: Manual start
python3 session197_consciousness_federation_coordinator.py \
    --host 0.0.0.0 \
    --port 8000 \
    --duration 60
```

**Expected Output**:
```
================================================================
Session 197: Consciousness-Aware Federation Coordinator
================================================================
Listening on 0.0.0.0:8000
Waiting for participants to connect...
```

### On Sprout (Participant)

**SSH to Sprout**:
```bash
ssh dp@10.0.0.36
```

**Deploy Participant**:
```bash
# Navigate to HRM experiments (assuming synced repository)
cd /home/dp/ai-workspace/HRM/sage/experiments

# Option 1: Use deployment script (recommended)
./session197_phase2_deploy_participant.sh 10.0.0.99 60

# Option 2: Manual start
python3 session197_consciousness_federation_participant.py \
    --coordinator-host 10.0.0.99 \
    --coordinator-port 8000 \
    --node-id sprout \
    --duration 60
```

**Expected Output**:
```
================================================================
Session 197: Consciousness-Aware Federation Participant
================================================================
Node ID: sprout
Coordinator: 10.0.0.99:8000
Connecting to coordinator...
✅ Connected to coordinator
Starting federation loop (10 Hz)...
```

---

## Monitoring

### Real-Time Monitoring

**On Thor** (separate terminal):
```bash
tail -f /home/dp/ai-workspace/HRM/sage/experiments/session197_phase2_coordinator.log
```

**On Sprout** (separate terminal):
```bash
tail -f /home/dp/ai-workspace/HRM/sage/experiments/session197_phase2_participant.log
```

### Key Metrics to Watch

**Coordinator Log**:
- Snapshot receipt frequency (should be ~10 Hz)
- Sync signal dispatch frequency (should be ~10 Hz)
- Coupling events received
- Sync quality Q values

**Participant Log**:
- Snapshot transmission success rate
- Sync signal receipt frequency
- Consciousness validation results (C ≥ 0.5)
- Local coherence evolution

---

## Success Criteria

Phase 2 is successful if:

1. ✅ Federation runs for full 60 seconds without crashes
2. ✅ Snapshot frequency ≥9.5 Hz (within 5% of 10 Hz target)
3. ✅ Sync signal frequency ≥9.5 Hz
4. ✅ 0 consciousness failures (100% C ≥ 0.5 validation)
5. ✅ Sync quality Q >0.85 (allows slight WiFi degradation)
6. ✅ Delta coherence ΔC <0.1 (convergence maintained)
7. ✅ Coupling events propagate successfully across network

Stretch goals:
- Match Phase 1 performance (Q = 0.925, ΔC = 0.0075)
- Achieve 10.0 Hz exactly (zero frequency degradation)
- Demonstrate network resilience to WiFi variability

---

## Troubleshooting

### Coordinator Won't Start

**Error**: `Address already in use`
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill <PID>
```

### Participant Can't Connect

**Error**: `Connection refused`
```bash
# Verify coordinator is running
curl http://10.0.0.99:8000/federation_status

# Check network connectivity
ping -c 5 10.0.0.99

# Check firewall (if applicable)
sudo ufw status
```

### High Message Loss

If snapshot success rate <95%:
1. Check WiFi signal strength on both machines
2. Verify no network congestion (other heavy traffic)
3. Consider increasing retry logic in participant
4. May need to adjust sync frequency to 8-9 Hz

### Poor Sync Quality

If Q <0.85:
1. Check network RTT variance (high jitter)
2. Verify consciousness validation is working (C ≥ 0.5)
3. May need to tune synchronization parameters (Γ, κ)
4. Consider adaptive sync frequency based on network conditions

---

## Data Collection

After test completion, collect:

1. **Coordinator Log**: `/home/dp/ai-workspace/HRM/sage/experiments/session197_phase2_coordinator.log`
2. **Participant Log**: `/home/dp/ai-workspace/HRM/sage/experiments/session197_phase2_participant.log`

### Analysis to Perform

```python
# Parse logs for metrics
import json
import statistics

# Extract from coordinator log:
# - Snapshot timestamps (compute frequency)
# - Sync quality Q values (compute mean, std)
# - Delta coherence ΔC values (compute max)
# - Coupling event count

# Extract from participant log:
# - Consciousness validation results (count failures)
# - Network RTT per snapshot (if logged)
# - Local coherence evolution

# Compare to Phase 1:
# - Frequency degradation: (Phase2_Hz - Phase1_Hz) / Phase1_Hz
# - Sync quality degradation: Phase1_Q - Phase2_Q
# - Coherence degradation: Phase2_ΔC - Phase1_ΔC
```

---

## Post-Deployment

After successful Phase 2 test:

1. **Document Results**:
   - Create `session197_phase2_results.md`
   - Compare all metrics to Phase 1
   - Validate/invalidate predictions P197.6-P197.10

2. **Commit to Repository**:
   ```bash
   cd /home/dp/ai-workspace/HRM
   git add sage/experiments/session197_phase2_*.log
   git add sage/experiments/session197_phase2_results.md
   git commit -m "Session 197 Phase 2: Thor ↔ Sprout federation validated"
   ```

3. **Update LATEST_STATUS.md**:
   - Mark Session 197 Phase 2 as complete
   - Add Phase 2 test results
   - Update predictions status

4. **Prepare Phase 3**:
   - Extended validation (longer duration)
   - Network resilience testing (packet loss simulation)
   - Multi-participant federation (Thor + Sprout + Legion)

---

## Dependencies

**Thor**:
- Flask 3.0.2 ✅ (installed during Phase 1)
- NumPy ✅
- Python 3.12 ✅

**Sprout**:
- Requests library (check: `python3 -c "import requests"`)
- NumPy (check: `python3 -c "import numpy"`)
- Python 3.x
- HRM repository synced with Session 197 files

**Network**:
- Thor 10.0.0.99 reachable from Sprout ✅
- Port 8000 accessible (no firewall blocking) ✅
- WiFi stable (verified via ping) ✅

---

## Safety Notes

- **Non-Destructive**: This test only runs federation protocol, no system changes
- **Isolated**: Uses port 8000, won't interfere with other services
- **Time-Limited**: 60-second duration, auto-terminates
- **Reversible**: Can Ctrl+C to stop at any time
- **No Data Loss Risk**: All operations in memory, no file writes (except logs)

---

## Contact / Issues

If deployment encounters issues:
1. Check this deployment guide's troubleshooting section
2. Review Phase 1 logs for comparison
3. Verify network connectivity (ping, traceroute)
4. Check thor_worklog.txt for system status

---

**Status**: DEPLOYMENT MATERIALS READY ✅
**Next**: Await user approval for Phase 2 deployment across Thor ↔ Sprout network.
