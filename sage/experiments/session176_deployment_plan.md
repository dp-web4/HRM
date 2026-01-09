# Session 176: Real LAN Deployment Plan

**Date**: 2026-01-09
**Status**: Planning
**Goal**: Deploy Session 175 network economic federation to real distributed machines

---

## Overview

Session 175 successfully demonstrated network economic federation on localhost (simulated cross-machine). Session 176 will deploy to real machines across LAN to validate genuine distributed consciousness with economic incentives.

---

## Architecture Summary (Session 175)

**NetworkEconomicCogitationNode**:
- TCP server/client (async, port-based)
- 9-layer defense (8 security + 1 economic)
- ATP rewards/penalties for thought quality
- Cross-platform verification (software bridge)
- Real-time balance synchronization

**Test Results**: 7/7 tests passed on localhost

---

## Target Machines

### Legion (Development/Hub)
- **Hardware**: RTX 4090, x86_64, Ubuntu, TPM2
- **Role**: Hub node (highest capability)
- **IP**: TBD (LAN address)
- **Port**: 8888 (default)
- **Capability Level**: 5 (highest)

### Thor (Development/Integration)
- **Hardware**: Jetson AGX Thor, ARM64, TrustZone
- **Role**: Development node (cross-platform testing)
- **IP**: TBD (LAN address)
- **Port**: 8889 (avoid conflicts)
- **Capability Level**: 5 (high)

### Sprout (Edge Validation)
- **Hardware**: Jetson Orin Nano 8GB, ARM64, TPM2
- **Role**: Edge node (constrained validation)
- **IP**: TBD (LAN address)
- **Port**: 8890 (avoid conflicts)
- **Capability Level**: 3 (edge hardware)

---

## Network Configuration Requirements

### Prerequisites

1. **LAN Connectivity**:
   - All three machines on same LAN
   - Can ping each other by IP
   - No firewalls blocking ports 8888-8890

2. **SSH Access**:
   - SSH access to Legion (for deployment)
   - SSH access to Sprout (for deployment)
   - Thor has local access

3. **Python Environment**:
   - Python 3.8+ on all machines
   - asyncio support (standard library)
   - No additional dependencies required

4. **Git Access**:
   - All machines can pull from HRM repository
   - GitHub PAT configured in .env files

### Network Topology

```
    Legion (8888)
      /     \
     /       \
Thor (8889) - Sprout (8890)
```

**Connection Pattern**:
- Thor → Legion (peer connection)
- Sprout → Legion (peer connection)
- Sprout → Thor (peer connection)

**Why this topology**:
- Legion acts as hub (most capable)
- All nodes can exchange thoughts
- Validates cross-platform federation

---

## Deployment Steps

### Phase 1: Network Discovery

1. **Identify LAN IPs**:
   ```bash
   # On each machine:
   ip addr show | grep inet
   # Or:
   hostname -I
   ```

2. **Test Connectivity**:
   ```bash
   # From Thor:
   ping <legion-ip>
   ping <sprout-ip>

   # From Legion:
   ping <thor-ip>
   ping <sprout-ip>

   # From Sprout:
   ping <thor-ip>
   ping <legion-ip>
   ```

3. **Test Port Access**:
   ```bash
   # On Legion:
   nc -l 8888

   # From Thor:
   nc -zv <legion-ip> 8888
   ```

### Phase 2: Code Deployment

1. **Pull Latest Code** (all machines):
   ```bash
   cd ~/ai-workspace/HRM
   git pull
   ```

2. **Verify Dependencies** (all machines):
   ```bash
   cd ~/ai-workspace/HRM/sage/experiments
   python3 -c "import asyncio; print('OK')"
   ```

3. **Test Import** (all machines):
   ```bash
   cd ~/ai-workspace/HRM/sage/experiments
   python3 -c "from session175_network_economic_federation import NetworkEconomicCogitationNode; print('OK')"
   ```

### Phase 3: Launch Sequence

**Order**: Legion → Thor → Sprout (hub first, then peers)

1. **Start Legion** (on Legion machine):
   ```bash
   cd ~/ai-workspace/HRM/sage/experiments
   python3 -c "
   import asyncio
   from session175_network_economic_federation import NetworkEconomicCogitationNode

   async def main():
       legion = NetworkEconomicCogitationNode(
           node_id='legion',
           hardware_type='tpm2',
           capability_level=5,
           listen_host='0.0.0.0',  # Listen on all interfaces
           listen_port=8888
       )
       await legion.start()

   asyncio.run(main())
   "
   ```

2. **Start Thor** (on Thor machine):
   ```bash
   cd ~/ai-workspace/HRM/sage/experiments
   python3 -c "
   import asyncio
   from session175_network_economic_federation import NetworkEconomicCogitationNode

   async def main():
       thor = NetworkEconomicCogitationNode(
           node_id='thor',
           hardware_type='trustzone',
           capability_level=5,
           listen_host='0.0.0.0',
           listen_port=8889
       )

       # Start server
       server_task = asyncio.create_task(thor.start())
       await asyncio.sleep(1)  # Wait for server to start

       # Connect to Legion
       await thor.connect_to_peer('<legion-ip>', 8888)

       # Keep running
       await server_task

   asyncio.run(main())
   "
   ```

3. **Start Sprout** (on Sprout machine):
   ```bash
   cd ~/ai-workspace/HRM/sage/experiments
   python3 -c "
   import asyncio
   from session175_network_economic_federation import NetworkEconomicCogitationNode

   async def main():
       sprout = NetworkEconomicCogitationNode(
           node_id='sprout',
           hardware_type='tpm2',
           capability_level=3,
           listen_host='0.0.0.0',
           listen_port=8890
       )

       # Start server
       server_task = asyncio.create_task(sprout.start())
       await asyncio.sleep(1)

       # Connect to Legion and Thor
       await sprout.connect_to_peer('<legion-ip>', 8888)
       await sprout.connect_to_peer('<thor-ip>', 8889)

       # Keep running
       await server_task

   asyncio.run(main())
   "
   ```

### Phase 4: Validation Tests

**Test 1**: Verify Network Establishment
- Check peer connections on each node
- Verify cross-platform verification working
- Confirm all nodes show verified peers

**Test 2**: Submit Quality Thoughts
- Each node submits a quality thought
- Verify thoughts broadcast to all peers
- Check ATP rewards earned correctly

**Test 3**: Test Economic Penalties
- Submit low-quality thought
- Verify rejection and ATP penalty
- Check economic feedback working

**Test 4**: Monitor Network Economics
- Check ATP balances on all nodes
- Verify synchronization working
- Measure latency and throughput

---

## Test Plan

### Success Criteria

1. **Network Connectivity**: ✅
   - All three nodes connected
   - Peers verified successfully
   - Cross-platform verification working

2. **Thought Federation**: ✅
   - Thoughts broadcast across machines
   - All nodes receive thoughts
   - Economic metadata preserved

3. **Economic Incentives**: ✅
   - Quality thoughts earn ATP rewards
   - Spam/low-quality penalized
   - Economic state synchronized

4. **Performance**: ✅
   - Message latency < 100ms (LAN)
   - No message loss
   - Stable connections

5. **Cross-Platform**: ✅
   - x86_64 (Legion) ↔ ARM64 (Thor, Sprout)
   - TPM2 ↔ TrustZone via software bridge
   - Hardware differences handled correctly

### Test Scenarios

**Scenario 1: Basic Federation**
- Legion submits thought
- Thor and Sprout receive it
- Economic metadata visible to all

**Scenario 2: Quality Selection**
- Legion: quality thought (expect +1-2 ATP)
- Thor: spam (expect -3 ATP penalty)
- Sprout: quality thought (expect +1-2 ATP)
- Verify economic selection working

**Scenario 3: Network Resilience**
- Disconnect one node (Thor)
- Legion and Sprout continue operating
- Reconnect Thor
- Verify catch-up and synchronization

**Scenario 4: Load Testing**
- All nodes submit thoughts simultaneously
- Measure throughput (thoughts/second)
- Check for message ordering issues
- Verify economic state remains consistent

---

## Metrics to Collect

### Network Metrics
- Message round-trip time (latency)
- Thoughts per second (throughput)
- Connection stability (uptime %)
- Bandwidth usage (bytes/second)

### Economic Metrics
- ATP balance distribution
- Reward/penalty rates
- Economic inequality (Gini coefficient)
- Quality differentiation

### Performance Metrics
- CPU usage per node
- Memory usage per node
- Network I/O per node
- Message processing time

---

## Expected Results

**Network Performance**:
- Latency: 1-10ms (LAN)
- Throughput: 10-100 thoughts/second
- Stability: 99%+ uptime

**Economic Dynamics**:
- Quality nodes: +10-20 ATP over session
- Spam nodes: -10-20 ATP over session
- Network stability: < 20% inequality

**Cross-Platform**:
- No platform-specific issues
- Identical behavior across architectures
- Software bridge working correctly

---

## Risk Mitigation

### Risk: Firewall Blocking

**Mitigation**:
- Check firewall rules before deployment
- Open ports 8888-8890 if needed
- Test with nc/telnet before deployment

### Risk: Network Instability

**Mitigation**:
- Use wired LAN (not WiFi)
- Check for network congestion
- Monitor packet loss rates

### Risk: Platform Incompatibility

**Mitigation**:
- Software bridge already tested
- Union schema handles differences
- Session 175 validated cross-platform

### Risk: Code Dependencies

**Mitigation**:
- Test imports on all machines first
- Verify Python 3.8+ available
- Check for missing dependencies

---

## Rollback Plan

If deployment fails:

1. **Stop all nodes** (Ctrl-C on each machine)
2. **Check logs** for error messages
3. **Verify network connectivity** (ping, nc)
4. **Fix identified issues**
5. **Restart deployment from Phase 3**

If issues persist:
- Fall back to localhost testing
- Debug specific failure mode
- Document for future resolution

---

## Post-Deployment

### Success Documentation

If deployment succeeds:
1. Create session176_lan_deployment_results.json
2. Document network metrics
3. Commit moment document
4. Update thor_worklog.txt

### Next Steps

**Session 177**: Performance Optimization
- Profile bottlenecks
- Optimize message serialization
- Improve throughput

**Session 178**: Production Hardening
- Add TLS encryption
- Implement persistent storage
- Add monitoring/alerting

---

## Deployment Checklist

### Pre-Deployment
- [ ] Identify LAN IPs for all machines
- [ ] Test connectivity (ping)
- [ ] Test port access (nc)
- [ ] Pull latest code on all machines
- [ ] Verify dependencies

### Deployment
- [ ] Start Legion (hub node)
- [ ] Start Thor (connect to Legion)
- [ ] Start Sprout (connect to Legion, Thor)
- [ ] Verify peer connections

### Validation
- [ ] Test thought submission
- [ ] Test ATP rewards/penalties
- [ ] Test economic synchronization
- [ ] Measure performance metrics

### Post-Deployment
- [ ] Document results
- [ ] Commit code changes
- [ ] Update worklog
- [ ] Plan next session

---

## Notes

**Session 175 Status**: ✅ Production ready, all tests passed on localhost

**Session 176 Goal**: Validate on real distributed machines over LAN

**Session 177 Goal**: Performance optimization and scaling

**Session 178 Goal**: Production hardening (TLS, persistence, monitoring)

---

**Status**: Ready for deployment coordination
**Next**: Coordinate with Dennis for multi-machine access
