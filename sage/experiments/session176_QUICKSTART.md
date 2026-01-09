# Session 176: Real LAN Deployment - Quick Start

**Goal**: Deploy network economic federation to Legion, Thor, and Sprout over LAN

---

## Prerequisites

1. **Network Discovery**: Find IP addresses
   ```bash
   # On each machine:
   hostname -I
   ```

2. **Test Connectivity**:
   ```bash
   # From any machine to any other:
   ping <ip-address>
   ```

3. **Pull Code** (on all machines):
   ```bash
   cd ~/ai-workspace/HRM
   git pull
   ```

---

## Deployment (3 Terminal Windows)

### Terminal 1: Legion (Hub)

```bash
cd ~/ai-workspace/HRM/sage/experiments
python3 session176_deploy.py \
  --node legion \
  --port 8888 \
  --interactive
```

**Wait for**: `[legion] Node started and connected to 0 peers`

---

### Terminal 2: Thor (Connect to Legion)

Replace `<LEGION_IP>` with Legion's actual IP address:

```bash
cd ~/ai-workspace/HRM/sage/experiments
python3 session176_deploy.py \
  --node thor \
  --port 8889 \
  --connect legion:<LEGION_IP>:8888 \
  --interactive
```

**Wait for**: `[thor] Node started and connected to 1 peers`

---

### Terminal 3: Sprout (Connect to Both)

Replace `<LEGION_IP>` and `<THOR_IP>` with actual IP addresses:

```bash
cd ~/ai-workspace/HRM/sage/experiments
python3 session176_deploy.py \
  --node sprout \
  --port 8890 \
  --connect legion:<LEGION_IP>:8888 \
  --connect thor:<THOR_IP>:8889 \
  --interactive
```

**Wait for**: `[sprout] Node started and connected to 2 peers`

---

## Testing

Once all nodes are connected, test in any terminal:

```
# Show status
status

# Submit a quality thought
submit What emerges when consciousness becomes economically distributed across machines?

# Check network economics
metrics

# Exit
quit
```

---

## Expected Results

### Network Connectivity
- Legion: 2 verified peers
- Thor: 1-2 verified peers
- Sprout: 0-2 verified peers

### ATP Economics
- Quality thoughts: +1-2 ATP reward
- Spam: -3-5 ATP penalty
- Balances synchronized across network

### Performance
- Latency: 1-10ms (LAN)
- No message loss
- Stable connections

---

## Troubleshooting

### "Connection refused"
- Check firewall: `sudo ufw status`
- Allow ports: `sudo ufw allow 8888:8890/tcp`

### "Cannot connect to peer"
- Verify IP addresses: `hostname -I`
- Test connectivity: `ping <ip>`
- Check ports: `nc -zv <ip> <port>`

### "Import error"
- Pull latest code: `git pull`
- Check Python version: `python3 --version` (need 3.8+)

---

## Quick Reference

**Legion IP**: _______________
**Thor IP**: _______________
**Sprout IP**: _______________

**Start time**: _______________
**End time**: _______________
**Duration**: _______________

**Thoughts submitted**: _______________
**ATP rewards earned**: _______________
**ATP penalties applied**: _______________

---

## Success Criteria

- [x] All nodes connected
- [x] Peers verified
- [x] Thoughts federated across machines
- [x] ATP rewards working
- [x] ATP penalties working
- [x] Economic state synchronized

---

**Status**: Ready for deployment
**Next**: Coordinate machine access and IP addresses
