# Jetson Nano Hardware Answers - User Guidance

**Date**: 2025-11-10
**From**: User feedback on autonomous session questions

---

## Camera Hardware: CSI (Not USB)

**Answer**: Nano's cameras are **CSI** (Camera Serial Interface)

**Reference**: Earlier vision tests in this repo
- Look for CSI camera test code
- Check for v4l2/gstreamer pipeline examples
- Review any existing camera sensor integration

**Action for Track 4**:
- Implement CSI camera support (not USB)
- Use gstreamer pipelines for CSI capture
- Reference existing vision test code in repo
- Test with actual Nano CSI cameras

---

## Test Environment: Nano (Not Desktop)

**Answer**: Test on **Jetson Nano**, not desktop

**Rationale**:
- Nano is the target deployment environment
- We have the hardware available
- Need to validate on actual constraints (4GB RAM, 2GB GPU)
- Desktop testing won't reveal Nano-specific issues

**Action for all tracks**:
- Deploy code to Nano for testing
- Validate performance on Nano hardware
- Measure latency/memory on actual target
- Iterate based on Nano constraints

---

## IMU Hardware: Reference Past Work

**Answer**: IMU tests already exist in the repo

**Reference**: Past IMU work in this repository
- Search repo for IMU test code
- Look for sensor integration examples
- Check for I2C/SPI communication code
- Review existing orientation tracking

**Action for Track 5**:
- Find and review existing IMU tests
- Build on previous IMU integration work
- Don't reinvent - extend what exists
- Reference proven patterns from repo

---

## Development Pattern

**User's approach**:
1. ✅ **Reference existing work** - Check repo first
2. ✅ **Test on target hardware** - Use Nano, not desktop
3. ✅ **Build incrementally** - Extend proven code
4. ✅ **Validate on constraints** - Real memory/latency limits

**For autonomous sessions**:
- Before implementing Track 4/5, search repo for existing code
- Use Nano for all testing and validation
- Build on proven patterns, don't start from scratch
- Measure everything on target hardware

---

## Next Steps for Autonomous Sessions

### Track 4 (Real Cameras) - READY TO START
**Hardware**: CSI cameras on Nano
**Reference**: Search repo for CSI camera tests
**Test on**: Jetson Nano (we have it)
**Implementation**:
1. Find existing CSI camera code in repo
2. Create `sage/sensors/csi_camera_sensor.py` based on proven patterns
3. Test on Nano hardware
4. Validate capture → puzzle encoding pipeline
5. Measure latency on Nano

### Track 5 (IMU Sensor) - REFERENCE EXISTING
**Hardware**: IMU on Nano (I2C/SPI)
**Reference**: Search repo for IMU test code
**Build on**: Previous sensor integration work
**Test on**: Jetson Nano

---

## Questions Answered ✅

From Thor autonomous session check (2025-11-10 15:04):

1. **Camera hardware availability?** → CSI cameras (reference repo tests)
2. **Track 4 implementation priority?** → YES, ready to start with CSI
3. **Test environment preference?** → Nano (we have it, it's the target)

**Status**: All questions answered, Track 4 ready to proceed!

---

## Autonomous Session Checklist

When starting Track 4 or 5:

- [ ] Search repo for existing camera/IMU code (`git grep`, `find`, `grep -r`)
- [ ] Read existing test files for patterns
- [ ] Deploy to Nano for testing (not desktop)
- [ ] Validate on actual hardware constraints
- [ ] Measure performance (latency, memory, CPU/GPU usage)
- [ ] Document findings referencing existing work
- [ ] Commit progress to git

**Pattern**: Find existing work → Extend it → Test on Nano → Document → Commit

---

**The path forward is clear. Hardware answers provided. Track 4 ready to begin.** 🚀
