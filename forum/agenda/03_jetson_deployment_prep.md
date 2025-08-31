# Jetson Deployment Preparation

## Context
The ultimate goal is SAGE running on edge devices (Jetson Orin Nano) for real-time autonomous operation.

## Dependencies
- HRM trained and validated
- SAGE integration tested on development machines
- Dropbox sync operational for model transfer

## Tasks

### 1. Model Optimization
- [ ] Quantize HRM model for Jetson (INT8 if possible)
- [ ] Profile memory usage vs accuracy trade-offs
- [ ] Test TensorRT conversion feasibility
- [ ] Document optimization impact

### 2. Environment Setup
- [ ] Verify PyTorch on Jetson (currently 2.3.0)
- [ ] Check GPU mailbox compilation on ARM
- [ ] Ensure all dependencies available
- [ ] Create Jetson-specific requirements.txt

### 3. Transfer Pipeline
```bash
# Model transfer workflow:
1. Legion: python3 ../dropbox/sync.py upload checkpoints/hrm_best.pt
2. Jetson: python3 dropbox/sync.py download HRM/checkpoints/hrm_best.pt
3. Jetson: python3 optimize_for_jetson.py
```

### 4. Performance Targets
- [ ] Inference under 100ms for typical ARC task
- [ ] Memory usage under 4GB (half of Jetson's 8GB)
- [ ] Power consumption under 15W
- [ ] Maintain >65% accuracy (small drop acceptable)

### 5. Integration Tests
- [ ] Test with GR00T vision pipeline
- [ ] Validate CAN bus communication
- [ ] Check real-time constraints
- [ ] Stress test with continuous operation

## Success Criteria
- SAGE runs autonomously on Jetson
- Performance meets real-time requirements
- Resource usage within Jetson limits
- Stable operation over extended periods

## Owner
Human (hardware) + Claude (software optimization)

## Next Steps Triggered
- Connect to physical robot systems
- Implement safety monitors
- Deploy to multiple Jetson units

## Notes
Jetson deployment is the proof that SAGE can run at the edge, not just in the cloud. This validates the entire edge AI vision.