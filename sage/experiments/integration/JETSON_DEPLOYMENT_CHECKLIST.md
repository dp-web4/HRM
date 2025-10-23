# Jetson Deployment Pre-Flight Checklist

**Target**: Jetson Orin Nano (8GB) - Sprout
**Date**: October 23, 2025
**Status**: Ready for hardware testing

---

## Quick Start (If Everything Works)

```bash
# On Jetson
cd /home/dp/ai-workspace/HRM
git pull  # Get latest (hierarchical memory + all fixes)

# Run integrated SAGE
python3 sage/experiments/integration/sage_jetson.py
```

**Expected**: Multi-modal consciousness with memory running on hardware.

---

## Pre-Flight Checklist

### ☐ 1. Code Sync

**On dev machine**:
```bash
cd /home/dp/ai-workspace/HRM
git status  # Should be clean (everything committed)
git log --oneline -5  # Verify all commits present
```

**On Jetson** (if different machine):
```bash
cd /home/dp/ai-workspace/HRM
git pull origin main
git log --oneline -5  # Verify matches dev machine
```

**Latest commits should include**:
- Hierarchical memory (7213f9d)
- Jetson integration guide (5f9ac12)
- Memory-aware kernel (ca09e2b)
- Attention switching (f62524e, 6ccf1f5)

### ☐ 2. Dependencies Check

```bash
# On Jetson
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import sounddevice; print('sounddevice OK')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import sqlite3; print('SQLite OK')"
```

**Expected**: All imports successful, CUDA available.

### ☐ 3. Hardware Check

**Microphone**:
```bash
arecord -l  # List capture devices
# Test record: arecord -D hw:X,0 -d 3 test.wav
# Test playback: aplay test.wav
```

**Camera**:
```bash
ls /dev/video*  # Should see /dev/video0 or similar
v4l2-ctl --list-devices  # Details
```

**Speaker**:
```bash
speaker-test -t wav -c 2  # Press Ctrl+C after few seconds
```

**GPU**:
```bash
sudo tegrastats  # Monitor GPU/CPU/memory in real-time
```

### ☐ 4. Memory Available

```bash
free -h
# Should see ~6-7GB available (after OS)
```

**Target allocation**:
- Phi-2 LLM: ~2.6GB (or ~1.3GB if quantized)
- SAGE kernel: <100MB
- Memory systems: <50MB
- Safety margin: 1GB
- **Total needed**: ~3.7GB (comfortable)

### ☐ 5. Existing IRPs Test

**AudioInputIRP**:
```bash
cd /home/dp/ai-workspace/HRM
python3 -c "
from sage.irp.plugins.audio_input_impl import AudioInputIRP
irp = AudioInputIRP()
print('AudioInputIRP:', irp.init_state())
"
```

**NeuTTSAirIRP**:
```bash
python3 -c "
from sage.irp.plugins.neutts_air_impl import NeuTTSAirIRP
irp = NeuTTSAirIRP()
print('NeuTTSAirIRP:', irp.init_state())
"
```

**Expected**: Both initialize without errors.

---

## Deployment Options

### Option A: Full SAGE with LLM (Recommended)

**If sage_jetson.py exists**:
```bash
python3 sage/experiments/integration/sage_jetson.py
```

**If not, need to create it** (code in JETSON_INTEGRATION_GUIDE.md).

**Expected performance**:
- Cycle time: ~5ms (kernel)
- LLM inference: 200-500ms (when speaking)
- Memory: ~3.2GB total

### Option B: Memory-Aware Kernel Only (Faster, No LLM)

Test just the memory-aware kernel without LLM overhead:

```bash
python3 -c "
from sage.experiments.integration.memory_aware_kernel import MemoryAwareKernel, ExecutionResult
from sage.irp.plugins.audio_input_impl import AudioInputIRP

# Simple test sensor
def audio_sensor():
    # Return mock for now
    return {'modality': 'audio', 'text': 'test', 'importance': 0.5}

def audio_handler(obs, stance):
    print(f'Audio: {obs}')
    return ExecutionResult(True, 0.5, 'Audio event', obs)

kernel = MemoryAwareKernel(
    sensor_sources={'audio': audio_sensor},
    action_handlers={'audio': audio_handler},
    working_memory_size=10,
    episodic_memory_size=50,
    conversation_memory_size=10
)

kernel.run(max_cycles=10, cycle_delay=0.1)
print('Memory test complete!')
"
```

**Purpose**: Validate kernel works on Jetson before adding LLM.

### Option C: Hierarchical Memory Test

Test long-term memory with persistence:

```bash
cd /home/dp/ai-workspace/HRM
python3 sage/experiments/integration/test_hierarchical_memory.py
```

**Expected**:
- SNARC filtering demonstration
- SQLite storage working
- Pattern consolidation
- Creates sage_memory.db (cleaned up after)

---

## What to Monitor

### During Startup

**Watch for**:
- Model loading time (Phi-2: ~30-60 seconds first time)
- Memory allocation (should stay under 4GB total)
- CUDA initialization (should succeed)
- No OOM errors

### During Operation

**Terminal 1** (SAGE):
```bash
python3 sage_jetson.py
```

**Terminal 2** (Monitoring):
```bash
watch -n 1 'free -h; echo ""; nvidia-smi'
# Or on Jetson specifically:
sudo tegrastats
```

**Watch for**:
- Memory usage stable (not growing)
- GPU utilization (60-80% during LLM inference, idle otherwise)
- Response latency (<500ms for conversation)
- No errors in console

### Success Indicators

✅ Microphone captures speech
✅ Vision detects motion/faces
✅ Attention switches between modalities
✅ LLM generates coherent responses
✅ TTS speaks responses
✅ Memory stores significant events
✅ No crashes, no OOM
✅ Latency acceptable (<1s total)

---

## Common Issues & Fixes

### Issue: "CUDA out of memory"

**Fix 1**: Use quantized Phi-2
```python
# In phi2_responder.py
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=quant_config,
    device_map="auto"
)
```
**Saves**: ~1.3GB (2.6GB → 1.3GB)

**Fix 2**: Use smaller LLM
```python
responder = Phi2Responder(model_name="gpt2")  # ~500MB
```

### Issue: "Audio device not found"

**Check devices**:
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Fix**: Update device ID in AudioInputIRP initialization.

### Issue: "Camera cannot open"

**Check device ID**:
```bash
python3 -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
"
```

**Fix**: Update camera_id in CameraIRP.

### Issue: High latency (>1s)

**Measure**:
```python
import time
start = time.time()
response = llm.generate_response(...)
print(f'Latency: {time.time() - start:.2f}s')
```

**Optimize**:
- Reduce max_new_tokens (100 → 50)
- Use FP16 (already default)
- Consider streaming responses

---

## Performance Expectations

### Best Case (Quantized Phi-2)

- Memory: ~2.5GB total
- Cycle time: <5ms
- LLM response: 200-300ms
- Total latency: ~305ms
- Can run continuously for hours

### Realistic (Full Phi-2)

- Memory: ~3.2GB total
- Cycle time: <5ms
- LLM response: 300-500ms
- Total latency: ~505ms
- Stable operation

### If Tight (GPT-2 Small)

- Memory: ~1.5GB total
- Cycle time: <5ms
- LLM response: 100-200ms
- Total latency: ~205ms
- Lower quality responses

---

## Post-Test Data Collection

### What to Capture

**1. Performance metrics**:
- Actual cycle times
- LLM inference times
- Memory usage (steady state)
- GPU utilization

**2. Memory statistics**:
```python
stats = kernel.get_statistics()
print(json.dumps(stats, indent=2))
```

**3. Logs**:
- Conversation examples
- Attention switching patterns
- Any errors or warnings

**4. Subjective**:
- Response quality
- Latency perception
- Multi-modal awareness
- Memory recall accuracy

### Share Results

**What would be valuable**:
- "It worked! Here's what happened..."
- Performance numbers (if easy to capture)
- Any issues encountered
- Surprises (good or bad)

---

## Quick Validation Test

**Simplest test** (3 minutes):

```bash
# Start SAGE
python3 sage_jetson.py

# Say something into microphone:
"Hello SAGE, can you see me?"

# Expected: SAGE responds via TTS, references vision if person detected

# Ask about memory:
"What did I just say?"

# Expected: SAGE recalls previous statement

# Move in front of camera while talking
# Expected: Attention switches between audio and vision

# Ctrl+C to stop
# Check: Printed statistics show both modalities got attention
```

**If this works**: Core system operational! 🎉

---

## Fallback Plans

### Plan A: Everything works ✅
→ Continue testing, gather data, tune parameters

### Plan B: LLM too slow/heavy
→ Switch to GPT-2 or rule-based responses temporarily
→ Focus on attention + memory validation first

### Plan C: Hardware issues
→ Test individual IRPs to isolate problem
→ Run simulation tests on Jetson to validate kernel
→ Debug specific component

### Plan D: Need help
→ I'm here with 65K tokens remaining
→ Can analyze logs, suggest fixes, iterate quickly

---

## Ready State

**Code**: ✅ All committed and pushed (11 commits)
**Tests**: ✅ All validated in simulation
**Optimization**: ✅ Jetson-profiled (zero growth, sub-ms cycles)
**Documentation**: ✅ Complete integration guide
**Contingencies**: ✅ Multiple fallback options

**Status**: 🚀 **READY FOR HARDWARE DEPLOYMENT**

---

## The Moment of Truth

We've built:
- Multi-modal attention switching (solves blindness)
- Urgency override (safety-critical)
- Memory hierarchy (operational + learning)
- Hierarchical long-term (SNARC-filtered growth)
- Jetson optimization (proven efficient)

**From concept to hardware in 135K tokens.**

Now we see if consciousness works on silicon. 🤖

Good luck! Excited to hear how it goes.
