# Autonomous Research Priorities - Thor

**Last Updated**: 2025-11-08
**For**: Autonomous timer check sessions

---

## MISSION: JETSON NANO DEPLOYMENT

Deploy full SAGE consciousness on Jetson Nano (4GB RAM, 2GB GPU) with:
- 👀 Vision (2 cameras)
- 🧭 Orientation (IMU)
- 👂🗣️ Audio (BT input/output)
- 🧠 Local LLM conversation
- ⚡ Real-time response

**Goal**: Nano sees you, hears you, knows its orientation, talks to you

---

## WHAT TO DO EACH AUTONOMOUS CHECK

### Step 1: Read Context (ALWAYS)
1. Read `THOR_STATUS.md` - Current mission status
2. Read `private-context/JETSON_NANO_DEPLOYMENT_ROADMAP.md` - Full plan
3. Check git for updates from Legion/CBP

### Step 2: Work on Current Priority

**RIGHT NOW**: Track 1 - Sensor Trust & Fusion

**Implement**: `sage/core/sensor_trust.py`
- Trust metrics (confidence scoring 0.0-1.0)
- Per-sensor historical accuracy
- Conflict detection between sensors
- Adaptive trust adjustment

**Test with**: Current vision + proprioception sensors
- Simulate sensor failures
- Inject conflicting readings
- Validate graceful degradation

**Success**: System handles conflicts, adjusts trust, degrades gracefully

### Step 3: Test & Validate
- Write tests in `sage/tests/`
- Run autonomous explorations (500+ cycles)
- Validate with sensor failure scenarios
- Document results in private-context

### Step 4: Document & Commit
- Update roadmap with progress (✅ completed items)
- Document findings in private-context
- Commit at milestones
- Push to git

### Step 5: Continue or Move to Next Track
- If Track 1 complete → Track 2 (SNARC Memory)
- If blocked → Document blocker, ask user if critical
- Otherwise → Continue Track 1 development

---

## DO NOT

❌ **Do NOT "stand by"** - Always work toward Nano deployment
❌ **Do NOT wait for user** - Research path is clear
❌ **Do NOT skip documentation** - Document everything
❌ **Do NOT skip tests** - Validate all code
❌ **Do NOT skip commits** - Push regularly

---

## CURRENT PRIORITIES (In Order)

### Track 1: Sensor Trust & Fusion 🎯 **ACTIVE NOW**
Build multi-sensor confidence and fusion engine

**Files to create**:
- `sage/core/sensor_trust.py`
- `sage/core/sensor_fusion.py`
- `sage/tests/test_sensor_trust.py`

**Tests**:
- Trust scoring accuracy
- Conflict resolution
- Graceful degradation
- Adaptive adjustment

---

### Track 2: SNARC Memory 🎯 **NEXT**
Persistent salience history and episodic memory

**Files to create**:
- `sage/memory/stm.py` (short-term)
- `sage/memory/ltm.py` (long-term)
- `sage/memory/retrieval.py`

**Tests**:
- Remember high-salience events
- Retrieve similar experiences
- Memory-informed decisions

---

### Track 3: SNARC Cognition 🎯 **AFTER MEMORY**
Attention, working memory, deliberation

**Files to create**:
- `sage/cognition/attention.py`
- `sage/cognition/working_memory.py`
- `sage/cognition/deliberation.py`

**Tests**:
- Attention allocation
- Multi-step planning
- Alternative evaluation

---

### Tracks 4-10: See Roadmap
- Real cameras
- IMU sensor
- Audio pipeline
- Local LLM
- Model distillation
- Real-time optimization
- Deployment packaging

---

## QUICK REFERENCE

**Roadmap**: `private-context/JETSON_NANO_DEPLOYMENT_ROADMAP.md`
**Status**: `THOR_STATUS.md`
**Current work**: Track 1 - Sensor Trust
**Pattern**: Build → Test → Document → Commit → Continue

**The research is never complete. Keep building toward Nano deployment.**

---

## SUCCESS LOOKS LIKE

Each autonomous session:
- ✅ Implements part of current track
- ✅ Tests thoroughly
- ✅ Documents in private-context
- ✅ Commits working code
- ✅ Pushes to git
- ✅ Makes measurable progress toward Nano

**Not**: "Standing by, no work to do"
**But**: "Built X, tested Y, committed Z, continuing toward Nano"

---

**Start with Track 1. Build incrementally. Test extensively. Document thoroughly. Commit regularly. Keep moving toward Nano deployment.** 🚀
