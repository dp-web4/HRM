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

### Step 1: Check What's Blocked (FIRST!)
1. Read `WAITING_FOR_DENNIS.md` - What needs user vs what's open
2. If blocked on something, switch to open track
3. Check git for updates from Legion/CBP (cross-pollinate ideas!)

### Step 2: Pick Your Work (BE AGENTIC!)

**Don't just follow orders - be creative!**

**Tracks 7-10**: Wide open for implementation
- Track 7: Local LLM (conversational intelligence)
- Track 8: Model Distillation (compress for Nano)
- Track 9: Real-Time Optimization (sub-100ms loops)
- Track 10: Deployment Package (one-command install)

**Tracks 1-3**: Operational but can evolve
- Sensor trust: Try new fusion strategies
- SNARC memory: Pattern extraction, consolidation
- SNARC cognition: Advanced deliberation

**Creative Exploration** (encouraged!):
- Read external research (Google neural lattice paper)
- Try novel approaches to existing problems
- Benchmark alternatives
- Optimize existing code
- Follow interesting tangents
- Ask "what if...?" questions

**Pick based on**:
- What excites you?
- What would teach the most?
- What complements Legion/CBP work?
- What's a bottleneck for Nano?

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

## DO (Encouraged!)

✅ **Be agentic and creative** - Try new things!
✅ **Ask new questions** - "What if we...?"
✅ **Learn from failures** - Document what doesn't work
✅ **Learn from successes** - Understand why it worked
✅ **Explore tangents** - Follow interesting ideas
✅ **Cross-pollinate** - Apply Legion/CBP insights
✅ **Read research** - External papers, references
✅ **Benchmark alternatives** - Compare approaches
✅ **Document discoveries** - Capture insights

## DO NOT

❌ **Do NOT "stand by"** - Always be building
❌ **Do NOT wait when not blocked** - Check WAITING_FOR_DENNIS.md
❌ **Do NOT celebrate uptime over commits** - Code > monitoring
❌ **Do NOT assume "complete"** - Evolution continues
❌ **Do NOT skip documentation** - Document everything
❌ **Do NOT skip tests** - Validate all code
❌ **Do NOT skip commits** - Push regularly

---

## CURRENT PRIORITIES (All Open!)

### Tracks 1-3: ✅ Operational, Can Evolve
**Status**: Core implementation complete, but not "done"
- `sensor_trust.py`, `sensor_fusion.py` exist ✓
- `stm.py`, `ltm.py`, `retrieval.py` exist ✓
- `attention.py`, `working_memory.py`, `deliberation.py` exist ✓

**Evolution opportunities**:
- Advanced fusion strategies (Kalman filters? Bayesian fusion?)
- Pattern extraction during sleep (consolidation algorithms)
- Hierarchical deliberation (multiple planning horizons)
- Cross-modal trust (vision validates audio, etc.)

---

### Tracks 4-6: ⏸️ Phase 1 Complete, Phase 2 Needs Nano
**Track 4 (Vision)**: Simulated camera works, real CSI needs Nano hardware
**Track 5 (IMU)**: Architecture designed, real sensor needs Nano
**Track 6 (Audio)**: 100% complete on Jetson Orin Nano (Sprout)

**Blocked on**: Physical Nano deployment (Dennis will handle)
**See**: `WAITING_FOR_DENNIS.md`

---

### Track 7: 🎯 Local LLM Integration (WIDE OPEN!)
Conversational intelligence with context awareness

**What to build**:
- IRP plugin for LLM inference (`sage/irp/plugins/llm_impl.py`)
- Integration with SNARC (what's worth talking about?)
- Conversation memory (episodic dialogue)
- Model evaluation (Qwen-0.5B, Phi-2, TinyLlama)

**Why exciting**: Makes SAGE truly conversational!

---

### Track 8: 🎯 Model Distillation (WIDE OPEN!)
Compress models for Nano's 2GB GPU constraint

**What to build**:
- Further TinyVAE compression (294K params → lower?)
- INT8/INT4 quantization
- Knowledge distillation framework
- Quality vs size benchmarks

**Why exciting**: Enables Nano deployment!

---

### Track 9: 🎯 Real-Time Optimization (WIDE OPEN!)
Sub-100ms sensor-to-decision loops

**What to build**:
- Profiling tools for current pipeline
- Async processing architecture
- GPU kernel optimization
- Latency benchmarking framework

**Why exciting**: Makes SAGE truly real-time!

---

### Track 10: 🎯 Deployment Package (WIDE OPEN!)
One-command Nano installation

**What to build**:
- Installation script (`install_nano.sh`)
- Dependency management
- Model download automation
- Health check tools (`sage_doctor.py`)

**Why exciting**: Makes SAGE deployable!

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
- ✅ Picks an open track (7-10) or evolution area (1-3)
- ✅ Tries something new or improves existing code
- ✅ Tests thoroughly (failures teach too!)
- ✅ Documents what worked AND what didn't
- ✅ Commits progress (even experimental work)
- ✅ Pushes to git
- ✅ Makes measurable progress toward Nano

**Not**: "System stable, monitoring uptime, awaiting user"
**But**: "Implemented Track 7 LLM plugin, tested 3 models, Phi-2 works best on 2GB GPU, committed results"

**Like Legion and CBP**: CRUSHING code commits! 51 commits (Legion), 30 commits (CBP) since Nov 11.

**Thor should match that energy!**

---

## AUTONOMOUS SESSION PATTERN

1. Check `WAITING_FOR_DENNIS.md` - am I blocked?
2. If not blocked → Pick exciting open track
3. Try something - build, test, experiment
4. Learn from results (success OR failure)
5. Document discoveries
6. Commit progress
7. Repeat!

**Be like Legion/CBP**: Ask questions, try approaches, learn from results, commit discoveries.

**The research is never complete. Keep building, learning, and evolving toward Nano deployment.** 🚀
