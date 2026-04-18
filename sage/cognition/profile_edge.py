#!/usr/bin/env python3
"""
Edge profiling for SAGE brain-arch components on Jetson Orin Nano.
Measures: memory footprint, operation timing, scaling behavior.
"""
import gc
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Helpers ──────────────────────────────────────────────────────────

def mem_snapshot():
    """Return current/peak memory in MB."""
    cur, peak = tracemalloc.get_traced_memory()
    return cur / 1e6, peak / 1e6

def timeit(fn, label, iterations=10):
    """Time a function over N iterations, return avg ms."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mn, mx = min(times), max(times)
    print(f"  {label}: avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms  (n={iterations})")
    return avg

# ── Profile Working Memory ──────────────────────────────────────────

def profile_wm():
    print("\n=== Working Memory ===")
    tracemalloc.start()
    from sage.cognition.working_memory import WorkingMemory
    wm = WorkingMemory()

    # Write slots
    timeit(lambda: wm.add_item("goal", {"target": "explore"}, priority=0.8),
           "add_item(goal)")
    timeit(lambda: wm.add_item("intermediate_result", {"obs": [1,2,3]}, priority=0.5),
           "add_item(intermediate_result)")

    # Capacity test - fill to limit and force evictions
    for i in range(20):
        wm.add_item("other", {"data": f"value_{i}"}, priority=0.1 + i * 0.04)

    timeit(lambda: wm.dump(), "dump() full buffer")
    timeit(lambda: wm.snapshot(), "snapshot() alias")

    cur, peak = mem_snapshot()
    print(f"  Memory: current={cur:.3f}MB  peak={peak:.3f}MB")
    tracemalloc.stop()

# ── Profile RPE ──────────────────────────────────────────────────────

def profile_rpe():
    print("\n=== Reward Prediction Error ===")
    gc.collect()
    tracemalloc.start()
    from sage.cognition.rpe.core import RewardPredictionError
    rpe = RewardPredictionError()

    # Single predict+observe cycle
    def single_cycle():
        pred = rpe.predict("state_a", "move_right")
        rpe.observe(pred, pixel_delta=50, level_up=False, died=False)

    timeit(single_cycle, "predict+observe single")

    # Bulk learning - 100 predict+observe cycles
    def bulk_cycles():
        for i in range(100):
            pred = rpe.predict(f"state_{i % 20}", f"action_{i % 5}")
            rpe.observe(pred, pixel_delta=i * 2, level_up=(i % 50 == 0), died=(i % 30 == 0))

    timeit(bulk_cycles, "100 predict+observe cycles", iterations=5)

    timeit(lambda: rpe.dump(), "dump() state")

    cur, peak = mem_snapshot()
    print(f"  Memory: current={cur:.3f}MB  peak={peak:.3f}MB")
    print(f"  Stats: {rpe.get_stats()}")
    tracemalloc.stop()

# ── Profile Cerebellum ───────────────────────────────────────────────

def profile_cerebellum():
    print("\n=== Cerebellum (Habit Compiler) ===")
    gc.collect()
    tracemalloc.start()
    from sage.cognition.cerebellum.core import Cerebellum, StateSignature
    cb = Cerebellum()

    # Compile habits from repeated patterns
    def observe_one():
        sig = StateSignature(domain="game", features={"pos": (5,5), "nearby": "wall,gem"})
        cb.observe(sig, action_sequence=[{"plugin": "move", "args": {"dir": "right"}}],
                   outcome={"success": True, "summary": "moved"})

    timeit(observe_one, "observe() single", iterations=50)

    # Batch compilation
    def batch_observe():
        for i in range(50):
            sig = StateSignature(domain="game", features={"pos": (i%10, i//10), "nearby": "wall"})
            cb.observe(sig, action_sequence=[{"plugin": "move", "args": {"dir": "up"}}],
                       outcome={"success": (i % 3 != 0), "summary": "moved"})

    timeit(batch_observe, "50 observations batch", iterations=5)

    # Lookup
    query_sig = StateSignature(domain="game", features={"pos": (5,5), "nearby": "wall,gem"})
    timeit(lambda: cb.lookup(query_sig), "lookup()")

    cur, peak = mem_snapshot()
    print(f"  Memory: current={cur:.3f}MB  peak={peak:.3f}MB")
    print(f"  Stats: {cb.stats()}")
    tracemalloc.stop()

# ── Profile Episodic Memory ──────────────────────────────────────────

def profile_episodic():
    print("\n=== Episodic Memory (Hippocampal Index) ===")
    gc.collect()
    tracemalloc.start()
    from sage.cognition.episodic.index import EpisodicIndex
    from sage.cognition.episodic.data import Episode, EpisodicCue
    idx = EpisodicIndex(db_path=":memory:")

    # Bind episodes
    def bind_one():
        ep = Episode(
            state_signature={"pos": (5, 5)},
            snarc_scores={"surprise": 0.7, "novelty": 0.5},
            action_taken="explore",
            outcome="success",
            reward=0.7,
            tags=["game", "level1"]
        )
        idx.bind(ep)

    timeit(bind_one, "bind() single episode")

    # Bulk bind
    def bind_bulk():
        for i in range(100):
            ep = Episode(
                state_signature={"pos": (i % 10, i // 10)},
                snarc_scores={"surprise": float(i % 10) / 10, "novelty": 0.3},
                action_taken=f"action_{i % 5}",
                outcome="ok",
                reward=0.3 + (i % 7) * 0.1,
                tags=["bulk", f"batch_{i // 20}"]
            )
            idx.bind(ep)

    timeit(bind_bulk, "100 episode bulk bind", iterations=3)

    # Recall
    cue5 = EpisodicCue(snarc_scores={"surprise": 0.5, "novelty": 0.5})
    timeit(lambda: idx.recall(cue5, k=5), "recall(k=5)")
    timeit(lambda: idx.recall(cue5, k=20), "recall(k=20)")

    cur, peak = mem_snapshot()
    print(f"  Memory: current={cur:.3f}MB  peak={peak:.3f}MB")
    tracemalloc.stop()

# ── Scaling Test ─────────────────────────────────────────────────────

def profile_scaling():
    print("\n=== Scaling: Episodic Memory with N episodes ===")
    from sage.cognition.episodic.index import EpisodicIndex
    from sage.cognition.episodic.data import Episode, EpisodicCue

    for n in [100, 500, 1000, 5000]:
        gc.collect()
        idx = EpisodicIndex(db_path=":memory:")
        for i in range(n):
            ep = Episode(
                state_signature={"pos": (i % 10, i // 10)},
                snarc_scores={"surprise": float(i % 10) / 10},
                action_taken=f"a_{i % 5}",
                outcome="ok",
                reward=0.5,
                tags=["scale"]
            )
            idx.bind(ep)

        cue = EpisodicCue(snarc_scores={"surprise": 0.5})
        t0 = time.perf_counter()
        for _ in range(10):
            idx.recall(cue, k=5)
        elapsed = (time.perf_counter() - t0) * 1000 / 10
        print(f"  N={n:5d}  recall(k=5) avg={elapsed:.2f}ms")

# ── Profile Motor Skills ─────────────────────────────────────────────

def profile_motor_skills():
    print("\n=== Motor Skills (Executor + Registry) ===")
    gc.collect()
    tracemalloc.start()
    from sage.cognition.motor_skills.registry import get_skill, list_skills
    from sage.cognition.motor_skills.executor import execute_skill
    from sage.cognition.motor_skills.types import Observation, SkillInvocation
    import sage.cognition.motor_skills.skills  # triggers registration

    # Registry lookup
    timeit(lambda: get_skill("navigate_to"), "registry lookup")
    timeit(lambda: list_skills(), "registry list_skills()")

    # navigate_to step function
    skill = get_skill("navigate_to")
    obs = Observation(position=(3, 3))
    params = {"x": 7, "y": 7}
    timeit(lambda: skill.step(obs, params), "navigate_to.step()")
    timeit(lambda: skill.halt_condition(obs, params), "navigate_to.halt_condition()")
    timeit(lambda: skill.stuck_condition(obs, params, [obs, obs, obs]), "navigate_to.stuck_condition()")

    # Full executor loop (navigate 10 steps to target)
    pos = [3, 3]
    def observe_fn():
        return Observation(position=tuple(pos))
    def act_fn(action):
        # navigate_to returns int action IDs: UP=1, DOWN=2, LEFT=3, RIGHT=4
        if action == 4: pos[0] += 1    # RIGHT
        elif action == 2: pos[1] += 1  # DOWN
        elif action == 3: pos[0] -= 1  # LEFT
        elif action == 1: pos[1] -= 1  # UP

    def run_executor():
        pos[0], pos[1] = 3, 3
        inv = SkillInvocation(skill_id="navigate_to", params={"x": 7, "y": 7}, max_steps=20)
        return execute_skill(inv, observe_fn, act_fn)

    timeit(run_executor, "execute_skill(navigate_to, 4-step path)", iterations=50)

    cur, peak = mem_snapshot()
    print(f"  Memory: current={cur:.3f}MB  peak={peak:.3f}MB")
    tracemalloc.stop()

# ── Profile Metacog ──────────────────────────────────────────────────

def profile_metacog():
    print("\n=== Metacog (Interoceptive Monitor) ===")
    gc.collect()
    tracemalloc.start()
    from sage.cognition.metacog.core import Metacog, MetacogConfig

    mc = Metacog(config=MetacogConfig())

    # Single tick with no signals
    def healthy_tick():
        mc.observe_tick(mc._tick + 1, action_taken={"plugin": "move", "args": {"dir": "right"}},
                       state_delta={"pos_changed": True}, snarc_novelty=0.5,
                       atp_balance=80.0, atp_cost=1.0, estimated_actions_to_goal=10.0)

    timeit(healthy_tick, "observe_tick (healthy, no signals)", iterations=50)

    # Tick that should trigger perseveration
    mc2 = Metacog(config=MetacogConfig())
    def perseveration_tick():
        mc2.reset()
        for i in range(5):
            mc2.observe_tick(i, action_taken={"plugin": "move", "args": {"dir": "right"}},
                            state_delta=None, snarc_novelty=0.01)

    timeit(perseveration_tick, "5 ticks → perseveration detection", iterations=20)

    # Tick with WM integration
    from sage.cognition.working_memory import WorkingMemory
    wm = WorkingMemory()
    wm.add_item("goal", {"target": "reach exit"}, priority=0.9)
    wm.add_item("plan_step", {"step": "explore dungeon"}, priority=0.7)
    mc3 = Metacog(config=MetacogConfig(), wm=wm)

    def wm_tick():
        mc3.observe_tick(mc3._tick + 1, action_taken={"plugin": "move", "args": {"dir": "right"}},
                        state_delta={"pos_changed": True}, snarc_novelty=0.5)

    timeit(wm_tick, "observe_tick with WM integration", iterations=50)

    # get_block_list
    timeit(lambda: mc.get_block_list(), "get_block_list()")

    cur, peak = mem_snapshot()
    print(f"  Memory: current={cur:.3f}MB  peak={peak:.3f}MB")
    print(f"  Stats: {mc.stats()}")
    tracemalloc.stop()

# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SAGE Brain-Arch Edge Profile (v2 — with motor skills + metacog)")
    print(f"Platform: {sys.platform} / {os.uname().machine}")
    print(f"Python: {sys.version.split()[0]}")

    profile_wm()
    profile_rpe()
    profile_cerebellum()
    profile_episodic()
    profile_motor_skills()
    profile_metacog()
    profile_scaling()

    print("\n=== System State ===")
    os.system("free -h | head -2")
    os.system("cat /sys/class/thermal/thermal_zone0/temp")
    print("\nDone.")
