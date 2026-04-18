#!/usr/bin/env python3
"""
Data gathering session — run the consciousness loop with:
1. Router shadow hook (captures dispatch decisions for training data)
2. Episodic index (binds high-salience experiences)
3. WM integration (consolidates on goal completion)

This produces training data for the thalamic router AND episodic
retrieval benchmarks simultaneously.

Usage:
    cd /home/dp/ai-workspace/SAGE
    source sage/gateway/router-shadow.env
    python3 sage/cognition/episodic/data_gathering.py [--cycles 50]
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.environ.setdefault('SAGE_ROUTER_SHADOW', '1')
os.environ.setdefault('SAGE_ROUTER_DATA_DIR',
                      os.path.expanduser('~/ai-workspace/private-context/training-data/router'))


def run_data_gathering(cycles: int = 50, db_path: str = None):
    """Run consciousness loop cycles and collect data."""

    from sage.cognition.episodic.index import EpisodicIndex

    # Initialize episodic index
    if db_path is None:
        db_path = os.path.expanduser('~/ai-workspace/private-context/training-data/episodic/thor_episodes.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    episodic = EpisodicIndex(db_path=db_path)
    print(f"Episodic index: {episodic.stats().get('count', 0)} existing episodes")

    # Try to initialize the consciousness loop
    try:
        from sage.core.sage_consciousness import SAGEConsciousness
        consciousness = SAGEConsciousness(
            simulation_mode=True,
            enable_circadian=False,
        )
        consciousness.set_episodic_index(episodic)
        print(f"Consciousness loop initialized (shadow={os.environ.get('SAGE_ROUTER_SHADOW', '?')})")
    except Exception as e:
        print(f"Could not initialize full consciousness loop: {e}")
        print("Running episodic-only data gathering from game logs instead.")
        _gather_from_game_logs(episodic)
        return

    # Run cycles
    print(f"\nRunning {cycles} consciousness loop cycles...")
    t0 = time.time()
    for i in range(cycles):
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(consciousness.step())
        except Exception as e:
            if i == 0:
                print(f"  Cycle 0 failed: {e}")
                print("  Falling back to episodic-only data gathering.")
                _gather_from_game_logs(episodic)
                return
            print(f"  Cycle {i} error: {e}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            ep_stats = episodic.stats()
            print(f"  Cycle {i+1}/{cycles}: {ep_stats.get('count', 0)} episodes, "
                  f"{elapsed:.1f}s elapsed")

    # Summary
    elapsed = time.time() - t0
    ep_stats = episodic.stats()
    print(f"\nData gathering complete:")
    print(f"  Cycles: {cycles}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Episodes: {ep_stats}")

    # Check router shadow data
    router_dir = os.environ.get('SAGE_ROUTER_DATA_DIR', '')
    if router_dir and os.path.exists(router_dir):
        import glob
        partitions = glob.glob(f"{router_dir}/**/*.jsonl*", recursive=True)
        total_records = 0
        for p in partitions:
            if p.endswith('.gz'):
                import gzip
                with gzip.open(p, 'rt') as f:
                    total_records += sum(1 for _ in f)
            else:
                with open(p) as f:
                    total_records += sum(1 for _ in f)
        print(f"  Router records: {total_records} across {len(partitions)} partition files")
    else:
        print(f"  Router data dir: {router_dir} (not found)")


def _gather_from_game_logs(episodic):
    """Fallback: bind game logs as episodes."""
    from sage.cognition.episodic.game_log_binder import LOG_DIR, bind_plh_log, bind_competition_log
    print("\nBinding game logs as episodes...")

    total = 0
    for log_file in sorted(LOG_DIR.glob("*.json")):
        try:
            name = log_file.stem
            if "autonomous" in name:
                continue
            if "plh" in name or "goalscaffold" in name or "twopipe" in name:
                count = bind_plh_log(log_file, episodic)
            elif "competition" in name:
                count = bind_competition_log(log_file, episodic)
            else:
                count = bind_plh_log(log_file, episodic)
            total += count
            print(f"  {name}: {count} episodes")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    print(f"\nThor logs: {total} episodes bound")

    # Also bind any McNugget logs if available
    from sage.cognition.episodic.game_log_binder import LOG_DIR
    mcnugget_dir = LOG_DIR.parent.parent / "mcnugget" / "logs"
    if mcnugget_dir.exists():
        mc_count = 0
        for log_file in sorted(mcnugget_dir.glob("*.json")):
            try:
                name = log_file.stem
                with open(log_file) as f:
                    data = json.load(f)
                entries = data.get("log", [])
                game = data.get("game", "unknown")
                for entry in entries:
                    if entry.get("type") != "action":
                        continue
                    from sage.cognition.episodic.data import Episode
                    ep = Episode(
                        session_id="mcnugget",
                        state_signature={"game": game, "experiment": "harness_v1"},
                        snarc_scores={"surprise": 0.3, "novelty": 0.4},
                        action_taken=entry.get("action", ""),
                        outcome=entry.get("diff", ""),
                        reward=0.3 if entry.get("diff", "") and "no change" not in entry.get("diff", "") else -0.1,
                        tags=[game.split("-")[0], "mcnugget", "harness_v1", "game_play"],
                    )
                    episodic.bind(ep)
                    mc_count += 1
                print(f"  mcnugget/{name}: bound")
            except Exception as e:
                print(f"  mcnugget/{name}: ERROR {e}")
        if mc_count:
            print(f"  McNugget episodes: {mc_count}")

    print(f"\nFinal episodic stats: {episodic.stats()}")

    # Run retrieval benchmarks
    _run_retrieval_benchmarks(episodic)


def _run_retrieval_benchmarks(episodic):
    """Test retrieval quality on real data."""
    from sage.cognition.episodic.data import EpisodicCue

    print("\n--- Retrieval Benchmarks ---")

    benchmarks = [
        ("cd82 game episodes", EpisodicCue(tags=["cd82"])),
        ("tr87 game episodes", EpisodicCue(tags=["tr87"])),
        ("ft09 game episodes", EpisodicCue(tags=["ft09"])),
        ("high-reward actions", EpisodicCue(snarc_scores={"reward": 0.8, "surprise": 0.7})),
        ("failed actions", EpisodicCue(snarc_scores={"reward": 0.0, "surprise": 0.1})),
        ("SELECT actions", EpisodicCue(action_taken="SELECT")),
        ("LEFT actions", EpisodicCue(action_taken="LEFT")),
        ("cross-machine (mcnugget)", EpisodicCue(tags=["mcnugget"])),
    ]

    for name, cue in benchmarks:
        t0 = time.time()
        results = episodic.recall(cue, k=5)
        elapsed_ms = (time.time() - t0) * 1000
        if results:
            best = results[0]
            print(f"  {name}: {len(results)} results, {elapsed_ms:.1f}ms, "
                  f"best_sim={best.similarity:.3f} action={best.episode.action_taken}")
        else:
            print(f"  {name}: 0 results, {elapsed_ms:.1f}ms")

    # Consolidation test
    print("\n--- Consolidation Test ---")
    pre = episodic.stats()
    report = episodic.consolidate(similarity_threshold=0.9, min_cluster_size=5)
    post = episodic.stats()
    print(f"  Before: {pre['count']} episodes")
    print(f"  After: {post['count']} episodes")
    print(f"  Prototypes: {report.prototypes_created}, Merged: {report.episodes_merged}")
    print(f"  Duration: {report.duration_seconds:.2f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=50)
    p.add_argument("--db-path", default=None)
    args = p.parse_args()
    run_data_gathering(args.cycles, args.db_path)
