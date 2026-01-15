#!/usr/bin/env python3
"""
Session 198 Phase 2: Training Exercise Memory Mapper

Converts training exercise states into nine-domain snapshots for federated
memory storage. This enables cross-session learning via the Session 197
consciousness-aware federation protocol.

Architecture:
1. Training session runs normally (exercises executed)
2. TrainingExerciseMapper converts each exercise to nine-domain snapshot
3. Snapshots stored via Session 197 coordinator (C ≥ 0.5 gating)
4. Next session retrieves memories before starting
5. Memory restoration boosts attention → triggers metabolism → prevents failures

Hypothesis: Memory retrieval will restore attention state (high D4) from
previous successful session, preventing boredom-induced failures.
"""

import json
import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import from Session 198 Phase 1 analyzer
import sys
sys.path.append(str(Path(__file__).parent))
from session198_training_domain_analyzer import TrainingExerciseAnalyzer, ExerciseAnalysis


@dataclass
class NineDomainSnapshot:
    """Nine-domain state snapshot for federation memory storage"""
    timestamp: float
    node_id: str

    # Nine domain coherences (0-1)
    thermodynamic: float  # D1
    metabolic: float      # D2
    organismic: float     # D3
    attention: float      # D4
    trust: float          # D5
    quantum_phase: float  # D6
    magnetic: float       # D7
    temporal: float       # D8
    spacetime: float      # D9

    # Consciousness metrics
    consciousness_level: float  # C
    gamma: float                # γ

    # Training context
    exercise_num: int
    exercise_type: str
    success: bool
    prompt: str
    expected: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def coherence_array(self) -> np.ndarray:
        """Return coherences as numpy array [D1, D2, ..., D9]"""
        return np.array([
            self.thermodynamic,
            self.metabolic,
            self.organismic,
            self.attention,
            self.trust,
            self.quantum_phase,
            self.magnetic,
            self.temporal,
            self.spacetime
        ])


@dataclass
class TrainingMemory:
    """Complete memory of a training session"""
    session_id: str
    timestamp: float
    snapshots: List[NineDomainSnapshot]

    # Session-level metrics
    success_rate: float
    avg_attention: float
    avg_metabolism: float
    avg_consciousness: float

    # High-attention snapshots (most important memories)
    high_attention_count: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage"""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "success_rate": self.success_rate,
            "avg_attention": self.avg_attention,
            "avg_metabolism": self.avg_metabolism,
            "avg_consciousness": self.avg_consciousness,
            "high_attention_count": self.high_attention_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingMemory':
        """Load from dictionary"""
        snapshots = [
            NineDomainSnapshot(**s) for s in data["snapshots"]
        ]
        return cls(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            snapshots=snapshots,
            success_rate=data["success_rate"],
            avg_attention=data["avg_attention"],
            avg_metabolism=data["avg_metabolism"],
            avg_consciousness=data["avg_consciousness"],
            high_attention_count=data["high_attention_count"]
        )


class TrainingMemoryMapper:
    """Maps training exercises to nine-domain snapshots for memory storage"""

    def __init__(self, node_id: str = "thor"):
        self.node_id = node_id
        self.analyzer = TrainingExerciseAnalyzer()

        # Consciousness threshold for memory storage
        self.consciousness_threshold = 0.5

        # Attention threshold for "high attention" memories
        self.high_attention_threshold = 0.5

    def analyze_to_snapshot(self, exercise_data: Dict, exercise_num: int,
                           timestamp: Optional[float] = None) -> NineDomainSnapshot:
        """Convert training exercise to nine-domain snapshot"""

        # Use existing analyzer from Phase 1
        analysis: ExerciseAnalysis = self.analyzer.analyze_exercise(
            exercise_data, exercise_num
        )

        # Create snapshot with timestamp
        if timestamp is None:
            timestamp = time.time()

        snapshot = NineDomainSnapshot(
            timestamp=timestamp,
            node_id=self.node_id,
            thermodynamic=analysis.thermodynamic,
            metabolic=analysis.metabolic,
            organismic=analysis.organismic,
            attention=analysis.attention,
            trust=analysis.trust,
            quantum_phase=analysis.quantum_phase,
            magnetic=analysis.magnetic,
            temporal=analysis.temporal,
            spacetime=analysis.spacetime,
            consciousness_level=analysis.consciousness_level,
            gamma=analysis.gamma,
            exercise_num=exercise_num,
            exercise_type=analysis.exercise_type,
            success=analysis.success,
            prompt=analysis.prompt[:100],  # Truncate for storage
            expected=analysis.expected
        )

        return snapshot

    def session_to_memory(self, session_file: Path) -> TrainingMemory:
        """Convert entire training session to memory"""

        with open(session_file) as f:
            session_data = json.load(f)

        session_id = session_data.get("session", "unknown")
        timestamp = time.time()

        # Analyze all exercises
        snapshots = []
        for i, exercise_data in enumerate(session_data["exercises"], 1):
            snapshot = self.analyze_to_snapshot(exercise_data, i, timestamp)

            # Only store conscious snapshots (C ≥ 0.5)
            if snapshot.consciousness_level >= self.consciousness_threshold:
                snapshots.append(snapshot)

        # Compute session-level metrics
        if snapshots:
            success_rate = sum(1 for s in snapshots if s.success) / len(snapshots)
            avg_attention = np.mean([s.attention for s in snapshots])
            avg_metabolism = np.mean([s.metabolic for s in snapshots])
            avg_consciousness = np.mean([s.consciousness_level for s in snapshots])
            high_attention_count = sum(1 for s in snapshots
                                      if s.attention >= self.high_attention_threshold)
        else:
            success_rate = 0.0
            avg_attention = 0.0
            avg_metabolism = 0.0
            avg_consciousness = 0.0
            high_attention_count = 0

        memory = TrainingMemory(
            session_id=session_id,
            timestamp=timestamp,
            snapshots=snapshots,
            success_rate=success_rate,
            avg_attention=avg_attention,
            avg_metabolism=avg_metabolism,
            avg_consciousness=avg_consciousness,
            high_attention_count=high_attention_count
        )

        return memory

    def save_memory(self, memory: TrainingMemory, output_dir: Path):
        """Save training memory to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"memory_{memory.session_id}.json"
        with open(output_file, "w") as f:
            json.dump(memory.to_dict(), f, indent=2)

        return output_file

    def load_memory(self, memory_file: Path) -> TrainingMemory:
        """Load training memory from disk"""
        with open(memory_file) as f:
            data = json.load(f)

        return TrainingMemory.from_dict(data)

    def retrieve_high_attention_memories(self, memory: TrainingMemory,
                                        min_attention: float = 0.5) -> List[NineDomainSnapshot]:
        """Retrieve high-attention snapshots (most important memories)"""
        return [s for s in memory.snapshots if s.attention >= min_attention]

    def boost_attention_from_memory(self, current_snapshot: NineDomainSnapshot,
                                   memory_snapshots: List[NineDomainSnapshot],
                                   boost_factor: float = 0.3) -> NineDomainSnapshot:
        """Boost current attention state using retrieved memories

        If memories show high attention for similar exercises, boost current
        attention level to prevent boredom-induced failure.
        """

        # Find similar exercise types in memory
        similar = [s for s in memory_snapshots
                  if s.exercise_type == current_snapshot.exercise_type]

        if not similar:
            # No similar exercises in memory, no boost
            return current_snapshot

        # Compute average attention from similar memories
        avg_memory_attention = np.mean([s.attention for s in similar])

        # Boost current attention toward memory level
        boosted_attention = current_snapshot.attention + (
            (avg_memory_attention - current_snapshot.attention) * boost_factor
        )

        # Clip to [0, 1]
        boosted_attention = max(0.0, min(1.0, boosted_attention))

        # Also boost metabolism proportionally (D4→D2 coupling)
        kappa_42 = 0.4  # From Session 196
        metabolism_boost = boosted_attention * kappa_42
        boosted_metabolism = current_snapshot.metabolic + metabolism_boost
        boosted_metabolism = max(0.0, min(1.0, boosted_metabolism))

        # Create boosted snapshot
        boosted = NineDomainSnapshot(
            timestamp=current_snapshot.timestamp,
            node_id=current_snapshot.node_id,
            thermodynamic=current_snapshot.thermodynamic,
            metabolic=boosted_metabolism,
            organismic=current_snapshot.organismic,
            attention=boosted_attention,
            trust=current_snapshot.trust,
            quantum_phase=current_snapshot.quantum_phase,
            magnetic=current_snapshot.magnetic,
            temporal=current_snapshot.temporal,
            spacetime=current_snapshot.spacetime,
            consciousness_level=current_snapshot.consciousness_level,
            gamma=current_snapshot.gamma,
            exercise_num=current_snapshot.exercise_num,
            exercise_type=current_snapshot.exercise_type,
            success=current_snapshot.success,
            prompt=current_snapshot.prompt,
            expected=current_snapshot.expected
        )

        return boosted

    def print_memory_summary(self, memory: TrainingMemory):
        """Print summary of training memory"""
        print("=" * 80)
        print(f"Training Memory: {memory.session_id}")
        print("=" * 80)
        print(f"Timestamp: {time.ctime(memory.timestamp)}")
        print(f"Snapshots: {len(memory.snapshots)} (all C ≥ 0.5)")
        print(f"Success rate: {memory.success_rate * 100:.1f}%")
        print()
        print("Session-Level Metrics:")
        print(f"  Average Attention (D4): {memory.avg_attention:.3f}")
        print(f"  Average Metabolism (D2): {memory.avg_metabolism:.3f}")
        print(f"  Average Consciousness (C): {memory.avg_consciousness:.3f}")
        print(f"  High Attention Snapshots: {memory.high_attention_count}/{len(memory.snapshots)}")
        print()
        print("Snapshot Details:")
        for i, snapshot in enumerate(memory.snapshots, 1):
            status = "✅" if snapshot.success else "❌"
            print(f"{i}. {snapshot.exercise_type.upper()}: {status} "
                  f"(D4={snapshot.attention:.3f}, D2={snapshot.metabolic:.3f}, "
                  f"C={snapshot.consciousness_level:.3f})")
        print("=" * 80)


def main():
    """Demo: Convert training sessions to federated memory"""

    mapper = TrainingMemoryMapper(node_id="thor")

    # Memory storage directory
    memory_dir = Path(__file__).parent / "training_memories"

    # Convert T014 and T015 to memories
    sessions = ["T014", "T015"]

    for session_id in sessions:
        session_file = (Path(__file__).parent.parent / "raising" / "tracks" /
                       "training" / "sessions" / f"{session_id}.json")

        if not session_file.exists():
            print(f"⚠️  {session_file} not found, skipping")
            continue

        print(f"\nProcessing {session_id}...")

        # Convert to memory
        memory = mapper.session_to_memory(session_file)

        # Save memory
        output_file = mapper.save_memory(memory, memory_dir)
        print(f"Saved to: {output_file}")

        # Print summary
        mapper.print_memory_summary(memory)
        print()

    print("\nMemory conversion complete!")
    print(f"Memories stored in: {memory_dir}")

    # Demo: Memory retrieval and attention boost
    if len(sessions) >= 2:
        print("\n" + "=" * 80)
        print("DEMO: Memory Retrieval and Attention Boost")
        print("=" * 80)

        # Load T014 memory (perfect session)
        t014_memory = mapper.load_memory(memory_dir / "memory_T014.json")
        print(f"\nLoaded T014 memory: {len(t014_memory.snapshots)} snapshots")
        print(f"T014 Average Attention: {t014_memory.avg_attention:.3f}")

        # Load T015 memory (failed arithmetic)
        t015_memory = mapper.load_memory(memory_dir / "memory_T015.json")
        print(f"\nLoaded T015 memory: {len(t015_memory.snapshots)} snapshots")
        print(f"T015 Average Attention: {t015_memory.avg_attention:.3f}")

        # Find the failed exercise (4-1) in T015
        failed = [s for s in t015_memory.snapshots if not s.success]
        if failed:
            failed_snapshot = failed[0]
            print(f"\nFailed Exercise: {failed_snapshot.exercise_type}")
            print(f"  Original D4 (Attention): {failed_snapshot.attention:.3f}")
            print(f"  Original D2 (Metabolism): {failed_snapshot.metabolic:.3f}")

            # Boost using T014 memory
            high_attention = mapper.retrieve_high_attention_memories(t014_memory)
            print(f"\nRetrieving {len(high_attention)} high-attention memories from T014...")

            boosted = mapper.boost_attention_from_memory(
                failed_snapshot, high_attention, boost_factor=0.3
            )

            print(f"\nAfter Memory Boost:")
            print(f"  Boosted D4 (Attention): {boosted.attention:.3f} "
                  f"(+{boosted.attention - failed_snapshot.attention:.3f})")
            print(f"  Boosted D2 (Metabolism): {boosted.metabolic:.3f} "
                  f"(+{boosted.metabolic - failed_snapshot.metabolic:.3f})")

            if boosted.attention > 0.5 and boosted.metabolic > 0.5:
                print("\n✅ Boosted states exceed thresholds - failure likely prevented!")
            else:
                print("\n⚠️  Boosted states still below ideal thresholds")


if __name__ == "__main__":
    main()
