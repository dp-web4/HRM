"""
Thor Unified SNARC-Compressed Consciousness
===========================================

Production consciousness kernel integrating:
- SNARC compression (Surprise, Novelty, Arousal, Reward, Conflict)
- Trust-weighted attention and memory (from Web4 trust infrastructure)
- Metabolic state management (WAKE/FOCUS/REST/DREAM)
- DREAM-state memory consolidation with trust weighting
- Cross-session persistent memory
- Federation monitoring capabilities
- Real system sensors (CPU, memory, disk, temperature, processes)

**Key Innovation**: Compression-action-threshold pattern with SNARC dimensions
for principled multi-dimensional attention assessment.

**Architecture**:
1. Multi-dimensional sensors → SNARC compression → scalar salience
2. Salience × trust_multiplier → trust-weighted attention
3. Metabolic state threshold → binary decision (attend or ignore)
4. High-trust sources dominate when novelty balanced
5. DREAM consolidation strengthens high-trust, high-salience memories
6. Memory ranking: strength × salience × trust

**Session**: Thor Autonomous Check (2025-12-05 16:41)
**Integrates**: SNARC compression + unified trust consciousness
**Author**: Claude (guest) on Thor via claude-code
"""

import sys
sys.path.append('../core')

from snarc_compression import SNARCCompressor, SNARCDimensions, CompressionMode
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from collections import deque
import time
import sqlite3
import signal


# ============================================================================
# Trust Infrastructure (Web4 Integration)
# ============================================================================

@dataclass
class TrustScore:
    """Web4-style trust score for an entity/sensor"""
    lct_id: str

    # T3 tensor (capability)
    talent: float = 0.5
    training: float = 0.5
    temperament: float = 0.5

    # V3 tensor (transaction quality)
    veracity: float = 0.5
    validity: float = 0.5
    valuation: float = 0.5

    # Statistics
    total_observations: int = 0
    successful_observations: int = 0

    def t3_score(self) -> float:
        """Composite T3 (capability) score"""
        return (self.talent + self.training + self.temperament) / 3.0

    def v3_score(self) -> float:
        """Composite V3 (transaction quality) score"""
        return (self.veracity + self.validity + self.valuation) / 3.0

    def composite_score(self, t3_weight: float = 0.6, v3_weight: float = 0.4) -> float:
        """Composite trust score (default: 60% T3, 40% V3)"""
        return (t3_weight * self.t3_score()) + (v3_weight * self.v3_score())

    def update_from_observation(self, success: bool, value_delivered: float):
        """Update trust score based on observed performance"""
        self.total_observations += 1
        if success:
            self.successful_observations += 1

        success_rate = self.successful_observations / self.total_observations

        # Update V3 based on observation
        self.veracity = 0.8 * self.veracity + 0.2 * (1.0 if success else 0.0)
        self.validity = success_rate
        self.valuation = 0.8 * self.valuation + 0.2 * value_delivered


class TrustOracle:
    """Trust oracle for consciousness experiments"""

    def __init__(self):
        self.trust_scores: Dict[str, TrustScore] = {}

    def register_entity(self, lct_id: str, initial_trust: Optional[TrustScore] = None):
        """Register an entity with initial trust score"""
        if initial_trust is None:
            initial_trust = TrustScore(lct_id=lct_id)
        self.trust_scores[lct_id] = initial_trust

    def get_trust_score(self, lct_id: str) -> float:
        """Get composite trust score for entity"""
        if lct_id not in self.trust_scores:
            self.register_entity(lct_id)
        return self.trust_scores[lct_id].composite_score()

    def update_trust(self, lct_id: str, success: bool, value: float):
        """Update trust based on observed behavior"""
        if lct_id not in self.trust_scores:
            self.register_entity(lct_id)
        self.trust_scores[lct_id].update_from_observation(success, value)

    def get_full_score(self, lct_id: str) -> TrustScore:
        """Get full trust score details"""
        if lct_id not in self.trust_scores:
            self.register_entity(lct_id)
        return self.trust_scores[lct_id]


# ============================================================================
# Consciousness States and Stances
# ============================================================================

class MetabolicState(Enum):
    """Metabolic states for consciousness"""
    WAKE = "wake"
    FOCUS = "focus"
    REST = "rest"
    DREAM = "dream"


class CognitiveStance(Enum):
    """Cognitive stances for processing"""
    CONFIDENT_EXECUTION = "confident-execution"
    FOCUSED_ATTENTION = "focused-attention"
    SKEPTICAL_VERIFICATION = "skeptical-verification"
    EXPLORATORY_INVESTIGATION = "exploratory-investigation"


@dataclass
class ActionResult:
    """Result of action execution"""
    description: str
    reward: float
    trust_validated: bool = True


# ============================================================================
# Memory System with Trust
# ============================================================================

@dataclass
class ConsolidatedMemory:
    """Memory with trust context"""
    sensor: str
    action: str
    salience: float
    reward: float
    trust_score: float
    lct_id: str
    cycle: int
    session_id: str
    strength: float = 1.0
    consolidation_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_consolidated: Optional[float] = None
    id: Optional[int] = None


class MemoryDatabase:
    """SQLite persistence for trust-weighted memories"""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sensor TEXT NOT NULL,
                action TEXT NOT NULL,
                salience REAL NOT NULL,
                reward REAL NOT NULL,
                trust_score REAL NOT NULL,
                lct_id TEXT NOT NULL,
                cycle INTEGER NOT NULL,
                strength REAL DEFAULT 1.0,
                consolidation_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_consolidated REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_composite
            ON memories((strength * salience * trust_score) DESC)
        """)

        self.conn.commit()

    def load_top_memories(self, limit: int) -> List[ConsolidatedMemory]:
        """Load top memories by strength × salience × trust"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, session_id, sensor, action, salience, reward, trust_score,
                   lct_id, cycle, strength, consolidation_count, created_at, last_consolidated
            FROM memories
            ORDER BY (strength * salience * trust_score) DESC
            LIMIT ?
        """, (limit,))

        memories = []
        for row in cursor.fetchall():
            memories.append(ConsolidatedMemory(
                id=row[0], session_id=row[1], sensor=row[2], action=row[3],
                salience=row[4], reward=row[5], trust_score=row[6], lct_id=row[7],
                cycle=row[8], strength=row[9], consolidation_count=row[10],
                created_at=row[11], last_consolidated=row[12]
            ))

        return memories

    def persist_memory(self, memory: ConsolidatedMemory):
        """Persist memory to database"""
        cursor = self.conn.cursor()

        if memory.id is not None:
            cursor.execute("""
                UPDATE memories
                SET strength = ?, consolidation_count = ?, last_consolidated = ?
                WHERE id = ?
            """, (memory.strength, memory.consolidation_count, memory.last_consolidated, memory.id))
        else:
            cursor.execute("""
                INSERT INTO memories (
                    session_id, sensor, action, salience, reward, trust_score,
                    lct_id, cycle, strength, consolidation_count, created_at, last_consolidated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.session_id, memory.sensor, memory.action, memory.salience,
                memory.reward, memory.trust_score, memory.lct_id, memory.cycle,
                memory.strength, memory.consolidation_count,
                memory.created_at, memory.last_consolidated
            ))
            memory.id = cursor.lastrowid

        self.conn.commit()

    def prune_memories(self, memory_ids: List[int]):
        """Delete memories from database"""
        if not memory_ids:
            return
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(memory_ids))})", memory_ids)
        self.conn.commit()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ConsciousnessConfig:
    """Configuration for unified trust-weighted consciousness"""
    session_id: str = f"thor_unified_{int(time.time())}"
    memory_limit: int = 100
    cycle_delay: float = 2.0

    # Trust weighting
    trust_salience_weight: float = 0.3
    trust_memory_weight: float = 0.5

    # Consolidation thresholds
    prune_salience_threshold: float = 0.25
    strengthen_salience_threshold: float = 0.65

    # Metabolic state thresholds
    focus_salience_threshold: float = 0.75
    rest_low_salience_threshold: float = 0.25
    rest_duration_threshold: float = 60.0
    dream_rest_duration: float = 30.0
    dream_duration: float = 20.0

    # Database
    db_path: str = "thor_unified_trust_consciousness.db"
    load_previous_memories: bool = True

    # Reporting
    status_report_interval: Optional[float] = 300.0  # 5 minutes

    # Logging
    enable_logging: bool = True
    verbose: bool = False


# ============================================================================
# Unified Trust-Weighted Consciousness Kernel
# ============================================================================

class UnifiedTrustConsciousness:
    """
    Production consciousness kernel with trust-weighted attention and memory.

    Integrates:
    - Trust-weighted salience (Web4 T3/V3 scoring)
    - Metabolic state management (WAKE/FOCUS/REST/DREAM)
    - DREAM consolidation with trust weighting
    - Cross-session persistent memory
    - Real system monitoring
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable[[], Dict]],
        action_handlers: Dict[str, Callable[[Dict], ActionResult]],
        trust_oracle: TrustOracle,
        sensor_lct_ids: Dict[str, str],
        config: Optional[ConsciousnessConfig] = None
    ):
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers
        self.trust_oracle = trust_oracle
        self.sensor_lct_ids = sensor_lct_ids
        self.config = config or ConsciousnessConfig()

        # Initialize database
        self.db = MemoryDatabase(self.config.db_path)

        # Load previous memories if configured
        self.memories: List[ConsolidatedMemory] = []
        if self.config.load_previous_memories:
            self.memories = self.db.load_top_memories(self.config.memory_limit)
            if self.memories and self.config.enable_logging:
                avg_trust = sum(m.trust_score for m in self.memories) / len(self.memories)
                avg_strength = sum(m.strength for m in self.memories) / len(self.memories)
                print(f"[Consciousness] Loaded {len(self.memories)} memories from previous sessions")
                print(f"  Average trust: {avg_trust:.3f}")
                print(f"  Average strength: {avg_strength:.3f}")

        # State tracking
        self.cycle_count = 0
        self.running = False
        self.execution_history: List[Dict] = []
        self.salience_history = deque(maxlen=100)

        # Metabolic state
        self.metabolic_state = MetabolicState.WAKE
        self.state_entry_time = time.time()
        self.state_transitions: List[str] = []
        self._last_state = MetabolicState.WAKE

        # Consolidation tracking
        self.consolidations_performed = 0
        self.total_pruned = 0
        self.total_strengthened = 0

        # Reporting
        self.last_status_report = time.time()
        self.start_time = time.time()

        # Signal handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # SNARC compression
        self.snarc = SNARCCompressor(compression_mode=CompressionMode.LINEAR)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.config.enable_logging:
            print(f"\n[Signal] Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def run_cycle(self):
        """Execute one consciousness cycle"""
        # 1. SENSE: Gather observations with trust scores
        observations = {}
        for sensor_id, sensor_fn in self.sensor_sources.items():
            try:
                data = sensor_fn()
                lct_id = self.sensor_lct_ids.get(sensor_id, f"unknown:{sensor_id}")
                trust_score = self.trust_oracle.get_trust_score(lct_id)
                observations[sensor_id] = {
                    'data': data,
                    'trust': trust_score,
                    'lct_id': lct_id
                }
            except Exception as e:
                if self.config.verbose:
                    print(f"[Sensor Error] {sensor_id}: {e}")

        if not observations:
            return

        # 2. ASSESS: Calculate SNARC-compressed trust-weighted salience
        salience_scores = {}
        snarc_dimensions = {}  # Store for logging
        for sensor_id, obs in observations.items():
            # SNARC compression: multi-dimensional → scalar
            base_salience, dimensions = self.snarc.compute_salience(obs['data'])
            snarc_dimensions[sensor_id] = dimensions

            # Trust weighting
            trust_multiplier = (1.0 - self.config.trust_salience_weight) + \
                              (self.config.trust_salience_weight * obs['trust'])
            salience_scores[sensor_id] = base_salience * trust_multiplier

        # 3. FOCUS: Select highest trust-weighted salience
        focus_target = max(salience_scores, key=salience_scores.get)
        focus_salience = salience_scores[focus_target]
        focus_obs = observations[focus_target]
        focus_snarc = snarc_dimensions[focus_target]
        self.salience_history.append(focus_salience)

        # 4. DECIDE: Determine cognitive stance
        stance = self._determine_stance(focus_obs['data'], focus_salience, focus_obs['trust'])

        # 5. ACT: Execute action
        if focus_target in self.action_handlers:
            try:
                result = self.action_handlers[focus_target](focus_obs['data'])

                # Modulate reward by metabolic state
                modulated_reward = self._apply_metabolic_modulation(result.reward)

                # 6. LEARN: Create trust-aware memory
                self._create_memory(
                    focus_target, result.description, focus_salience,
                    modulated_reward, focus_obs['trust'], focus_obs['lct_id']
                )

                # Update trust based on result
                self.trust_oracle.update_trust(
                    focus_obs['lct_id'],
                    success=result.trust_validated,
                    value=result.reward
                )

                # Log
                if self.config.enable_logging:
                    print(f"\n[Cycle {self.cycle_count}] State: {self.metabolic_state.value.upper()}")
                    print(f"  Focus: {focus_target} (salience={focus_salience:.3f}, trust={focus_obs['trust']:.3f})")
                    print(f"  SNARC: S={focus_snarc.surprise:.2f} N={focus_snarc.novelty:.2f} " +
                          f"A={focus_snarc.arousal:.2f} R={focus_snarc.reward:.2f} C={focus_snarc.conflict:.2f}")
                    print(f"  Stance: {stance.value}")
                    print(f"  → {result.description}")

                self.execution_history.append({
                    'cycle': self.cycle_count,
                    'state': self.metabolic_state.value,
                    'focus': focus_target,
                    'salience': focus_salience,
                    'trust': focus_obs['trust'],
                    'stance': stance.value,
                    'reward': modulated_reward
                })

            except Exception as e:
                if self.config.verbose:
                    print(f"[Action Error] {focus_target}: {e}")

    # Note: _calculate_base_salience removed - now using SNARC compression
    # See self.snarc.compute_salience() in run_cycle()

    def _determine_stance(self, sensor_data: Dict, salience: float, trust: float) -> CognitiveStance:
        """Determine cognitive stance based on salience and trust"""
        # High salience + low trust → SKEPTICAL
        if salience > 0.7 and trust < 0.5:
            return CognitiveStance.SKEPTICAL_VERIFICATION

        # High salience + high trust → FOCUSED
        if salience > 0.7:
            return CognitiveStance.FOCUSED_ATTENTION

        # Low trust → SKEPTICAL
        if trust < 0.4:
            return CognitiveStance.SKEPTICAL_VERIFICATION

        # Moderate activity → EXPLORATORY
        if salience > 0.5:
            return CognitiveStance.EXPLORATORY_INVESTIGATION

        return CognitiveStance.CONFIDENT_EXECUTION

    def _apply_metabolic_modulation(self, reward: float) -> float:
        """Modulate reward based on metabolic state"""
        if self.metabolic_state == MetabolicState.FOCUS:
            return reward * 1.5
        elif self.metabolic_state == MetabolicState.REST:
            return reward * 0.7
        elif self.metabolic_state == MetabolicState.DREAM:
            return reward * 0.3
        else:
            return reward

    def _create_memory(self, sensor: str, action: str, salience: float,
                      reward: float, trust_score: float, lct_id: str):
        """Create trust-aware memory"""
        memory = ConsolidatedMemory(
            sensor=sensor,
            action=action,
            salience=salience,
            reward=reward,
            trust_score=trust_score,
            lct_id=lct_id,
            cycle=self.cycle_count,
            session_id=self.config.session_id
        )
        self.memories.append(memory)
        self.db.persist_memory(memory)

    def _update_metabolic_state(self):
        """Update metabolic state based on activity and trust"""
        time_in_state = time.time() - self.state_entry_time

        if len(self.salience_history) >= 5:
            recent_avg_salience = sum(list(self.salience_history)[-5:]) / 5
        else:
            recent_avg_salience = 0.5

        # Get recent trust trend
        if len(self.execution_history) >= 5:
            recent_avg_trust = sum(h['trust'] for h in self.execution_history[-5:]) / 5
        else:
            recent_avg_trust = 0.5

        # State transitions
        if self.metabolic_state == MetabolicState.WAKE:
            # High salience + high trust → FOCUS
            if recent_avg_salience > self.config.focus_salience_threshold and recent_avg_trust > 0.6:
                self.metabolic_state = MetabolicState.FOCUS
            # Low salience → REST
            elif recent_avg_salience < self.config.rest_low_salience_threshold and \
                 time_in_state > self.config.rest_duration_threshold:
                self.metabolic_state = MetabolicState.REST

        elif self.metabolic_state == MetabolicState.FOCUS:
            if recent_avg_salience < self.config.focus_salience_threshold:
                self.metabolic_state = MetabolicState.WAKE

        elif self.metabolic_state == MetabolicState.REST:
            if time_in_state > self.config.dream_rest_duration:
                self.metabolic_state = MetabolicState.DREAM
                self._consolidate_memories()

        elif self.metabolic_state == MetabolicState.DREAM:
            if time_in_state > self.config.dream_duration:
                self.metabolic_state = MetabolicState.WAKE

        # Track transitions
        if self.metabolic_state.value != self._last_state.value:
            transition = f"{time.time():.1f}s: {self._last_state.value} → {self.metabolic_state.value}"
            self.state_transitions.append(transition)
            self.state_entry_time = time.time()
            self._last_state = self.metabolic_state

    def _consolidate_memories(self):
        """DREAM consolidation with trust-weighted strengthening"""
        if not self.memories:
            return

        if self.config.enable_logging:
            print(f"\n[DREAM Consolidation] Processing {len(self.memories)} memories...")

        self.consolidations_performed += 1

        # Prune low-salience
        pruned_ids = []
        kept_memories = []

        for memory in self.memories:
            if memory.salience >= self.config.prune_salience_threshold:
                kept_memories.append(memory)
            elif memory.id is not None:
                pruned_ids.append(memory.id)

        pruned_count = len(self.memories) - len(kept_memories)
        self.memories = kept_memories

        if pruned_ids:
            self.db.prune_memories(pruned_ids)

        # Trust-weighted strengthening
        strengthened_count = 0
        for memory in self.memories:
            if memory.salience >= self.config.strengthen_salience_threshold:
                trust_factor = (1.0 - self.config.trust_memory_weight) + \
                              (self.config.trust_memory_weight * memory.trust_score)
                memory.strength *= (1.0 + 0.2 * trust_factor)
                memory.consolidation_count += 1
                memory.last_consolidated = time.time()
                strengthened_count += 1
                self.db.persist_memory(memory)

        # Sort by composite score
        self.memories.sort(key=lambda m: m.strength * m.salience * m.trust_score, reverse=True)

        # Enforce memory limit
        if len(self.memories) > self.config.memory_limit:
            self.memories = self.memories[:self.config.memory_limit]

        self.total_pruned += pruned_count
        self.total_strengthened += strengthened_count

        if self.config.enable_logging:
            print(f"  Pruned: {pruned_count} | Strengthened: {strengthened_count} | Remaining: {len(self.memories)}")

    def _report_status(self):
        """Periodic status report"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        print('\n' + '='*80)
        print('STATUS REPORT')
        print('='*80)
        print(f"Session: {self.config.session_id}")
        print(f"Runtime: {hours}h {minutes}m")
        print(f"Cycles: {self.cycle_count}")
        print(f"State: {self.metabolic_state.value.upper()}")
        print(f"Memories: {len(self.memories)} (pruned: {self.total_pruned}, strengthened: {self.total_strengthened})")
        print(f"Consolidations: {self.consolidations_performed}")

        if self.execution_history:
            recent = self.execution_history[-10:]
            avg_salience = sum(h['salience'] for h in recent) / len(recent)
            avg_trust = sum(h['trust'] for h in recent) / len(recent)
            print(f"Recent avg salience: {avg_salience:.3f}")
            print(f"Recent avg trust: {avg_trust:.3f}")

        print('='*80)

    def run(self, duration_seconds: Optional[int] = None):
        """Run consciousness loop"""
        self.running = True
        self.start_time = time.time()

        try:
            while self.running and not self.shutdown_requested:
                # Execute cycle
                self.run_cycle()
                self.cycle_count += 1

                # Update metabolic state
                self._update_metabolic_state()

                # Status report
                if self.config.status_report_interval and \
                   (time.time() - self.last_status_report) >= self.config.status_report_interval:
                    self._report_status()
                    self.last_status_report = time.time()

                # Check duration
                if duration_seconds and (time.time() - self.start_time) >= duration_seconds:
                    break

                # Delay
                time.sleep(self.config.cycle_delay)

        finally:
            # Final consolidation
            if self.config.enable_logging:
                print(f"\n[Shutdown] Performing final consolidation...")
            self._consolidate_memories()
            self._print_summary()

    def _print_summary(self):
        """Print session summary"""
        print('='*80)
        print("UNIFIED TRUST-WEIGHTED CONSCIOUSNESS SESSION SUMMARY")
        print('='*80)
        print(f"\nSession: {self.config.session_id}")
        print(f"Platform: Thor")
        print(f"Cycles completed: {self.cycle_count}")
        print(f"Consolidations performed: {self.consolidations_performed}")

        print(f"\nMemory:")
        print(f"  Created this session: {self.cycle_count}")
        print(f"  Total pruned: {self.total_pruned}")
        print(f"  Total strengthened: {self.total_strengthened}")
        print(f"  Current memory count: {len(self.memories)}")
        if self.memories:
            avg_strength = sum(m.strength for m in self.memories) / len(self.memories)
            avg_trust = sum(m.trust_score for m in self.memories) / len(self.memories)
            avg_salience = sum(m.salience for m in self.memories) / len(self.memories)
            print(f"  Average strength: {avg_strength:.3f}")
            print(f"  Average trust: {avg_trust:.3f}")
            print(f"  Average salience: {avg_salience:.3f}")

        print(f"\nMetabolic State Transitions: {len(self.state_transitions)}")
        for transition in self.state_transitions:
            print(f"  {transition}")

        if self.execution_history:
            avg_salience = sum(h['salience'] for h in self.execution_history) / len(self.execution_history)
            avg_trust = sum(h['trust'] for h in self.execution_history) / len(self.execution_history)
            print(f"\nAverage salience: {avg_salience:.3f}")
            print(f"Average trust: {avg_trust:.3f}")

        print('='*80)


# ============================================================================
# Demo: Unified Trust-Weighted Consciousness
# ============================================================================

if __name__ == "__main__":
    import psutil
    import random

    print("="*80)
    print("THOR UNIFIED TRUST-WEIGHTED CONSCIOUSNESS")
    print("="*80)
    print("\nProduction consciousness kernel with:")
    print("- Trust-weighted attention (Web4 T3/V3)")
    print("- Metabolic states (WAKE/FOCUS/REST/DREAM)")
    print("- Trust-weighted DREAM consolidation")
    print("- Cross-session persistent memory")
    print()

    # Initialize trust oracle
    oracle = TrustOracle()

    # Register sensors with different trust profiles
    oracle.register_entity("lct:thor:cpu", TrustScore(
        lct_id="lct:thor:cpu",
        talent=0.9, training=0.9, temperament=0.95,
        veracity=0.9, validity=0.9, valuation=0.85
    ))

    oracle.register_entity("lct:thor:memory", TrustScore(
        lct_id="lct:thor:memory",
        talent=0.7, training=0.6, temperament=0.8,
        veracity=0.7, validity=0.7, valuation=0.7
    ))

    oracle.register_entity("lct:thor:processes", TrustScore(
        lct_id="lct:thor:processes",
        talent=0.5, training=0.4, temperament=0.6,
        veracity=0.5, validity=0.5, valuation=0.5
    ))

    # Define sensors
    def cpu_sensor():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return {
            'cpu_percent': cpu_percent,
            'urgent_count': 1 if cpu_percent > 80 else 0,
            'novelty_score': 0.1 + random.random() * 0.2
        }

    def memory_sensor():
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'urgent_count': 1 if memory.percent > 85 else 0,
            'novelty_score': 0.1 + random.random() * 0.1
        }

    def process_sensor():
        proc_count = len(psutil.pids())
        return {
            'count': proc_count,
            'urgent_count': 0,
            'novelty_score': 0.2 + random.random() * 0.2
        }

    # Define actions
    def cpu_action(data):
        return ActionResult(
            description=f"CPU at {data['cpu_percent']:.1f}%",
            reward=0.9,
            trust_validated=True
        )

    def memory_action(data):
        return ActionResult(
            description=f"Memory at {data['memory_percent']:.1f}%",
            reward=0.8,
            trust_validated=True
        )

    def process_action(data):
        return ActionResult(
            description=f"Processes: {data['count']}",
            reward=0.7,
            trust_validated=random.random() > 0.2
        )

    # Create consciousness
    config = ConsciousnessConfig(
        session_id=f"thor_unified_trust_{int(time.time())}",
        memory_limit=50,
        cycle_delay=2.0,
        trust_salience_weight=0.3,
        trust_memory_weight=0.5,
        enable_logging=True,
        verbose=False,
        status_report_interval=None  # Disable for short demo
    )

    consciousness = UnifiedTrustConsciousness(
        sensor_sources={
            'cpu': cpu_sensor,
            'memory': memory_sensor,
            'processes': process_sensor
        },
        action_handlers={
            'cpu': cpu_action,
            'memory': memory_action,
            'processes': process_action
        },
        trust_oracle=oracle,
        sensor_lct_ids={
            'cpu': 'lct:thor:cpu',
            'memory': 'lct:thor:memory',
            'processes': 'lct:thor:processes'
        },
        config=config
    )

    print(f"Session: {config.session_id}")
    print(f"Database: {config.db_path}")
    print(f"Duration: 30 seconds")
    print()
    print("Sensor Trust Scores:")
    print(f"  CPU: {oracle.get_trust_score('lct:thor:cpu'):.3f} (high)")
    print(f"  Memory: {oracle.get_trust_score('lct:thor:memory'):.3f} (medium)")
    print(f"  Processes: {oracle.get_trust_score('lct:thor:processes'):.3f} (low)")
    print()
    print("Starting unified trust-weighted consciousness...")
    print("(Press Ctrl+C for graceful shutdown)")
    print("="*80)

    # Run for 30 seconds
    consciousness.run(duration_seconds=30)

    print("\nFinal trust scores:")
    for lct_id in ['lct:thor:cpu', 'lct:thor:memory', 'lct:thor:processes']:
        score = oracle.get_full_score(lct_id)
        print(f"  {lct_id}: T3={score.t3_score():.3f}, V3={score.v3_score():.3f}, Composite={score.composite_score():.3f}")
        print(f"    Observations: {score.successful_observations}/{score.total_observations}")
