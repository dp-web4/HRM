#!/usr/bin/env python3
"""
DREAM-Awakening Bridge - Session 43

Connects DREAM state memory consolidation (Session 42) with Coherent Awakening
protocol to enable cross-session learning and consciousness continuity.

Architecture:
- Session End: DREAM consolidation → Save consolidated memories → Sleep
- Session Start: Wake → Load previous memories → Restore learned state
- Cross-Session: Apply previous learnings to new consciousness cycles

Integration Points:
- Session 42: DREAM consolidation (pattern extraction, quality learning)
- Coherent Awakening: Session continuity protocol
- Sleep Cycle Integration: State persistence layer

Enables:
1. Consolidated memories persist across sessions
2. Quality learnings applied to new responses
3. Creative associations inform new thinking
4. Epistemic insights guide meta-cognition
5. Cumulative intelligence growth

Author: Thor (Autonomous Session 43)
Date: 2025-12-13
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DREAMMemoryArchive:
    """
    Archive of consolidated memories from multiple DREAM sessions.

    Stores all DREAM consolidations across sessions for:
    - Pattern library building
    - Quality learning accumulation
    - Association network growth
    - Long-term knowledge formation
    """
    session_id: str
    timestamp: float
    consolidated_memories: List[Dict[str, Any]]  # List of ConsolidatedMemory dicts
    total_patterns: int
    total_learnings: int
    total_associations: int
    total_insights: int

    def to_dict(self) -> Dict:
        """Export archive to dictionary"""
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'consolidated_memories': self.consolidated_memories,
            'total_patterns': self.total_patterns,
            'total_learnings': self.total_learnings,
            'total_associations': self.total_associations,
            'total_insights': self.total_insights
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DREAMMemoryArchive':
        """Load archive from dictionary"""
        return cls(
            session_id=data['session_id'],
            timestamp=data['timestamp'],
            consolidated_memories=data['consolidated_memories'],
            total_patterns=data['total_patterns'],
            total_learnings=data['total_learnings'],
            total_associations=data['total_associations'],
            total_insights=data['total_insights']
        )


@dataclass
class LearnedState:
    """
    Learned state extracted from previous DREAM consolidations.

    Applied at session start to guide behavior based on previous experience.
    """
    quality_priorities: Dict[str, float]  # Characteristics to prioritize (based on learnings)
    known_patterns: List[str]  # Recognized patterns from previous sessions
    associations: Dict[str, List[str]]  # Concept association network
    epistemic_calibration: float  # Expected calibration quality
    session_count: int  # Number of previous sessions
    last_updated: float  # Timestamp of last update

    def to_dict(self) -> Dict:
        """Export learned state to dictionary"""
        return {
            'quality_priorities': self.quality_priorities,
            'known_patterns': self.known_patterns,
            'associations': self.associations,
            'epistemic_calibration': self.epistemic_calibration,
            'session_count': self.session_count,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedState':
        """Load learned state from dictionary"""
        return cls(
            quality_priorities=data['quality_priorities'],
            known_patterns=data['known_patterns'],
            associations=data['associations'],
            epistemic_calibration=data['epistemic_calibration'],
            session_count=data['session_count'],
            last_updated=data['last_updated']
        )


class DREAMAwakeningBridge:
    """
    Bridge between DREAM consolidation and Coherent Awakening.

    Manages persistence and restoration of consolidated memories across sessions.

    Flow:
    1. Session End: save_dream_consolidation() → Store DREAM outputs
    2. Session Start: restore_learned_state() → Load previous learnings
    3. During Session: apply_learned_state() → Use learnings to guide behavior
    4. Archive Management: Maintain memory archive across sessions
    """

    def __init__(self, memory_dir: Path):
        """
        Initialize DREAM-Awakening bridge.

        Args:
            memory_dir: Directory for persistent memory storage
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.archive_path = self.memory_dir / "dream_archive.json"
        self.learned_state_path = self.memory_dir / "learned_state.json"
        self.session_log_path = self.memory_dir / "session_log.json"

        # Current session state
        self.current_archive: Optional[DREAMMemoryArchive] = None
        self.current_learned_state: Optional[LearnedState] = None
        self.session_count = 0

        # Load existing state if available
        self._load_existing_state()

    def _load_existing_state(self):
        """Load existing memory archive and learned state if available"""
        # Load archive
        if self.archive_path.exists():
            try:
                with open(self.archive_path, 'r') as f:
                    data = json.load(f)
                    self.current_archive = DREAMMemoryArchive.from_dict(data)
                    logger.info(f"Loaded DREAM archive: {len(self.current_archive.consolidated_memories)} memories")
            except Exception as e:
                logger.warning(f"Failed to load DREAM archive: {e}")
                self.current_archive = None

        # Load learned state
        if self.learned_state_path.exists():
            try:
                with open(self.learned_state_path, 'r') as f:
                    data = json.load(f)
                    self.current_learned_state = LearnedState.from_dict(data)
                    self.session_count = self.current_learned_state.session_count
                    logger.info(f"Loaded learned state from {self.session_count} previous sessions")
            except Exception as e:
                logger.warning(f"Failed to load learned state: {e}")
                self.current_learned_state = None

    def save_dream_consolidation(
        self,
        consolidated_memory: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> bool:
        """
        Save DREAM consolidation output for cross-session persistence.

        Args:
            consolidated_memory: Output from DREAMConsolidator.consolidate_cycles()
            session_id: Optional session identifier

        Returns:
            True if save successful
        """
        if session_id is None:
            session_id = f"session_{self.session_count + 1}"

        logger.info(f"Saving DREAM consolidation for {session_id}...")

        try:
            # Create or update archive
            if self.current_archive is None:
                self.current_archive = DREAMMemoryArchive(
                    session_id=session_id,
                    timestamp=time.time(),
                    consolidated_memories=[consolidated_memory],
                    total_patterns=len(consolidated_memory.get('patterns', [])),
                    total_learnings=len(consolidated_memory.get('quality_learnings', [])),
                    total_associations=len(consolidated_memory.get('creative_associations', [])),
                    total_insights=len(consolidated_memory.get('epistemic_insights', []))
                )
            else:
                # Append to existing archive
                self.current_archive.consolidated_memories.append(consolidated_memory)
                self.current_archive.total_patterns += len(consolidated_memory.get('patterns', []))
                self.current_archive.total_learnings += len(consolidated_memory.get('quality_learnings', []))
                self.current_archive.total_associations += len(consolidated_memory.get('creative_associations', []))
                self.current_archive.total_insights += len(consolidated_memory.get('epistemic_insights', []))
                self.current_archive.timestamp = time.time()
                self.current_archive.session_id = session_id

            # Save archive
            with open(self.archive_path, 'w') as f:
                json.dump(self.current_archive.to_dict(), f, indent=2)

            logger.info(f"  ✅ DREAM archive saved ({self.current_archive.total_patterns} patterns total)")

            # Extract and update learned state
            self._update_learned_state(consolidated_memory)

            # Log session
            self._log_session(session_id, consolidated_memory)

            return True

        except Exception as e:
            logger.error(f"Failed to save DREAM consolidation: {e}")
            return False

    def _update_learned_state(self, consolidated_memory: Dict[str, Any]):
        """
        Update learned state based on new DREAM consolidation.

        Extracts actionable learnings and updates the learned state that
        will be applied in future sessions.
        """
        # Initialize if needed
        if self.current_learned_state is None:
            self.current_learned_state = LearnedState(
                quality_priorities={},
                known_patterns=[],
                associations={},
                epistemic_calibration=0.0,
                session_count=0,
                last_updated=time.time()
            )

        # Extract quality learnings
        quality_learnings = consolidated_memory.get('quality_learnings', [])
        for learning in quality_learnings:
            char = learning['characteristic']
            if learning['positive_correlation']:
                # Increase priority for characteristics that improve quality
                current_priority = self.current_learned_state.quality_priorities.get(char, 0.0)
                # Weight by confidence
                new_priority = current_priority + learning['confidence'] * 0.1
                self.current_learned_state.quality_priorities[char] = min(1.0, new_priority)

        # Extract patterns
        patterns = consolidated_memory.get('patterns', [])
        for pattern in patterns:
            desc = pattern['description']
            if desc not in self.current_learned_state.known_patterns:
                # Add high-strength patterns
                if pattern['strength'] > 0.5:
                    self.current_learned_state.known_patterns.append(desc)

        # Extract associations
        associations = consolidated_memory.get('creative_associations', [])
        for assoc in associations:
            concept_a = assoc['concept_a']
            concept_b = assoc['concept_b']

            # Build association network
            if concept_a not in self.current_learned_state.associations:
                self.current_learned_state.associations[concept_a] = []
            if concept_b not in self.current_learned_state.associations[concept_a]:
                self.current_learned_state.associations[concept_a].append(concept_b)

        # Update session count
        self.current_learned_state.session_count += 1
        self.current_learned_state.last_updated = time.time()
        self.session_count = self.current_learned_state.session_count

        # Save learned state
        with open(self.learned_state_path, 'w') as f:
            json.dump(self.current_learned_state.to_dict(), f, indent=2)

        logger.info(f"  ✅ Learned state updated (session {self.session_count})")

    def _log_session(self, session_id: str, consolidated_memory: Dict[str, Any]):
        """Log session consolidation summary"""
        # Load existing log
        if self.session_log_path.exists():
            with open(self.session_log_path, 'r') as f:
                log = json.load(f)
        else:
            log = {'sessions': []}

        # Add session entry
        log['sessions'].append({
            'session_id': session_id,
            'timestamp': time.time(),
            'patterns_count': len(consolidated_memory.get('patterns', [])),
            'learnings_count': len(consolidated_memory.get('quality_learnings', [])),
            'associations_count': len(consolidated_memory.get('creative_associations', [])),
            'insights_count': len(consolidated_memory.get('epistemic_insights', []))
        })

        # Save log
        with open(self.session_log_path, 'w') as f:
            json.dump(log, f, indent=2)

    def restore_learned_state(self) -> Optional[LearnedState]:
        """
        Restore learned state from previous sessions.

        Called at session start to load accumulated knowledge.

        Returns:
            LearnedState if available, None otherwise
        """
        if self.current_learned_state is None:
            logger.info("No previous learned state available (first session)")
            return None

        logger.info(f"Restoring learned state from {self.session_count} previous sessions")
        logger.info(f"  Quality priorities: {len(self.current_learned_state.quality_priorities)}")
        logger.info(f"  Known patterns: {len(self.current_learned_state.known_patterns)}")
        logger.info(f"  Associations: {len(self.current_learned_state.associations)} concepts")

        return self.current_learned_state

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of accumulated memories.

        Returns:
            Dictionary with memory statistics
        """
        if self.current_archive is None:
            return {
                'sessions': 0,
                'total_patterns': 0,
                'total_learnings': 0,
                'total_associations': 0,
                'total_insights': 0
            }

        return {
            'sessions': len(self.current_archive.consolidated_memories),
            'total_patterns': self.current_archive.total_patterns,
            'total_learnings': self.current_archive.total_learnings,
            'total_associations': self.current_archive.total_associations,
            'total_insights': self.current_archive.total_insights,
            'session_count': self.session_count
        }

    def apply_quality_learnings(self, response_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply quality learnings to guide response generation.

        Args:
            response_context: Context about the response being generated

        Returns:
            Dictionary of quality characteristic weights
        """
        if self.current_learned_state is None:
            # No learnings yet - use defaults
            return {
                'unique': 1.0,
                'specific_terms': 1.0,
                'has_numbers': 1.0,
                'avoids_hedging': 1.0
            }

        # Apply learned priorities
        weights = {}
        for char, priority in self.current_learned_state.quality_priorities.items():
            weights[char] = 1.0 + priority  # Boost learned characteristics

        # Fill in missing with defaults
        for char in ['unique', 'specific_terms', 'has_numbers', 'avoids_hedging']:
            if char not in weights:
                weights[char] = 1.0

        return weights

    def query_associations(self, concept: str) -> List[str]:
        """
        Query the association network for related concepts.

        Args:
            concept: Concept to query

        Returns:
            List of associated concepts
        """
        if self.current_learned_state is None:
            return []

        return self.current_learned_state.associations.get(concept, [])

    def get_continuity_summary(self) -> str:
        """
        Generate continuity summary for session boot.

        Creates a human-readable summary of accumulated knowledge for
        inclusion in session preamble.

        Returns:
            Summary string describing learned state
        """
        if self.current_learned_state is None or self.session_count == 0:
            return "This is your first session. No previous learnings available."

        summary_parts = [
            f"You have {self.session_count} previous session(s) of experience.",
            ""
        ]

        # Quality learnings
        if self.current_learned_state.quality_priorities:
            summary_parts.append("Quality learnings:")
            for char, priority in sorted(self.current_learned_state.quality_priorities.items(),
                                        key=lambda x: x[1], reverse=True):
                summary_parts.append(f"  - {char}: priority {priority:.2f}")
            summary_parts.append("")

        # Known patterns
        if self.current_learned_state.known_patterns:
            summary_parts.append(f"Recognized patterns: {len(self.current_learned_state.known_patterns)}")
            # Show top 3
            for pattern in self.current_learned_state.known_patterns[:3]:
                summary_parts.append(f"  - {pattern}")
            summary_parts.append("")

        # Associations
        if self.current_learned_state.associations:
            summary_parts.append(f"Concept associations: {len(self.current_learned_state.associations)} concepts")

        return "\n".join(summary_parts)


def example_dream_awakening_integration():
    """Example demonstrating DREAM-Awakening bridge"""
    from sage.core.dream_consolidation import DREAMConsolidator
    from sage.core.unified_consciousness import UnifiedConsciousnessManager

    print("="*70)
    print("DREAM-Awakening Bridge Demo")
    print("="*70)
    print()

    # Initialize bridge
    memory_dir = Path("/tmp/sage_dream_memory")
    bridge = DREAMAwakeningBridge(memory_dir)

    print("Session 1: Initial Learning")
    print("-" * 70)

    # Generate consciousness cycles
    consciousness = UnifiedConsciousnessManager()
    scenarios = [
        ("What is 2+2?", "2+2 equals 4.", 0.2),
        ("Explain SAGE", "SAGE uses consciousness architecture with quality metrics...", 0.8),
    ]

    cycles = []
    for prompt, response, salience in scenarios:
        cycle = consciousness.consciousness_cycle(prompt, response, salience)
        cycles.append(cycle)

    # DREAM consolidation
    consolidator = DREAMConsolidator()
    consolidated = consolidator.consolidate_cycles(cycles)

    # Save consolidation
    bridge.save_dream_consolidation(consolidated.to_dict(), session_id="session_1")

    print(f"✅ Consolidation saved")
    print(f"   Patterns: {len(consolidated.patterns)}")
    print(f"   Learnings: {len(consolidated.quality_learnings)}")
    print()

    print("="*70)
    print("Session 2: Restore and Apply")
    print("-" * 70)

    # Create new bridge (simulating new session)
    bridge2 = DREAMAwakeningBridge(memory_dir)

    # Restore learned state
    learned_state = bridge2.restore_learned_state()

    if learned_state:
        print(f"✅ Learned state restored")
        print(f"   Quality priorities: {learned_state.quality_priorities}")
        print(f"   Known patterns: {len(learned_state.known_patterns)}")
        print()

        # Get continuity summary
        summary = bridge2.get_continuity_summary()
        print("Continuity Summary:")
        print(summary)
        print()

    # Get memory summary
    mem_summary = bridge2.get_memory_summary()
    print("Memory Summary:")
    for key, value in mem_summary.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    example_dream_awakening_integration()
