"""
Epistemic Memory Bridge - SAGE ↔ Epistemic Proprioception Integration

Connects SAGE's consciousness loop to the epistemic database for:
- Persistent memory across sessions
- High-salience observation capture
- Learning session recording
- Context retrieval from past experiences

Architecture:
    SAGE Consciousness → Bridge → Epistemic DB + Blockchain

Integration Points:
    1. store_observation() - High-salience obs → discoveries
    2. store_learning_session() - Episodes with quality scores
    3. query_relevant_context() - Semantic search for past experiences
    4. get_recent_episodes() - Session history

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
Status: Phase 1 Implementation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json

# Add memory repo to path
_memory_root = Path(__file__).parent.parent.parent.parent / "memory"
if str(_memory_root) not in sys.path:
    sys.path.insert(0, str(_memory_root))

try:
    # Import epistemic tools (will work if memory repo is available)
    from epistemic.tools.add_discovery import add_discovery_programmatic
    from epistemic.tools.episode_tracker import record_episode_programmatic
    from epistemic.query.search import query_context_programmatic
    EPISTEMIC_AVAILABLE = True
except ImportError:
    # Fallback mode - SAGE works without epistemic integration
    EPISTEMIC_AVAILABLE = False
    print("[EpistemicBridge] Warning: Epistemic tools not available. Running in fallback mode.")


@dataclass
class SNARCScores:
    """
    SNARC salience scores from SAGE.

    Maps to epistemic dimensions:
    - surprise → novelty (epistemic convention)
    - novelty → surprise (epistemic convention)
    - arousal → arousal
    - reward → confidence
    - conflict → validation status
    """
    surprise: float  # Unexpectedness (0-1)
    novelty: float   # First-time-ness (0-1)
    arousal: float   # Emotional energy (0-1)
    reward: float    # Value/importance (0-1)
    conflict: float  # Competing interpretations (0-1)

    def composite_score(self) -> float:
        """Calculate composite salience (weighted average)"""
        return (
            0.3 * self.surprise +
            0.25 * self.novelty +
            0.2 * self.arousal +
            0.15 * self.reward +
            0.1 * self.conflict
        )

    def to_epistemic_scores(self) -> Dict[str, float]:
        """
        Convert SNARC to epistemic dimensions.

        Note: Epistemic uses 'novelty' for surprise and 'surprise' for novelty.
        This is intentional per epistemic convention.
        """
        return {
            'surprise': self.novelty,  # Swap: epistemic 'surprise' = SNARC 'novelty'
            'novelty': self.surprise,  # Swap: epistemic 'novelty' = SNARC 'surprise'
            'arousal': self.arousal,
            'confidence': self.reward,
            'validation_status': self._conflict_to_validation()
        }

    def _conflict_to_validation(self) -> str:
        """Map conflict score to validation status"""
        if self.conflict < 0.3:
            return 'proven'      # Low conflict = high certainty
        elif self.conflict < 0.6:
            return 'tested'      # Medium conflict = tested but uncertain
        else:
            return 'speculative' # High conflict = multiple interpretations


@dataclass
class Observation:
    """Structured observation from SAGE consciousness"""
    description: str
    modality: str  # 'vision', 'audio', 'language', 'multimodal'
    snarc_scores: SNARCScores
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        """Generate brief summary for storage"""
        return self.description[:200]  # Truncate for title

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'description': self.description,
            'modality': self.modality,
            'snarc': asdict(self.snarc_scores),
            'timestamp': self.timestamp.isoformat(),
            'context': self.context or {}
        }


@dataclass
class LearningSession:
    """SAGE learning session data"""
    session_id: str
    started: datetime
    ended: datetime
    iterations: int
    plugins_used: List[str]
    high_salience_count: int
    convergence_failures: int
    trust_updates: int
    discoveries_witnessed: List[str]  # Knowledge IDs
    quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'session_id': self.session_id,
            'started': self.started.isoformat(),
            'ended': self.ended.isoformat(),
            'iterations': self.iterations,
            'plugins_used': self.plugins_used,
            'high_salience_count': self.high_salience_count,
            'convergence_failures': self.convergence_failures,
            'trust_updates': self.trust_updates,
            'discoveries': self.discoveries_witnessed,
            'quality': self.quality_score
        }


class EpistemicMemoryBridge:
    """
    Bridge between SAGE consciousness and epistemic database.

    Provides:
    - Persistent memory (discoveries, episodes)
    - Context retrieval (semantic search)
    - Blockchain witnessing (attribution)

    Usage:
        bridge = EpistemicMemoryBridge(machine='thor', project='sage')

        # Store high-salience observation
        obs = Observation(...)
        kid = bridge.store_observation(obs)

        # Record learning session
        session = LearningSession(...)
        eid = bridge.store_learning_session(session)

        # Query relevant context
        context = bridge.query_relevant_context("designing attention mechanism")
    """

    def __init__(
        self,
        machine: str = 'thor',
        project: str = 'sage',
        salience_threshold: float = 0.7,
        enable_witnessing: bool = True,
        witness_manager: Optional[Any] = None
    ):
        """
        Initialize epistemic memory bridge.

        Args:
            machine: Hardware entity ('thor', 'sprout', 'cbp', 'legion')
            project: Project context ('sage', 'hrm', etc.)
            salience_threshold: Minimum composite score for storage (0.7 = high)
            enable_witnessing: Witness on blockchain (default True)
            witness_manager: Optional external witness manager (Phase 3)
        """
        self.machine = machine
        self.project = project
        self.salience_threshold = salience_threshold
        self.enable_witnessing = enable_witnessing and EPISTEMIC_AVAILABLE

        # Initialize witness manager (Phase 3)
        if witness_manager:
            self.witness_manager = witness_manager
        elif enable_witnessing:
            from .witness_manager import create_witness_manager
            self.witness_manager = create_witness_manager(
                machine=machine,
                project=project,
                enable_witnessing=EPISTEMIC_AVAILABLE
            )
        else:
            self.witness_manager = None

        if not EPISTEMIC_AVAILABLE:
            print(f"[EpistemicBridge] Running in fallback mode (epistemic tools unavailable)")
            print(f"[EpistemicBridge] Observations will be stored locally only")
        else:
            print(f"[EpistemicBridge] Initialized for {machine}/{project}")
            print(f"[EpistemicBridge] Salience threshold: {salience_threshold}")
            print(f"[EpistemicBridge] Blockchain witnessing: {self.enable_witnessing}")

        # Local cache for fallback mode
        self.local_cache = {
            'observations': [],
            'sessions': [],
            'discoveries': []
        }

    def is_high_salience(self, snarc_scores: SNARCScores) -> bool:
        """Check if observation meets salience threshold"""
        return snarc_scores.composite_score() >= self.salience_threshold

    def store_observation(self, observation: Observation) -> Optional[str]:
        """
        Store high-salience observation as discovery.

        Args:
            observation: Observation with SNARC scores

        Returns:
            Knowledge ID if stored, None if below threshold
        """
        if not self.is_high_salience(observation.snarc_scores):
            return None

        # Convert SNARC to epistemic scores
        epistemic_scores = observation.snarc_scores.to_epistemic_scores()

        if EPISTEMIC_AVAILABLE:
            try:
                # Store in epistemic database
                discovery_id = add_discovery_programmatic(
                    title=observation.summary(),
                    summary=observation.description,
                    domain=f"{self.project}/observations/{observation.modality}",
                    type='core-insights',
                    tags=['sage', 'observation', self.machine, observation.modality],
                    **epistemic_scores
                )

                # Witness on blockchain (Phase 3)
                if self.witness_manager:
                    self.witness_manager.witness_discovery(
                        knowledge_id=discovery_id,
                        title=observation.summary(),
                        domain=f"{self.project}/observations",
                        quality=observation.snarc_scores.composite_score(),
                        tags=[f'sage', observation.modality]
                    )

                print(f"[EpistemicBridge] ✅ Stored observation: {discovery_id[:12]}...")
                return discovery_id

            except Exception as e:
                print(f"[EpistemicBridge] ⚠️ Failed to store observation: {e}")
                # Fall through to local cache

        # Fallback: Store locally
        obs_record = {
            'id': f"local-obs-{len(self.local_cache['observations'])}",
            'observation': observation.to_dict(),
            'stored_at': datetime.now(timezone.utc).isoformat()
        }
        self.local_cache['observations'].append(obs_record)
        return obs_record['id']

    def store_learning_session(self, session: LearningSession) -> Optional[str]:
        """
        Store SAGE learning session as episode.

        Args:
            session: Learning session data with quality score

        Returns:
            Episode ID if stored, None on failure
        """
        if EPISTEMIC_AVAILABLE:
            try:
                # Record episode in epistemic database
                episode_id = record_episode_programmatic(
                    session_id=session.session_id,
                    machine=self.machine,
                    project=self.project,
                    task_type='consciousness_loop',
                    started=session.started,
                    ended=session.ended,
                    discoveries=session.high_salience_count,
                    failures=session.convergence_failures,
                    shifts=session.trust_updates,
                    commits=0,  # SAGE doesn't use git during runtime
                    knowledge_ids=session.discoveries_witnessed
                )

                # Witness episode on blockchain (Phase 3)
                if self.witness_manager:
                    self.witness_manager.witness_episode(
                        episode_id=episode_id,
                        quality=session.quality_score,
                        discoveries=session.high_salience_count,
                        failures=session.convergence_failures,
                        shifts=session.trust_updates
                    )

                print(f"[EpistemicBridge] ✅ Stored session: {session.session_id}")
                print(f"[EpistemicBridge]    Quality: {session.quality_score:.2f}")
                print(f"[EpistemicBridge]    Discoveries: {session.high_salience_count}")
                return episode_id

            except Exception as e:
                print(f"[EpistemicBridge] ⚠️ Failed to store session: {e}")
                # Fall through to local cache

        # Fallback: Store locally
        session_record = {
            'id': f"local-session-{len(self.local_cache['sessions'])}",
            'session': session.to_dict(),
            'stored_at': datetime.now(timezone.utc).isoformat()
        }
        self.local_cache['sessions'].append(session_record)
        return session_record['id']

    def query_relevant_context(
        self,
        current_situation: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Query epistemic DB for relevant past experiences.

        Args:
            current_situation: Description of current situation
            limit: Maximum results to return

        Returns:
            Dictionary with similar_episodes, relevant_skills, known_failures
        """
        if EPISTEMIC_AVAILABLE:
            try:
                results = query_context_programmatic(
                    text=current_situation,
                    project=self.project,
                    limit=limit
                )

                return {
                    'similar_episodes': results.get('episodes', []),
                    'relevant_skills': results.get('skills', []),
                    'known_failures': results.get('failures', []),
                    'cross_refs': results.get('cross_references', [])
                }

            except Exception as e:
                print(f"[EpistemicBridge] ⚠️ Context query failed: {e}")
                # Fall through to empty result

        # Fallback: Return empty context
        return {
            'similar_episodes': [],
            'relevant_skills': [],
            'known_failures': [],
            'cross_refs': []
        }

    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent SAGE learning sessions.

        Args:
            limit: Maximum episodes to return

        Returns:
            List of episode dictionaries
        """
        if EPISTEMIC_AVAILABLE:
            try:
                # Query recent episodes for this machine/project
                context = self.query_relevant_context("", limit=limit)
                return context.get('similar_episodes', [])

            except Exception as e:
                print(f"[EpistemicBridge] ⚠️ Episode query failed: {e}")

        # Fallback: Return local sessions
        return self.local_cache['sessions'][-limit:]

    def summarize_integration_status(self) -> Dict[str, Any]:
        """
        Get integration status summary.

        Returns:
            Dictionary with counts and status
        """
        if EPISTEMIC_AVAILABLE:
            status = "connected"
            observations_count = "managed by epistemic DB"
            sessions_count = "managed by epistemic DB"
        else:
            status = "fallback (local only)"
            observations_count = len(self.local_cache['observations'])
            sessions_count = len(self.local_cache['sessions'])

        return {
            'status': status,
            'machine': self.machine,
            'project': self.project,
            'salience_threshold': self.salience_threshold,
            'witnessing_enabled': self.enable_witnessing,
            'observations_stored': observations_count,
            'sessions_recorded': sessions_count,
            'epistemic_available': EPISTEMIC_AVAILABLE
        }


# Convenience function for SAGE integration
def create_bridge(
    machine: str = 'thor',
    project: str = 'sage',
    **kwargs
) -> EpistemicMemoryBridge:
    """
    Create epistemic memory bridge for SAGE.

    Args:
        machine: Hardware entity
        project: Project context
        **kwargs: Additional bridge configuration

    Returns:
        Configured EpistemicMemoryBridge instance
    """
    return EpistemicMemoryBridge(
        machine=machine,
        project=project,
        **kwargs
    )
