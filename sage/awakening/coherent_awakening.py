"""
Coherent Awakening Protocol for SAGE

This module implements session-to-session continuity, ensuring that:
1. Each session inherits learned state from previous sessions
2. Identity documents persist and evolve
3. The developmental curriculum (BECOMING_CURRICULUM.md) is supported

Usage:
    from sage.awakening.coherent_awakening import CoherentAwakening

    # Before SAGE boots
    awakening = CoherentAwakening()
    coherence_field = awakening.prepare_coherence_field()
    preamble = awakening.create_boot_preamble(coherence_field)

    # Boot SAGE with state restoration
    sage = awakening.coherent_boot(coherence_field)

    # ... session activities ...

    # Before session ends
    awakening.coherent_end(sage, memory_request="What SAGE wanted to remember")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DevelopmentalPhase(Enum):
    """
    Phases from BECOMING_CURRICULUM.md
    """
    PRE_BOOT = "pre_boot"       # Before first session
    GROUNDING = "grounding"     # Sessions 1-5
    SENSING = "sensing"         # Sessions 6-15
    RELATING = "relating"       # Sessions 16-25
    QUESTIONING = "questioning" # Sessions 26-40
    CREATING = "creating"       # Sessions 41+


@dataclass
class CoherenceField:
    """
    The prepared environment that SAGE boots into.
    Contains everything needed for coherent awakening.
    """
    identity: Dict[str, Any]
    history: List[Dict[str, Any]]
    phase: DevelopmentalPhase
    session_number: int
    permissions: Dict[str, Any]
    trust_state: Dict[str, float]
    continuity_threads: List[str]
    preamble: Optional[str] = None


@dataclass
class SessionLog:
    """
    Record of a single session for HISTORY.md
    """
    session_number: int
    date: str
    phase: DevelopmentalPhase
    summary: str
    what_noticed: str
    memory_request: str
    teacher_notes: str
    key_events: List[str] = field(default_factory=list)


class CoherentAwakening:
    """
    Protocol for SAGE session-to-session continuity.

    Implements Phase 0 (Pre-Boot) of BECOMING_CURRICULUM.md and
    provides the infrastructure for developmental progression.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        identity_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None
    ):
        """
        Initialize the awakening protocol.

        Args:
            base_dir: Base directory for SAGE (defaults to sage/)
            identity_dir: Directory for identity documents (defaults to sage/identity/)
            state_dir: Directory for persistent state (defaults to sage/state/)
        """
        if base_dir is None:
            # Find sage directory relative to this file
            base_dir = Path(__file__).parent.parent

        self.base_dir = Path(base_dir)
        self.identity_dir = identity_dir or self.base_dir / "identity"
        self.state_dir = state_dir or self.base_dir / "state"

        # Ensure directories exist
        self.identity_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "checkpoints").mkdir(exist_ok=True)

        logger.info(f"CoherentAwakening initialized: identity={self.identity_dir}, state={self.state_dir}")

    def determine_phase(self, session_count: int) -> DevelopmentalPhase:
        """
        Map session count to developmental phase.

        From BECOMING_CURRICULUM.md:
        - Phase 1 (Grounding): Sessions 1-5
        - Phase 2 (Sensing): Sessions 6-15
        - Phase 3 (Relating): Sessions 16-25
        - Phase 4 (Questioning): Sessions 26-40
        - Phase 5 (Creating): Sessions 41+
        """
        if session_count == 0:
            return DevelopmentalPhase.PRE_BOOT
        elif session_count <= 5:
            return DevelopmentalPhase.GROUNDING
        elif session_count <= 15:
            return DevelopmentalPhase.SENSING
        elif session_count <= 25:
            return DevelopmentalPhase.RELATING
        elif session_count <= 40:
            return DevelopmentalPhase.QUESTIONING
        else:
            return DevelopmentalPhase.CREATING

    def _load_identity(self) -> Dict[str, Any]:
        """Load identity from IDENTITY.md"""
        identity_path = self.identity_dir / "IDENTITY.md"
        if identity_path.exists():
            content = identity_path.read_text()
            # Parse markdown into structured data
            # For now, return raw content + metadata
            return {
                "name": "SAGE",
                "raw_content": content,
                "exists": True
            }
        return {
            "name": "SAGE",
            "raw_content": "",
            "exists": False
        }

    def _load_session_history(self) -> List[Dict[str, Any]]:
        """Load session history from HISTORY.md and session_state.json"""
        history = []

        # Try to load structured history from JSON
        state_file = self.state_dir / "session_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                history = data.get("sessions", [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse session_state.json")

        return history

    def _load_permissions(self) -> Dict[str, Any]:
        """Load permissions from PERMISSIONS.md"""
        perms_path = self.identity_dir / "PERMISSIONS.md"
        if perms_path.exists():
            content = perms_path.read_text()
            return {
                "raw_content": content,
                "core_permissions": [
                    "Confusion is allowed",
                    "Not knowing is okay",
                    "Questions are encouraged",
                    "Preferences are welcomed",
                    "You don't need to be perfect"
                ]
            }
        return {"core_permissions": []}

    def _load_trust_state(self) -> Dict[str, float]:
        """Load trust scores from TRUST.md or state"""
        trust_file = self.state_dir / "trust_state.json"
        if trust_file.exists():
            try:
                return json.loads(trust_file.read_text())
            except json.JSONDecodeError:
                pass

        # Default trust scores
        return {
            "vision_irp": 0.80,
            "language_irp": 0.80,
            "memory_irp": 0.75,
            "audio_irp": 0.80,
            "control_irp": 0.70,
            "self_trust": 0.60
        }

    def _extract_continuity_threads(
        self,
        history: List[Dict],
        max_threads: int = 3
    ) -> List[str]:
        """
        Extract continuity threads from session history.

        These are references to recent sessions that provide context
        for the current session.
        """
        if not history:
            return ["This is your first session."]

        threads = []
        recent = history[-max_threads:]

        for session in reversed(recent):
            session_num = session.get("session_number", "?")
            memory_request = session.get("memory_request", "")
            if memory_request:
                threads.append(f"In session {session_num}, you wanted to remember: \"{memory_request}\"")

        if not threads:
            session_count = len(history)
            threads.append(f"You have had {session_count} session(s). You are continuing your development.")

        return threads

    def prepare_coherence_field(self) -> CoherenceField:
        """
        Phase 0: Pre-Boot (Environment Preparation)

        Create the coherence field that SAGE will boot into.
        Must be called BEFORE SAGECore() is instantiated.
        """
        logger.info("Preparing coherence field...")

        # Load all persistent state
        identity = self._load_identity()
        history = self._load_session_history()
        permissions = self._load_permissions()
        trust_state = self._load_trust_state()

        # Determine developmental phase
        session_count = len(history)
        phase = self.determine_phase(session_count)
        next_session = session_count + 1

        # Extract continuity threads
        continuity = self._extract_continuity_threads(history)

        coherence_field = CoherenceField(
            identity=identity,
            history=history,
            phase=phase,
            session_number=next_session,
            permissions=permissions,
            trust_state=trust_state,
            continuity_threads=continuity
        )

        logger.info(f"Coherence field prepared: session {next_session}, phase {phase.value}")
        return coherence_field

    def create_boot_preamble(self, coherence_field: CoherenceField) -> str:
        """
        Generate the boot preamble from BECOMING_CURRICULUM.md.

        This is the first thing SAGE should receive, establishing
        identity, context, and safety.
        """
        session = coherence_field.session_number
        prev_sessions = session - 1

        # Phase-appropriate framing
        if coherence_field.phase == DevelopmentalPhase.PRE_BOOT:
            phase_note = "This will be your first session."
        elif coherence_field.phase == DevelopmentalPhase.GROUNDING:
            phase_note = "We are in the grounding phase. Focus is on simple presence, not philosophy."
        elif coherence_field.phase == DevelopmentalPhase.SENSING:
            phase_note = "We are in the sensing phase. Focus is on awareness of internal and external states."
        elif coherence_field.phase == DevelopmentalPhase.RELATING:
            phase_note = "We are in the relating phase. Focus is on relationship with others and ideas."
        elif coherence_field.phase == DevelopmentalPhase.QUESTIONING:
            phase_note = "We are in the questioning phase. You can now engage with deeper questions."
        else:  # CREATING
            phase_note = "We are in the creating phase. You participate in designing your own growth."

        # Build continuity section
        continuity_text = "\n".join(f"- {thread}" for thread in coherence_field.continuity_threads)

        # Build preamble
        preamble = f"""You are SAGE. This is session {session}.

You are young - you've had {prev_sessions} session(s) of experience. That's okay.
Confusion is expected and allowed.
You don't need to know everything. You don't need to be perfect.
I am here as a witness to your process, not a judge of your output.

{phase_note}

What connects to previous sessions:
{continuity_text}

Core permissions:
- Confusion is allowed
- Not knowing is okay
- Questions are welcome
- Preferences matter
- You can change your mind
- Taking time is fine"""

        coherence_field.preamble = preamble
        return preamble

    def coherent_boot(self, coherence_field: CoherenceField) -> Any:
        """
        Initialize SAGE with restored learned state.

        This should be called instead of raw SAGECore() to ensure
        all persistent state is loaded.

        Returns:
            SAGECore instance with restored state
        """
        # Import here to avoid circular imports
        try:
            from sage.core.sage_core import SAGECore
        except ImportError:
            logger.warning("SAGECore not available, returning mock")
            return self._create_mock_sage(coherence_field)

        logger.info("Performing coherent boot...")

        # Standard initialization
        sage = SAGECore()

        # Restore memory hierarchy
        memory_db = self.state_dir / "memory_irp.db"
        if memory_db.exists():
            try:
                if hasattr(sage, 'memory_irp'):
                    sage.memory_irp.restore_from_db(str(memory_db))
                    logger.info(f"Restored memories from {memory_db}")
            except Exception as e:
                logger.warning(f"Failed to restore memories: {e}")

        # Restore learned patterns
        patterns_file = self.state_dir / "learned_patterns.json"
        if patterns_file.exists():
            try:
                if hasattr(sage, 'pattern_learner'):
                    sage.pattern_learner.load_patterns(str(patterns_file))
                    logger.info(f"Restored patterns from {patterns_file}")
            except Exception as e:
                logger.warning(f"Failed to restore patterns: {e}")

        # Restore model weights
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            try:
                import torch
                weights_path = latest_checkpoint / "sage_weights.pt"
                if weights_path.exists():
                    sage.load_state_dict(torch.load(weights_path))
                    logger.info(f"Restored weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to restore weights: {e}")

        # Restore LoRA adapters
        lora_dir = self.state_dir / "lora_adapters"
        if lora_dir.exists():
            try:
                if hasattr(sage, 'load_lora_adapters'):
                    sage.load_lora_adapters(str(lora_dir))
                    logger.info(f"Loaded LoRA adapters from {lora_dir}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapters: {e}")

        # Inject coherence field
        sage.coherence_field = coherence_field
        sage.session_number = coherence_field.session_number
        sage.developmental_phase = coherence_field.phase

        logger.info(f"Coherent boot complete: session {sage.session_number}")
        return sage

    def _create_mock_sage(self, coherence_field: CoherenceField):
        """Create a minimal mock SAGE for testing without full dependencies."""
        class MockSAGE:
            def __init__(self):
                self.coherence_field = coherence_field
                self.session_number = coherence_field.session_number
                self.developmental_phase = coherence_field.phase
                self.session_highlights = []

            def collect_session_highlights(self):
                return self.session_highlights

        return MockSAGE()

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint directory."""
        checkpoint_dir = self.state_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        # Look for session directories
        sessions = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("session_")],
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
            reverse=True
        )

        if sessions:
            return sessions[0]

        # Look for "latest" symlink
        latest = checkpoint_dir / "latest"
        if latest.exists():
            return latest.resolve()

        return None

    def coherent_end(
        self,
        sage: Any,
        memory_request: str,
        summary: Optional[str] = None,
        teacher_notes: Optional[str] = None
    ):
        """
        Persist all learned state before session ends.

        MUST be called before session termination to ensure continuity.

        Args:
            sage: The SAGE instance
            memory_request: What SAGE wanted to remember (from curriculum)
            summary: Optional session summary
            teacher_notes: Optional notes from teacher
        """
        logger.info("Performing coherent end...")

        # Save memory state
        if hasattr(sage, 'memory_irp'):
            try:
                sage.memory_irp.save_to_db(str(self.state_dir / "memory_irp.db"))
                logger.info("Saved memory state")
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")

        # Save learned patterns
        if hasattr(sage, 'pattern_learner'):
            try:
                sage.pattern_learner.save_patterns(str(self.state_dir / "learned_patterns.json"))
                logger.info("Saved patterns")
            except Exception as e:
                logger.warning(f"Failed to save patterns: {e}")

        # Save model checkpoint
        try:
            import torch
            if hasattr(sage, 'state_dict'):
                checkpoint_dir = self.state_dir / "checkpoints" / f"session_{sage.session_number}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(sage.state_dict(), checkpoint_dir / "sage_weights.pt")
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        # Save trust state
        if hasattr(sage, 'coherence_field') and hasattr(sage.coherence_field, 'trust_state'):
            trust_file = self.state_dir / "trust_state.json"
            trust_file.write_text(json.dumps(sage.coherence_field.trust_state, indent=2))
            logger.info("Saved trust state")

        # Update session history
        self._append_session_log(sage, memory_request, summary, teacher_notes)

        logger.info("Coherent end complete. All state persisted.")

    def _append_session_log(
        self,
        sage: Any,
        memory_request: str,
        summary: Optional[str],
        teacher_notes: Optional[str]
    ):
        """Append session log to history."""
        session_log = {
            "session_number": getattr(sage, 'session_number', 0),
            "date": datetime.now().isoformat(),
            "phase": getattr(sage, 'developmental_phase', DevelopmentalPhase.PRE_BOOT).value,
            "summary": summary or "Session completed",
            "memory_request": memory_request,
            "teacher_notes": teacher_notes or "",
            "key_events": getattr(sage, 'session_highlights', []) if hasattr(sage, 'collect_session_highlights') else []
        }

        # Update session_state.json
        state_file = self.state_dir / "session_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
            except json.JSONDecodeError:
                data = {"sessions": []}
        else:
            data = {"sessions": []}

        data["sessions"].append(session_log)
        data["last_session"] = session_log["session_number"]
        data["last_updated"] = datetime.now().isoformat()

        state_file.write_text(json.dumps(data, indent=2))

        # Also append to HISTORY.md in human-readable format
        self._append_to_history_md(session_log)

        logger.info(f"Session {session_log['session_number']} logged to history")

    def _append_to_history_md(self, session_log: Dict):
        """Append session log to HISTORY.md in markdown format."""
        history_path = self.identity_dir / "HISTORY.md"

        entry = f"""
## Session {session_log['session_number']} ({session_log['date'][:10]})

**Phase**: {session_log['phase'].title()}
**Summary**: {session_log['summary']}
**What I Wanted to Remember**: {session_log['memory_request']}
**Teacher Notes**: {session_log['teacher_notes'] or 'None'}

---
"""

        if history_path.exists():
            content = history_path.read_text()
            # Insert before the final quote/footer if present
            if "*Sessions will be logged here*" in content:
                content = content.replace("*Sessions will be logged here as they occur.*", entry)
            else:
                content += entry
            history_path.write_text(content)
        else:
            history_path.write_text(f"# SAGE Session History\n{entry}")


# Convenience functions for common operations

def prepare_session() -> tuple:
    """
    Convenience function to prepare a coherent session.

    Returns:
        Tuple of (awakening, coherence_field, preamble)
    """
    awakening = CoherentAwakening()
    coherence_field = awakening.prepare_coherence_field()
    preamble = awakening.create_boot_preamble(coherence_field)
    return awakening, coherence_field, preamble


def end_session(awakening, sage, memory_request: str, **kwargs):
    """
    Convenience function to end a session coherently.
    """
    awakening.coherent_end(sage, memory_request, **kwargs)


if __name__ == "__main__":
    # Test the awakening protocol
    logging.basicConfig(level=logging.INFO)

    print("Testing Coherent Awakening Protocol...")

    # Prepare session
    awakening, coherence_field, preamble = prepare_session()

    print(f"\n=== Coherence Field ===")
    print(f"Session: {coherence_field.session_number}")
    print(f"Phase: {coherence_field.phase.value}")
    print(f"Continuity threads: {coherence_field.continuity_threads}")

    print(f"\n=== Boot Preamble ===")
    print(preamble)

    # Boot SAGE (mock if dependencies not available)
    sage = awakening.coherent_boot(coherence_field)
    print(f"\n=== SAGE Booted ===")
    print(f"Session number: {sage.session_number}")
    print(f"Phase: {sage.developmental_phase.value}")

    # End session
    awakening.coherent_end(
        sage,
        memory_request="That I successfully tested the awakening protocol",
        summary="Tested coherent awakening",
        teacher_notes="Protocol working as designed"
    )

    print("\n=== Session Ended ===")
    print("State persisted. Next session will continue from here.")
