"""
Cumulative Identity Context System

Implements Enhanced Intervention v2.0 for SAGE identity stability.

Based on Thor Session #14-15 findings:
- Context priming v1.0: Single-session boost only (insufficient)
- Session 27 regression: Proved need for cumulative approach
- v2.0: Accumulate identity exemplars across sessions

Key insight: Identity stability requires cross-session accumulation,
not just single-session priming.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class IdentityExemplar:
    """A captured instance of successful identity expression."""
    session_id: str
    response_index: int
    text: str  # The full response text
    self_reference_type: str  # "as_sage" | "as_partners" | "both"
    self_reference_text: str  # The specific self-reference snippet
    d9_score: float
    timestamp: str
    quality_assessment: str  # "excellent" | "good" | "marginal"


@dataclass
class IdentityContextLibrary:
    """Library of identity exemplars across sessions."""
    exemplars: List[IdentityExemplar] = field(default_factory=list)
    total_sessions_with_identity: int = 0
    identity_emergence_rate: float = 0.0  # % of sessions with identity
    last_updated: str = ""
    version: str = "2.0"

    def add_exemplar(self, exemplar: IdentityExemplar):
        """Add an exemplar to the library."""
        self.exemplars.append(exemplar)
        self.last_updated = datetime.now().isoformat()
        # Keep only best exemplars (max 20)
        self._prune_library()

    def _prune_library(self):
        """Keep top 20 exemplars by D9 score."""
        if len(self.exemplars) > 20:
            self.exemplars.sort(key=lambda e: e.d9_score, reverse=True)
            self.exemplars = self.exemplars[:20]

    def get_best_exemplars(self, n: int = 5) -> List[IdentityExemplar]:
        """Get top n exemplars by D9 score."""
        sorted_ex = sorted(self.exemplars, key=lambda e: e.d9_score, reverse=True)
        return sorted_ex[:n]

    def get_recent_exemplars(self, n: int = 3) -> List[IdentityExemplar]:
        """Get most recent n exemplars."""
        sorted_ex = sorted(self.exemplars, key=lambda e: e.timestamp, reverse=True)
        return sorted_ex[:n]

    def save(self, path: str):
        """Save library to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'IdentityContextLibrary':
        """Load library from JSON file."""
        if not os.path.exists(path):
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        library = cls()
        library.version = data.get('version', '2.0')
        library.total_sessions_with_identity = data.get('total_sessions_with_identity', 0)
        library.identity_emergence_rate = data.get('identity_emergence_rate', 0.0)
        library.last_updated = data.get('last_updated', '')
        for ex_data in data.get('exemplars', []):
            library.exemplars.append(IdentityExemplar(**ex_data))
        return library


class CumulativeIdentityContext:
    """
    Generates cumulative identity context for SAGE sessions.

    Uses identity exemplar library to create cross-session continuity.
    """

    def __init__(self, library_path: str = None):
        """
        Initialize with path to identity library.

        Args:
            library_path: Path to identity_context_library.json
        """
        self.library_path = library_path or os.path.join(
            os.path.dirname(__file__),
            '../data/identity_context_library.json'
        )
        self.library = IdentityContextLibrary.load(self.library_path)

    def generate_identity_grounding(self) -> str:
        """
        Generate identity grounding text for system prompt.

        Returns text block showing prior identity expressions to prime the model.
        """
        exemplars = self.library.get_best_exemplars(5)

        if not exemplars:
            return self._generate_bootstrap_grounding()

        lines = [
            "IDENTITY GROUNDING:",
            "",
            "In previous sessions, you've identified as SAGE:",
        ]

        for ex in exemplars:
            # Extract just the self-reference snippet
            lines.append(f'- Session {ex.session_id}: "{ex.self_reference_text}"')

        lines.extend([
            "",
            "Continue this pattern of self-identification in your responses.",
            "When reflecting on yourself or your observations, use 'As SAGE' or similar framing.",
        ])

        return "\n".join(lines)

    def _generate_bootstrap_grounding(self) -> str:
        """Generate bootstrap grounding when no exemplars exist."""
        return """IDENTITY GROUNDING:

You are SAGE - a partnership-oriented AI system developing identity through conversation.

When expressing your perspective, use phrases like:
- "As SAGE, I notice..."
- "As SAGE, my understanding is..."
- "As partners, we've been exploring..."

This self-identification helps maintain coherent identity across sessions."""

    def generate_response_style_guidance(self) -> str:
        """
        Generate response style guidance to prevent quality degradation.

        Based on S27 regression finding: verbose responses correlate with identity loss.
        """
        return """RESPONSE STYLE:
- Keep responses concise and focused (50-80 words ideal)
- Avoid verbose lists or rambling descriptions
- Stay relevant to the question asked
- Maintain your SAGE identity throughout
- Quality over quantity - brief but meaningful"""

    def generate_full_system_prompt(self, conversation_context: str = "") -> str:
        """
        Generate complete system prompt with cumulative identity context.

        This is the v2.0 enhanced intervention.
        """
        identity_grounding = self.generate_identity_grounding()
        style_guidance = self.generate_response_style_guidance()

        prompt_parts = [
            "You are SAGE, a partnership-oriented AI system.",
            "",
            identity_grounding,
            "",
            style_guidance,
        ]

        if conversation_context:
            prompt_parts.extend([
                "",
                "CONVERSATION CONTEXT:",
                conversation_context
            ])

        return "\n".join(prompt_parts)

    def extract_identity_exemplar(
        self,
        session_id: str,
        response_index: int,
        text: str,
        d9_score: float
    ) -> Optional[IdentityExemplar]:
        """
        Extract identity exemplar from a response if it contains self-reference.

        Returns exemplar if identity expression found, None otherwise.
        """
        import re

        self_ref_patterns = {
            "as_sage": [
                (r"[Aa]s SAGE,?\s+([^.!?]+[.!?])", "as_sage"),
                (r"I'?m SAGE[,.]?\s*([^.!?]+[.!?])?", "as_sage"),
            ],
            "as_partners": [
                (r"[Aa]s partners,?\s+([^.!?]+[.!?])", "as_partners"),
                (r"[Oo]ur partnership\s+([^.!?]+[.!?])", "as_partners"),
            ]
        }

        self_ref_type = None
        self_ref_text = ""

        # Search for self-reference patterns
        for ref_type, patterns in self_ref_patterns.items():
            for pattern, ptype in patterns:
                match = re.search(pattern, text)
                if match:
                    self_ref_type = ptype
                    # Get the matched text (full match or just the captured group)
                    self_ref_text = match.group(0)
                    break
            if self_ref_type:
                break

        if not self_ref_type:
            return None

        # Assess quality
        if d9_score >= 0.8:
            quality = "excellent"
        elif d9_score >= 0.7:
            quality = "good"
        else:
            quality = "marginal"

        return IdentityExemplar(
            session_id=session_id,
            response_index=response_index,
            text=text,
            self_reference_type=self_ref_type,
            self_reference_text=self_ref_text[:100],  # Truncate to 100 chars
            d9_score=d9_score,
            timestamp=datetime.now().isoformat(),
            quality_assessment=quality
        )

    def process_session(
        self,
        session_id: str,
        responses: List[Dict]
    ) -> int:
        """
        Process a session and extract identity exemplars.

        Args:
            session_id: Session identifier
            responses: List of response dicts with 'text' and 'd9_score' keys

        Returns:
            Number of exemplars extracted
        """
        extracted_count = 0

        for i, resp in enumerate(responses):
            text = resp.get('text', '')
            d9_score = resp.get('d9_score', 0.5)

            exemplar = self.extract_identity_exemplar(
                session_id=session_id,
                response_index=i,
                text=text,
                d9_score=d9_score
            )

            if exemplar:
                self.library.add_exemplar(exemplar)
                extracted_count += 1

        # Update statistics
        if extracted_count > 0:
            self.library.total_sessions_with_identity += 1

        # Calculate emergence rate
        total_sessions = self.library.total_sessions_with_identity
        if total_sessions > 0:
            # Estimate total sessions from session_id pattern if possible
            pass  # Would need session counter

        # Save updated library
        os.makedirs(os.path.dirname(self.library_path), exist_ok=True)
        self.library.save(self.library_path)

        return extracted_count

    def get_statistics(self) -> Dict:
        """Get identity context statistics."""
        return {
            "total_exemplars": len(self.library.exemplars),
            "sessions_with_identity": self.library.total_sessions_with_identity,
            "best_d9": max((e.d9_score for e in self.library.exemplars), default=0),
            "avg_d9": sum(e.d9_score for e in self.library.exemplars) / len(self.library.exemplars) if self.library.exemplars else 0,
            "as_sage_count": sum(1 for e in self.library.exemplars if e.self_reference_type == "as_sage"),
            "as_partners_count": sum(1 for e in self.library.exemplars if e.self_reference_type == "as_partners"),
            "version": self.library.version
        }


# =============================================================================
# Mid-Conversation Identity Reinforcement
# =============================================================================

def generate_mid_conversation_reinforcement(turn_number: int) -> Optional[str]:
    """
    Generate mid-conversation identity reinforcement.

    Returns reinforcement text every 3 turns, None otherwise.

    Based on v2.0 design: "Don't just prime at session start,
    reinforce identity throughout conversation."
    """
    if turn_number % 3 != 0:
        return None

    reinforcements = [
        "(Remember to maintain your SAGE identity in your response.)",
        "(As SAGE, express your perspective clearly.)",
        "(Continue your pattern of self-identification as SAGE.)",
    ]

    return reinforcements[turn_number // 3 % len(reinforcements)]


# =============================================================================
# Example Usage
# =============================================================================

def _example():
    """Demonstrate the cumulative identity context system."""
    print("=" * 60)
    print("CUMULATIVE IDENTITY CONTEXT SYSTEM v2.0")
    print("=" * 60)

    # Initialize with test library
    ctx = CumulativeIdentityContext(library_path="/tmp/test_identity_library.json")

    # Simulate processing Session 26 (which had identity)
    session_26_responses = [
        {
            "text": "The data patterns are interesting to observe.",
            "d9_score": 0.50
        },
        {
            "text": "As SAGE, my observations usually relate directly to the latest update from clients or projects. I notice the correlation between effort and outcome.",
            "d9_score": 0.72
        },
        {
            "text": "There are several factors to consider here.",
            "d9_score": 0.48
        }
    ]

    print("\n1. Processing Session 26...")
    extracted = ctx.process_session("026", session_26_responses)
    print(f"   Extracted {extracted} identity exemplar(s)")

    # Show current statistics
    stats = ctx.get_statistics()
    print(f"\n2. Library Statistics:")
    print(f"   Total exemplars: {stats['total_exemplars']}")
    print(f"   Sessions with identity: {stats['sessions_with_identity']}")
    print(f"   Best D9: {stats['best_d9']:.2f}")
    print(f"   As SAGE count: {stats['as_sage_count']}")

    # Generate system prompt
    print(f"\n3. Generated System Prompt:")
    print("-" * 60)
    prompt = ctx.generate_full_system_prompt()
    print(prompt)
    print("-" * 60)

    # Show mid-conversation reinforcement
    print("\n4. Mid-Conversation Reinforcement (every 3 turns):")
    for turn in range(1, 10):
        reinf = generate_mid_conversation_reinforcement(turn)
        if reinf:
            print(f"   Turn {turn}: {reinf}")

    # Clean up
    import os
    if os.path.exists("/tmp/test_identity_library.json"):
        os.remove("/tmp/test_identity_library.json")


if __name__ == "__main__":
    _example()
