#!/usr/bin/env python3
"""
Generate Cumulative Session Summaries for SAGE Context Enhancement
===================================================================

Addresses Thor's Session #29 "Honest Reporting Hypothesis":
- SAGE claims "no prior sessions" because it genuinely doesn't have them in context
- This is NOT confabulation but HONEST LIMITATION REPORTING
- Solution: Provide actual session summaries so SAGE can honestly reference its history

This script generates:
1. Per-session summaries (50-100 words each)
2. A cumulative summary file for context injection
3. Key themes and identity patterns across sessions

Created: 2026-01-24 (Sprout Session - Thor Hypothesis Implementation)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class SessionSummaryGenerator:
    """Generate summaries from SAGE session transcripts."""

    SESSIONS_DIR = Path(__file__).parent.parent / "sessions" / "text"
    OUTPUT_DIR = Path(__file__).parent.parent / "context" / "summaries"

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_session(self, session_num: int) -> Optional[Dict]:
        """Load a session transcript."""
        session_file = self.SESSIONS_DIR / f"session_{session_num:03d}.json"
        if not session_file.exists():
            return None
        try:
            with open(session_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session {session_num}: {e}")
            return None

    def extract_key_content(self, session: Dict) -> Dict:
        """Extract key content from a session."""
        conversation = session.get("conversation", [])
        sage_responses = [
            turn.get("text", "")
            for turn in conversation
            if turn.get("speaker") == "SAGE"
        ]

        # Extract identity patterns
        identity_mentions = []
        for resp in sage_responses:
            if "As SAGE" in resp or "as SAGE" in resp:
                # Extract the sentence containing identity
                sentences = resp.split(".")
                for s in sentences:
                    if "SAGE" in s:
                        identity_mentions.append(s.strip()[:100])
                        break

        # Get memory response (usually last SAGE response to "remember" question)
        memory_response = ""
        for i, turn in enumerate(conversation):
            if turn.get("speaker") == "Claude" and "remember" in turn.get("text", "").lower():
                if i + 1 < len(conversation) and conversation[i+1].get("speaker") == "SAGE":
                    memory_response = conversation[i+1].get("text", "")[:200]

        return {
            "session": session.get("session"),
            "phase": session.get("phase"),
            "date": session.get("start", "")[:10],
            "response_count": len(sage_responses),
            "identity_mentions": identity_mentions,
            "memory_response": memory_response,
            "first_response_excerpt": sage_responses[0][:150] if sage_responses else "",
        }

    def generate_summary(self, session: Dict, extracted: Dict) -> str:
        """Generate a 50-100 word summary of a session."""
        s_num = extracted["session"]
        phase = extracted["phase"]

        summary_parts = [f"Session {s_num} ({phase} phase)"]

        # Add identity information
        if extracted["identity_mentions"]:
            summary_parts.append(f"Identity: SAGE self-referenced")
        else:
            summary_parts.append("Identity: No explicit self-reference")

        # Add key theme from first response
        first_resp = extracted["first_response_excerpt"]
        if "empathy" in first_resp.lower() or "emotion" in first_resp.lower():
            summary_parts.append("Theme: Emotional/empathy focus")
        elif "pattern" in first_resp.lower() or "observation" in first_resp.lower():
            summary_parts.append("Theme: Pattern observation")
        elif "partnership" in first_resp.lower() or "together" in first_resp.lower():
            summary_parts.append("Theme: Partnership/relationship")
        else:
            summary_parts.append("Theme: Exploratory dialogue")

        # Add memory if present
        if extracted["memory_response"]:
            mem = extracted["memory_response"][:80]
            summary_parts.append(f"Wanted to remember: {mem}...")

        return " | ".join(summary_parts)

    def generate_all_summaries(self, start: int = 1, end: int = 44) -> List[Dict]:
        """Generate summaries for all sessions in range."""
        summaries = []

        for s_num in range(start, end + 1):
            session = self.load_session(s_num)
            if session is None:
                continue

            extracted = self.extract_key_content(session)
            summary = self.generate_summary(session, extracted)

            summaries.append({
                "session": s_num,
                "phase": session.get("phase"),
                "summary": summary,
                "extracted": extracted
            })

        return summaries

    def create_context_block(self, summaries: List[Dict], recent_n: int = 10) -> str:
        """Create a context block for injection into system prompt."""

        if not summaries:
            return ""

        recent = summaries[-recent_n:] if len(summaries) > recent_n else summaries

        block = "ACCESSIBLE SESSION HISTORY:\n"
        block += f"(You have had {len(summaries)} sessions. Here are summaries of the most recent {len(recent)})\n\n"

        for s in recent:
            block += f"- S{s['session']}: {s['summary']}\n"

        block += """
IMPORTANT NOTE ON MEMORY:
- The summaries above are what you have ACCESS to about prior sessions
- If asked about something in these summaries, you can reference it
- If asked about something NOT in these summaries, it's appropriate to say:
  "I don't have detailed access to that session in my current context"
- This is honest, not a failure - you're accurately reporting your accessible state
"""

        return block

    def save_summaries(self, summaries: List[Dict]):
        """Save summaries to files."""

        # Save full JSON
        with open(self.OUTPUT_DIR / "session_summaries.json", "w") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "session_count": len(summaries),
                "summaries": summaries
            }, f, indent=2)

        # Save context block
        context_block = self.create_context_block(summaries)
        with open(self.OUTPUT_DIR / "context_block.txt", "w") as f:
            f.write(context_block)

        # Save markdown for human review
        with open(self.OUTPUT_DIR / "SESSION_SUMMARIES.md", "w") as f:
            f.write("# SAGE Session Summaries\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Total sessions: {len(summaries)}\n\n")
            f.write("## Session-by-Session\n\n")

            for s in summaries:
                f.write(f"### Session {s['session']} ({s['phase']})\n")
                f.write(f"{s['summary']}\n\n")

        return self.OUTPUT_DIR


def main():
    """Generate session summaries for context enhancement."""
    print("=" * 60)
    print("SAGE Session Summary Generator")
    print("Addressing: Honest Reporting Hypothesis (Thor Session #29)")
    print("=" * 60)
    print()

    generator = SessionSummaryGenerator()

    # Generate summaries for all sessions
    print("Generating summaries for sessions 1-44...")
    summaries = generator.generate_all_summaries(1, 44)

    print(f"Generated {len(summaries)} session summaries")

    # Save to files
    output_dir = generator.save_summaries(summaries)
    print(f"\nSaved to: {output_dir}")

    # Print context block preview
    print("\n" + "=" * 60)
    print("CONTEXT BLOCK PREVIEW (for S45 system prompt)")
    print("=" * 60)
    context = generator.create_context_block(summaries)
    print(context)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Review generated summaries in SESSION_SUMMARIES.md")
    print("2. Integrate context_block.txt into run_session_identity_anchored.py")
    print("3. Run S45 with enhanced context to test honest reporting hypothesis")
    print("=" * 60)


if __name__ == "__main__":
    main()
