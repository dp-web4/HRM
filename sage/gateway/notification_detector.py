"""
Notification detector — regex-based scan for human-directed messages.

Scans LLM output for patterns that explicitly address the operator or
request human input. Used by the gateway and raising sessions to surface
notifications in the dashboard.

Operator names are extracted from the instance's identity.json at init
time — nothing is hardcoded.
"""

import re
from typing import Optional, List, Dict, Any


def extract_operator_names(identity_state: Optional[Dict[str, Any]] = None) -> List[str]:
    """Extract operator/creator names from identity relationships.

    Looks for relationships with role 'operator' or 'creator' and uses
    the relationship key as the human name. Returns both original case
    and title case variants for pattern matching.
    """
    if not identity_state:
        return []
    relationships = identity_state.get('relationships', {})
    names = set()
    for key, rel in relationships.items():
        role = rel.get('role', '')
        if role in ('operator', 'creator'):
            names.add(key)
            names.add(key.capitalize())
    return sorted(names) if names else []


class NotificationDetector:
    """Detect human-directed messages in SAGE output using compiled regex patterns."""

    def __init__(self, human_names: Optional[list] = None):
        # Name-specific patterns are only compiled if names are provided
        self._patterns = []

        if human_names:
            name_alt = '|'.join(re.escape(n) for n in human_names)
            name_patterns = [
                # Direct address: "Name," or "Name:" at start or after newline
                (f'(?:^|\\n)\\s*(?:{name_alt})\\s*[,:]', 'direct_address'),
                # @mention
                (f'@(?:{name_alt})', 'at_mention'),
                # Request phrases: "ask Name", "tell Name", "let Name know"
                (f'(?:ask|tell|inform|notify|let)\\s+(?:{name_alt})\\b', 'request_phrase'),
                # Post-positioned: "could you, Name"
                (f'could you[,.]?\\s*(?:{name_alt})', 'request_phrase'),
                # "I'd like to ask Name" / "I need Name to"
                (f"(?:I'?d like to (?:ask|tell|inform)|I need)\\s+(?:{name_alt})\\b",
                 'request_phrase'),
            ]
            for pat, label in name_patterns:
                self._patterns.append((re.compile(pat, re.IGNORECASE), label))

        # Explicit operator requests (name-independent — always active)
        generic_patterns = [
            (r'\b(?:I need human input|requesting (?:operator|human) (?:attention|input|review))\b',
             'explicit_request'),
            (r'\b(?:human (?:review|intervention|decision) (?:needed|required|requested))\b',
             'explicit_request'),
            (r'\b(?:operator[,:]?\s*(?:please|could you))\b', 'explicit_request'),
        ]
        for pat, label in generic_patterns:
            self._patterns.append((re.compile(pat, re.IGNORECASE), label))

    def scan(self, text: str, source: str = 'chat') -> Optional[list]:
        """Scan text for human-directed patterns.

        Returns a list of match dicts or None if no matches found.
        Each dict: {pattern, matched_text, context_snippet}
        """
        if not text:
            return None

        matches = []
        for regex, label in self._patterns:
            for m in regex.finditer(text):
                start = max(0, m.start() - 25)
                end = min(len(text), m.end() + 25)
                snippet = text[start:end].replace('\n', ' ').strip()
                matches.append({
                    'pattern': label,
                    'matched_text': m.group().strip(),
                    'context_snippet': snippet,
                })

        return matches if matches else None
