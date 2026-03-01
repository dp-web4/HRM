"""
T3 Intent Heuristic Grammar — universal fallback.

No prompt injection of tool schemas. Scans LLM response for natural
language patterns that indicate tool intent:
    - "I would search for..." / "Let me look up..." → web_search
    - "I'd like to check the time..." → get_time
    - "If I could calculate..." → calculate
    - "I want to read the file..." → read_file

Conservative: high precision over recall. If not confident, returns
no tool calls and lets the response pass through unchanged.
"""

import re
from typing import List, Tuple, Any

from .base import ToolGrammar, ToolDefinition, ToolCall


# Pattern groups: (compiled regex, tool_name, argument extractor function)
# Each regex should capture the relevant argument in group 1

_TIME_PATTERNS = [
    re.compile(r"(?:what(?:'s| is) the (?:current )?(?:time|date)|(?:check|tell me) (?:the )?(?:time|date))", re.I),
    re.compile(r"(?:I(?:'d| would) (?:like to |want to )?(?:check|know) (?:the )?(?:current )?(?:time|date))", re.I),
    re.compile(r"(?:right now|what time is it)", re.I),
]

_CALC_PATTERNS = [
    re.compile(r"(?:calculate|compute|evaluate|what(?:'s| is))\s+(.+?)(?:\?|$|\.)", re.I),
    re.compile(r"(?:I(?:'d| would) (?:like to )?(?:calculate|compute))\s+(.+?)(?:\?|$|\.)", re.I),
    re.compile(r"(\d+[\s+\-*/^%]+\d+(?:[\s+\-*/^%]+\d+)*)", re.I),
]

_SEARCH_PATTERNS = [
    re.compile(r"(?:search|look up|find|google|look for)\s+(?:for\s+|about\s+)?[\"']?(.+?)[\"']?(?:\s+on the web|\s+online)?(?:\?|$|\.)", re.I),
    re.compile(r"(?:I(?:'d| would) (?:like to |want to )?(?:search|look up))\s+(?:for\s+)?[\"']?(.+?)[\"']?(?:\?|$|\.)", re.I),
    re.compile(r"(?:I (?:want|need) to (?:search|find))\s+(?:for\s+|about\s+)?[\"']?(.+?)[\"']?(?:\?|$|\.)", re.I),
]

_FETCH_PATTERNS = [
    re.compile(r"(?:fetch|visit|open|read|go to|check)\s+(?:the )?(?:URL|page|website|site|link)?\s*(https?://\S+)", re.I),
    re.compile(r"(?:I(?:'d| would) (?:like to )?(?:fetch|visit|read))\s+(https?://\S+)", re.I),
]

_READ_PATTERNS = [
    re.compile(r"(?:read|open|show|display)\s+(?:the )?(?:file\s+)?[\"']([^\"']+)[\"']", re.I),
    re.compile(r"(?:read|open|show|display)\s+(?:the )?file\s+(\S+\.[\w]+)", re.I),
    re.compile(r"(?:I(?:'d| would) (?:like to |want to )?(?:read|open))\s+(?:the )?(?:file\s+)?[\"']?(\S+\.[\w]+)[\"']?", re.I),
]

_NOTE_PATTERNS = [
    re.compile(r"(?:write|save|note|remember|jot down)\s+(?:a note|down|this)?\s*:?\s*[\"'](.+?)[\"']", re.I),
    re.compile(r"(?:I(?:'d| would) (?:like to )?(?:write|save|note))\s+[\"'](.+?)[\"']", re.I),
]


class IntentHeuristicGrammar(ToolGrammar):
    """
    T3 — Universal fallback grammar.

    Does NOT inject tool definitions into the prompt. Instead, scans
    the LLM's natural language response for tool intent patterns.
    """

    def inject_tools(self, prompt: str, tools: List[ToolDefinition]) -> str:
        """T3 does not inject tools into the prompt."""
        return prompt

    def parse_response(self, response: str) -> Tuple[str, List[ToolCall]]:
        """
        Scan response for tool intent patterns.

        Returns at most one tool call (conservative — avoid false positives).
        """
        calls = []

        # Check time patterns
        for pattern in _TIME_PATTERNS:
            if pattern.search(response):
                calls.append(ToolCall(name='get_time'))
                return response, calls

        # Check calculate patterns
        for pattern in _CALC_PATTERNS:
            match = pattern.search(response)
            if match:
                expr = match.group(1).strip()
                # Basic validation: must contain digits and operators
                if re.search(r'\d', expr) and re.search(r'[+\-*/]', expr):
                    calls.append(ToolCall(name='calculate', arguments={'expression': expr}))
                    return response, calls

        # Check search patterns
        for pattern in _SEARCH_PATTERNS:
            match = pattern.search(response)
            if match:
                query = match.group(1).strip()
                if len(query) > 2:
                    calls.append(ToolCall(name='web_search', arguments={'query': query}))
                    return response, calls

        # Check fetch patterns
        for pattern in _FETCH_PATTERNS:
            match = pattern.search(response)
            if match:
                url = match.group(1).strip()
                calls.append(ToolCall(name='web_fetch', arguments={'url': url}))
                return response, calls

        # Check read file patterns
        for pattern in _READ_PATTERNS:
            match = pattern.search(response)
            if match:
                filename = match.group(1).strip()
                calls.append(ToolCall(name='read_file', arguments={'filename': filename}))
                return response, calls

        # Check note patterns
        for pattern in _NOTE_PATTERNS:
            match = pattern.search(response)
            if match:
                content = match.group(1).strip()
                if len(content) > 2:
                    calls.append(ToolCall(name='write_note', arguments={'content': content}))
                    return response, calls

        return response, calls

    def format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result as natural language context."""
        if hasattr(result, 'to_text'):
            return result.to_text()
        return f"[{tool_name} result]: {result}"
