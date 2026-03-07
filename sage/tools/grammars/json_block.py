"""
T2 JSON Block Grammar — structured tool calling via markdown JSON blocks.

Variant of T2 where the model outputs ```json blocks containing tool calls.
Some models prefer this format over XML tags.

Prompt format:
    Available tools (respond with a ```json block to call):
    ...

Response format:
    ```json
    {"tool": "web_search", "args": {"query": "..."}}
    ```
"""

import json
import re
from typing import List, Tuple, Any

from .base import ToolGrammar, ToolDefinition, ToolCall


# Match ```json ... ``` blocks
_JSON_BLOCK_PATTERN = re.compile(
    r'```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```',
    re.DOTALL
)

# Match bare JSON tool calls (no fencing) — for models that omit markdown
# Must have "tool" key to avoid matching arbitrary JSON
_BARE_JSON_PATTERN = re.compile(
    r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\})',
    re.DOTALL
)


class JsonBlockGrammar(ToolGrammar):
    """
    T2 variant — JSON code block-based tool calling.

    Models output tool calls as markdown JSON fenced code blocks.
    """

    def inject_tools(self, prompt: str, tools: List[ToolDefinition]) -> str:
        """Inject tool definitions with JSON block instruction."""
        if not tools:
            return prompt

        tool_block = 'Available tools:\n'
        for tool in tools:
            tool_block += tool.to_prompt_text() + '\n'
        tool_block += (
            '\nTo call a tool, include a JSON code block in your response:\n'
            '```json\n'
            '{"tool": "tool_name", "args": {"param": "value"}}\n'
            '```\n'
            'You may call at most one tool per response.\n\n'
        )

        separator_idx = prompt.find('\n---\n')
        if separator_idx >= 0:
            return prompt[:separator_idx] + '\n\n' + tool_block + prompt[separator_idx:]
        else:
            return tool_block + prompt

    def parse_response(self, response: str) -> Tuple[str, List[ToolCall]]:
        """Extract tool calls from ```json blocks or bare JSON in the response."""
        calls = []
        clean = response

        # Try fenced first, then bare JSON
        for pattern in [_JSON_BLOCK_PATTERN, _BARE_JSON_PATTERN]:
            for match in pattern.finditer(response):
                try:
                    data = json.loads(match.group(1))

                    # Support both formats:
                    # {"tool": "name", "args": {...}}
                    # {"name": "name", "arguments": {...}}
                    name = data.get('tool', data.get('name', ''))
                    args = data.get('args', data.get('arguments', {}))

                    if name:
                        calls.append(ToolCall(name=name, arguments=args))
                except (json.JSONDecodeError, AttributeError):
                    continue

            if calls:
                clean = pattern.sub('', clean).strip()
                break  # Don't double-parse

        return clean, calls

    def format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result for re-injection."""
        if hasattr(result, 'to_text'):
            text = result.to_text()
        else:
            text = str(result)
        return f'Tool result ({tool_name}):\n{text}'
