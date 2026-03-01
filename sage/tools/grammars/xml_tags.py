"""
T2 XML Tags Grammar — structured tool calling via XML tags.

Injects tool definitions in a <tools> block in the prompt.
Parses <tool_call>{"name": "...", "arguments": {...}}</tool_call> from response.

Works well with models that have structured output training but don't
support native Ollama /api/chat tools parameter.
"""

import json
import re
from typing import List, Tuple, Any

from .base import ToolGrammar, ToolDefinition, ToolCall


# Match <tool_call>...</tool_call> blocks
_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL
)

# Also accept [tool_call]...[/tool_call] as an alternative
_TOOL_CALL_ALT_PATTERN = re.compile(
    r'\[tool_call\]\s*(\{.*?\})\s*\[/tool_call\]',
    re.DOTALL
)


class XmlTagsGrammar(ToolGrammar):
    """
    T2 — XML tag-based tool calling.

    Prompt injection format:
        <tools>
        - tool_name: description
          Parameters: {"param": "type"}
        ...
        </tools>

        To use a tool, respond with:
        <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

    Response parsing: extracts <tool_call> JSON blocks.
    """

    def inject_tools(self, prompt: str, tools: List[ToolDefinition]) -> str:
        """Inject tool definitions as XML block at the start of the prompt."""
        if not tools:
            return prompt

        tool_block = '<tools>\n'
        for tool in tools:
            tool_block += tool.to_prompt_text() + '\n'
        tool_block += '</tools>\n\n'
        tool_block += (
            'To use a tool, include in your response:\n'
            '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>\n'
            'You may use at most one tool per response. '
            'The tool result will be provided and you can then give your final answer.\n\n'
        )

        # Insert tool block before the conversation
        # Look for the separator (---) that divides system context from conversation
        separator_idx = prompt.find('\n---\n')
        if separator_idx >= 0:
            # Insert tools between system context and conversation
            return prompt[:separator_idx] + '\n\n' + tool_block + prompt[separator_idx:]
        else:
            # Prepend to prompt
            return tool_block + prompt

    def parse_response(self, response: str) -> Tuple[str, List[ToolCall]]:
        """Extract tool calls from <tool_call> tags in the response."""
        calls = []
        clean = response

        for pattern in [_TOOL_CALL_PATTERN, _TOOL_CALL_ALT_PATTERN]:
            for match in pattern.finditer(response):
                try:
                    data = json.loads(match.group(1))
                    name = data.get('name', '')
                    args = data.get('arguments', data.get('args', {}))
                    if name:
                        calls.append(ToolCall(name=name, arguments=args))
                except (json.JSONDecodeError, AttributeError):
                    continue

            if calls:
                # Remove tool_call tags from clean text
                clean = pattern.sub('', clean).strip()
                break

        return clean, calls

    def format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result as XML for re-injection."""
        if hasattr(result, 'to_text'):
            text = result.to_text()
        else:
            text = str(result)
        return f'<tool_result name="{tool_name}">\n{text}\n</tool_result>'
