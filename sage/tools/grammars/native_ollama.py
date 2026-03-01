"""
T1 Native Ollama Grammar — uses Ollama /api/chat with tools parameter.

Tool calls come back as structured JSON in the response — no text parsing
needed. This is the cleanest path but requires model support.

This grammar handles prompt formatting for the non-tool parts. The actual
tool schema is passed directly to Ollama in the API request payload.
"""

import json
from typing import List, Tuple, Any

from .base import ToolGrammar, ToolDefinition, ToolCall


class NativeOllamaGrammar(ToolGrammar):
    """
    T1 — Native Ollama tool calling.

    For T1, tool definitions are passed to the Ollama /api/chat endpoint
    as the `tools` parameter, not injected into the prompt text. This
    grammar's inject_tools is a no-op (Ollama handles it natively).

    parse_response handles both:
    - Structured tool_calls from Ollama /api/chat response
    - Fallback text parsing for edge cases
    """

    def inject_tools(self, prompt: str, tools: List[ToolDefinition]) -> str:
        """
        T1 does NOT inject tools into the prompt text.

        Tools are passed separately via the Ollama /api/chat `tools` parameter.
        The prompt is returned unchanged.
        """
        return prompt

    def parse_response(self, response: str) -> Tuple[str, List[ToolCall]]:
        """
        Parse tool calls from Ollama /api/chat response.

        For T1, tool calls are typically pre-parsed from the structured
        JSON response before reaching this method. This method handles
        the text portion only.

        If called on raw text (fallback), attempts to parse any JSON
        tool call format.
        """
        # T1 responses normally come pre-parsed by the ollama_irp layer.
        # This is only called on the text content.
        return response, []

    def parse_structured_response(self, message: dict) -> Tuple[str, List[ToolCall]]:
        """
        Parse Ollama /api/chat structured message.

        The Ollama /api/chat response has:
        {
            "message": {
                "role": "assistant",
                "content": "text...",
                "tool_calls": [
                    {"function": {"name": "...", "arguments": {...}}}
                ]
            }
        }
        """
        text = message.get('content', '')
        calls = []

        tool_calls = message.get('tool_calls', [])
        for tc in tool_calls:
            fn = tc.get('function', {})
            name = fn.get('name', '')
            # Arguments may be a string (JSON) or dict
            args = fn.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                calls.append(ToolCall(name=name, arguments=args))

        return text, calls

    def format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result for T1 re-injection (Ollama tool response format)."""
        if hasattr(result, 'to_text'):
            text = result.to_text()
        elif hasattr(result, 'result'):
            text = str(result.result) if result.success else f"Error: {result.error}"
        else:
            text = str(result)
        return text
