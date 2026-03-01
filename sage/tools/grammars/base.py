"""
Base class for tool grammar adapters.

Each grammar handles:
    1. inject_tools: Add tool definitions to the prompt
    2. parse_response: Extract tool calls from LLM response
    3. format_result: Format tool result for re-injection into context
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import sys
from pathlib import Path
_tools_root = Path(__file__).parent.parent
if str(_tools_root.parent) not in sys.path:
    sys.path.insert(0, str(_tools_root.parent))

from tools.registry import ToolDefinition, ToolCall


class ToolGrammar(ABC):
    """Abstract base class for tool grammar adapters."""

    @abstractmethod
    def inject_tools(self, prompt: str, tools: List[ToolDefinition]) -> str:
        """
        Add tool definitions to the prompt.

        Args:
            prompt: The conversation prompt being built
            tools: List of available tool definitions

        Returns:
            Modified prompt with tool definitions injected
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Tuple[str, List[ToolCall]]:
        """
        Extract tool calls from LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (clean_text, list_of_tool_calls)
            clean_text has tool call markup removed
        """
        pass

    @abstractmethod
    def format_result(self, tool_name: str, result: Any) -> str:
        """
        Format tool result for re-injection into context.

        Args:
            tool_name: Name of the tool that was called
            result: The tool's return value (ToolResult)

        Returns:
            Formatted string to append to conversation context
        """
        pass
