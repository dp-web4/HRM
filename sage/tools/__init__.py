"""
SAGE Tool Use — external capability invocation for SAGE instances.

Provides:
    - ToolDefinition / ToolRegistry: Central tool registration and schema
    - ToolCapability: Runtime detection of model tool-calling tier
    - Grammar adapters: Model-specific prompt injection and response parsing
    - Built-in tools: get_time, calculate, web_search, etc.

Architecture:
    T1 (native):   Ollama /api/chat with tools parameter
    T2 (grammar):  Prompt injection + structured response parsing
    T3 (heuristic): Regex intent detection from natural language
"""

from .registry import ToolDefinition, ToolRegistry, ToolCall
