"""
Grammar adapters for SAGE tool use.

Each adapter handles model-specific translation between natural language
and structured tool calls.

Usage:
    grammar = get_grammar('xml_tags')
    prompt = grammar.inject_tools(prompt, tools)
    text, calls = grammar.parse_response(response)
    result_text = grammar.format_result('web_search', result)
"""

from .base import ToolGrammar
from .intent_heuristic import IntentHeuristicGrammar
from .xml_tags import XmlTagsGrammar
from .json_block import JsonBlockGrammar
from .native_ollama import NativeOllamaGrammar


_GRAMMARS = {
    'intent_heuristic': IntentHeuristicGrammar,
    'xml_tags': XmlTagsGrammar,
    'json_block': JsonBlockGrammar,
    'native_ollama': NativeOllamaGrammar,
}


def get_grammar(grammar_id: str) -> ToolGrammar:
    """Get a grammar adapter by ID."""
    cls = _GRAMMARS.get(grammar_id)
    if cls is None:
        raise ValueError(f"Unknown grammar: {grammar_id}. Available: {list(_GRAMMARS.keys())}")
    return cls()
