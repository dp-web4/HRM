"""
SAGE LLM Runtime (Tier 1)

On-demand language model inference for deep reasoning.
Invoked by Tier 0 kernel during THINK state.
"""

from .base import LLMBackend, LLMRequest, LLMResponse, BackendState
from .runtime import LLMRuntime, BackendType
from .ollama_backend import OllamaBackend

__all__ = [
    'LLMBackend',
    'LLMRequest',
    'LLMResponse',
    'BackendState',
    'LLMRuntime',
    'BackendType',
    'OllamaBackend',
]


def __getattr__(name):
    """Lazy import for torch-dependent backends."""
    if name == 'TransformersBackend':
        from .transformers_backend import TransformersBackend
        return TransformersBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
