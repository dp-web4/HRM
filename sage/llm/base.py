"""
Base interfaces for LLM backends

Defines the contract that all LLM backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class BackendState(Enum):
    """LLM backend lifecycle states"""
    COLD = 'cold'        # Not loaded, minimal memory
    WARMING = 'warming'  # Loading model
    HOT = 'hot'          # Model loaded, ready for inference
    COOLING = 'cooling'  # Unloading model
    ERROR = 'error'      # Failed state


@dataclass
class LLMRequest:
    """Request to LLM for inference"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    tool_schema: Optional[Dict[str, Any]] = None  # For tool-use models
    constraints: Optional[Dict[str, Any]] = None  # Budget, timeout, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backend consumption"""
        return {
            'prompt': self.prompt,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'stop_sequences': self.stop_sequences or [],
            'tool_schema': self.tool_schema,
            'constraints': self.constraints or {},
        }


@dataclass
class LLMResponse:
    """Response from LLM inference"""
    text: str
    finish_reason: str  # 'stop', 'length', 'error', 'tool_use'
    tokens_generated: int
    inference_time_ms: float
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/experience capture"""
        return {
            'text': self.text,
            'finish_reason': self.finish_reason,
            'tokens_generated': self.tokens_generated,
            'inference_time_ms': self.inference_time_ms,
            'tool_calls': self.tool_calls,
            'metadata': self.metadata or {},
        }


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends

    Backends implement the hot/cold lifecycle and provide
    generate() method for inference.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backend with configuration

        Args:
            config: Backend-specific configuration
                - model_name: Model identifier
                - device: 'cpu', 'cuda', 'mps'
                - dtype: 'float32', 'float16', 'bfloat16'
                - max_memory_gb: Memory budget
                - ... backend-specific params
        """
        self.config = config
        self.state = BackendState.COLD
        self.model_name = config.get('model_name', 'unknown')
        self.device = config.get('device', 'cpu')

    @abstractmethod
    async def warm(self) -> bool:
        """
        Load model into memory, prepare for inference

        Returns:
            True if warming successful, False otherwise
        """
        pass

    @abstractmethod
    async def cool(self) -> bool:
        """
        Unload model from memory, free resources

        Returns:
            True if cooling successful, False otherwise
        """
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response from prompt

        Args:
            request: LLM inference request

        Returns:
            LLM response with text and metadata

        Raises:
            RuntimeError: If backend not in HOT state
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check backend health and resource usage

        Returns:
            Health status dictionary with:
                - state: Current backend state
                - memory_mb: Memory usage
                - last_inference_ms: Time of last inference
                - error: Error message if state is ERROR
        """
        pass

    def get_state(self) -> BackendState:
        """Get current backend state"""
        return self.state

    def is_ready(self) -> bool:
        """Check if backend is ready for inference"""
        return self.state == BackendState.HOT
