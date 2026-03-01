"""
Ollama backend for LLM Runtime

Connects to Ollama server (local or remote) for inference.
Ollama manages model lifecycle, so warm/cool are no-ops.
"""

import time
import asyncio
from typing import Dict, Any
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import LLMBackend, LLMRequest, LLMResponse, BackendState


class OllamaBackend(LLMBackend):
    """
    Ollama backend implementation

    Ollama server manages model lifecycle externally.
    This adapter just provides the generate() interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama backend

        Config:
            - model_name: Ollama model identifier (e.g., 'llama3.2:3b')
            - base_url: Ollama API endpoint (default: 'http://localhost:11434')
            - timeout_s: Request timeout in seconds (default: 120)
        """
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.timeout_s = config.get('timeout_s', 120)
        self.client = None
        self.last_inference_time = None
        self.error_message = None

        if not HTTPX_AVAILABLE:
            self.state = BackendState.ERROR
            self.error_message = "httpx not installed (pip install httpx)"
        else:
            self.state = BackendState.COLD

    async def warm(self) -> bool:
        """
        'Warm' Ollama backend by testing connectivity

        Ollama manages models externally, so this just checks
        that the server is reachable.
        """
        if not HTTPX_AVAILABLE:
            return False

        try:
            self.state = BackendState.WARMING

            # Create httpx client with timeout
            self.client = httpx.AsyncClient(timeout=self.timeout_s)

            # Test connectivity with /api/tags endpoint
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if self.model_name not in model_names:
                print(f"[OllamaBackend] Warning: {self.model_name} not in available models: {model_names}")
                print(f"[OllamaBackend] Ollama will pull on first use")

            self.state = BackendState.HOT
            print(f"[OllamaBackend] Warmed: {self.model_name} @ {self.base_url}")
            return True

        except Exception as e:
            self.state = BackendState.ERROR
            self.error_message = str(e)
            print(f"[OllamaBackend] Failed to warm: {e}")
            return False

    async def cool(self) -> bool:
        """
        'Cool' Ollama backend by closing HTTP client

        Ollama server continues running externally.
        """
        try:
            self.state = BackendState.COOLING

            if self.client:
                await self.client.aclose()
                self.client = None

            self.state = BackendState.COLD
            print(f"[OllamaBackend] Cooled: {self.model_name}")
            return True

        except Exception as e:
            self.state = BackendState.ERROR
            self.error_message = str(e)
            print(f"[OllamaBackend] Failed to cool: {e}")
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using Ollama /api/generate endpoint

        Args:
            request: LLM inference request

        Returns:
            LLM response with generated text

        Raises:
            RuntimeError: If backend not ready
        """
        if self.state != BackendState.HOT:
            raise RuntimeError(f"Backend not ready (state={self.state.value})")

        start_time = time.time()

        try:
            # Build Ollama request payload
            payload = {
                'model': self.model_name,
                'prompt': request.prompt,
                'stream': False,  # Get full response at once
                'options': {
                    'num_predict': request.max_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                }
            }

            if request.stop_sequences:
                payload['options']['stop'] = request.stop_sequences

            # Call Ollama API
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            # Extract response fields
            text = result.get('response', '')
            finish_reason = 'stop' if result.get('done', False) else 'error'
            inference_time_ms = (time.time() - start_time) * 1000
            self.last_inference_time = time.time()

            # Token count (Ollama provides this in eval_count)
            tokens_generated = result.get('eval_count', len(text.split()))

            return LLMResponse(
                text=text,
                finish_reason=finish_reason,
                tokens_generated=tokens_generated,
                inference_time_ms=inference_time_ms,
                metadata={
                    'model': result.get('model', self.model_name),
                    'total_duration_ns': result.get('total_duration', 0),
                    'load_duration_ns': result.get('load_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                }
            )

        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            self.error_message = str(e)
            print(f"[OllamaBackend] Generate failed: {e}")

            return LLMResponse(
                text='',
                finish_reason='error',
                tokens_generated=0,
                inference_time_ms=inference_time_ms,
                metadata={'error': str(e)}
            )

    async def chat(
        self,
        messages: list,
        tools: list = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate response using Ollama /api/chat endpoint.

        Supports structured messages and native tool calling (T1).

        Args:
            messages: List of {"role": "...", "content": "..."}
            tools: Optional list of Ollama tool definitions
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            Dict with 'content', 'tool_calls', 'role', 'raw'
        """
        if self.state != BackendState.HOT:
            raise RuntimeError(f"Backend not ready (state={self.state.value})")

        payload: Dict[str, Any] = {
            'model': self.model_name,
            'messages': messages,
            'stream': False,
            'options': {
                'num_predict': kwargs.get('max_tokens', 512),
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.9),
            },
        }

        if tools:
            payload['tools'] = tools

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            message = result.get('message', {})
            return {
                'content': message.get('content', ''),
                'tool_calls': message.get('tool_calls', []),
                'role': message.get('role', 'assistant'),
                'raw': result,
            }

        except Exception as e:
            self.error_message = str(e)
            return {
                'content': '',
                'tool_calls': [],
                'role': 'assistant',
                'raw': {'error': str(e)},
            }

    def health_check(self) -> Dict[str, Any]:
        """Check Ollama backend health"""
        return {
            'backend': 'ollama',
            'state': self.state.value,
            'model': self.model_name,
            'base_url': self.base_url,
            'last_inference_time': self.last_inference_time,
            'error': self.error_message,
            'httpx_available': HTTPX_AVAILABLE,
        }
