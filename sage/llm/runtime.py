"""
LLM Runtime Orchestrator

Manages LLM backend lifecycle and provides unified inference interface.
Supports hot/cold state transitions and automatic resource management.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from enum import Enum

from .base import LLMBackend, LLMRequest, LLMResponse, BackendState
from .ollama_backend import OllamaBackend
from .transformers_backend import TransformersBackend


class BackendType(Enum):
    """Supported backend types"""
    OLLAMA = 'ollama'
    TRANSFORMERS = 'transformers'


class LLMRuntime:
    """
    LLM Runtime service (Tier 1)

    Manages backend lifecycle and provides inference API.
    Stays cold by default, warms on demand, cools after idle timeout.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM Runtime

        Config:
            - backend_type: 'ollama' or 'transformers' (default: 'ollama')
            - backend_config: Backend-specific configuration
            - auto_warm: Warm backend on init (default: False)
            - auto_cool_timeout_s: Seconds idle before auto-cool (default: 300)
            - enable_auto_cool: Enable auto-cooling (default: False)
        """
        config = config or {}

        # Runtime configuration
        self.backend_type_str = config.get('backend_type', 'ollama')
        self.backend_config = config.get('backend_config', {})
        self.auto_warm = config.get('auto_warm', False)
        self.auto_cool_timeout_s = config.get('auto_cool_timeout_s', 300)
        self.enable_auto_cool = config.get('enable_auto_cool', False)

        # Initialize backend
        self.backend = self._create_backend(
            self.backend_type_str,
            self.backend_config
        )

        # Lifecycle tracking
        self.last_request_time = None
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_inference_time_ms = 0.0
        self.auto_cool_task = None

        print(f"[LLMRuntime] Initialized with {self.backend_type_str} backend")

    def _create_backend(
        self,
        backend_type: str,
        backend_config: Dict[str, Any]
    ) -> LLMBackend:
        """Create backend instance based on type"""
        if backend_type == 'ollama':
            return OllamaBackend(backend_config)
        elif backend_type == 'transformers':
            return TransformersBackend(backend_config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    async def start(self):
        """
        Start LLM Runtime

        Warms backend if auto_warm enabled.
        Starts auto-cool monitor if enabled.
        """
        print(f"[LLMRuntime] Starting...")

        if self.auto_warm:
            await self.warm()

        if self.enable_auto_cool:
            self.auto_cool_task = asyncio.create_task(self._auto_cool_monitor())

        print(f"[LLMRuntime] Started (state={self.backend.state.value})")

    async def stop(self):
        """
        Stop LLM Runtime

        Cools backend and stops auto-cool monitor.
        """
        print(f"[LLMRuntime] Stopping...")

        # Cancel auto-cool task
        if self.auto_cool_task:
            self.auto_cool_task.cancel()
            try:
                await self.auto_cool_task
            except asyncio.CancelledError:
                pass

        # Cool backend if hot
        if self.backend.is_ready():
            await self.cool()

        print(f"[LLMRuntime] Stopped")

    async def warm(self) -> bool:
        """
        Warm backend (load model)

        Returns:
            True if warming successful
        """
        if self.backend.is_ready():
            print(f"[LLMRuntime] Backend already hot")
            return True

        print(f"[LLMRuntime] Warming {self.backend_type_str} backend...")
        success = await self.backend.warm()

        if success:
            print(f"[LLMRuntime] Backend warmed successfully")
        else:
            print(f"[LLMRuntime] Backend warming failed")

        return success

    async def cool(self) -> bool:
        """
        Cool backend (unload model)

        Returns:
            True if cooling successful
        """
        if self.backend.state == BackendState.COLD:
            print(f"[LLMRuntime] Backend already cold")
            return True

        print(f"[LLMRuntime] Cooling {self.backend_type_str} backend...")
        success = await self.backend.cool()

        if success:
            print(f"[LLMRuntime] Backend cooled successfully")
        else:
            print(f"[LLMRuntime] Backend cooling failed")

        return success

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from prompt

        Automatically warms backend if cold.
        Updates statistics and last request time.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LLMRequest parameters

        Returns:
            LLM response
        """
        # Warm backend if needed
        if not self.backend.is_ready():
            print(f"[LLMRuntime] Backend cold, warming...")
            warmed = await self.warm()
            if not warmed:
                return LLMResponse(
                    text='',
                    finish_reason='error',
                    tokens_generated=0,
                    inference_time_ms=0.0,
                    metadata={'error': 'Failed to warm backend'}
                )

        # Build request
        request = LLMRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Generate response
        response = await self.backend.generate(request)

        # Update statistics
        self.last_request_time = time.time()
        self.total_requests += 1
        self.total_tokens_generated += response.tokens_generated
        self.total_inference_time_ms += response.inference_time_ms

        return response

    async def _auto_cool_monitor(self):
        """
        Background task to auto-cool backend after idle timeout

        Runs continuously, checking every 10 seconds if backend
        has been idle longer than auto_cool_timeout_s.
        """
        print(f"[LLMRuntime] Auto-cool monitor started (timeout={self.auto_cool_timeout_s}s)")

        try:
            while True:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Skip if backend not hot
                if not self.backend.is_ready():
                    continue

                # Skip if no requests yet
                if self.last_request_time is None:
                    continue

                # Check idle time
                idle_time = time.time() - self.last_request_time
                if idle_time >= self.auto_cool_timeout_s:
                    print(f"[LLMRuntime] Auto-cooling after {idle_time:.1f}s idle")
                    await self.cool()

        except asyncio.CancelledError:
            print(f"[LLMRuntime] Auto-cool monitor stopped")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics"""
        stats = {
            'backend_type': self.backend_type_str,
            'backend_state': self.backend.state.value,
            'total_requests': self.total_requests,
            'total_tokens_generated': self.total_tokens_generated,
            'total_inference_time_ms': self.total_inference_time_ms,
            'last_request_time': self.last_request_time,
        }

        # Add average stats if requests made
        if self.total_requests > 0:
            stats['avg_tokens_per_request'] = self.total_tokens_generated / self.total_requests
            stats['avg_inference_time_ms'] = self.total_inference_time_ms / self.total_requests

        # Add idle time if applicable
        if self.last_request_time and self.backend.is_ready():
            stats['idle_time_s'] = time.time() - self.last_request_time

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Get health status"""
        backend_health = self.backend.health_check()
        runtime_stats = self.get_stats()

        return {
            'runtime': runtime_stats,
            'backend': backend_health,
        }

    def is_ready(self) -> bool:
        """Check if runtime is ready for inference"""
        return self.backend.is_ready()

    def get_state(self) -> BackendState:
        """Get backend state"""
        return self.backend.get_state()
