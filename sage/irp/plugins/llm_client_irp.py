"""LLMClientIRP — IRP-compatible wrapper for any LLMClient.

Mirrors DaemonIRP's interface (init_state / step / halt / get_response /
energy / health_check) but backs onto an LLMClient (OllamaClient,
ClaudeCLIClient, AnthropicClient) instead of the resident SAGE daemon.

This lets raising sessions use any gameplay-defined LLM backend by
swapping:

    from sage.irp.plugins.daemon_irp import DaemonIRP
    model = DaemonIRP({'daemon_host': '...', ...})

with:

    from sage.cognition.thalamic_router.llm_dispatch import OllamaClient
    from sage.irp.plugins.llm_client_irp import LLMClientIRP
    model = LLMClientIRP(OllamaClient(model='qwen3.5:14b'))

Same init_state/step contract; raising code keeps working unchanged.

Sprint: Phase 4 P1.1 (alignment — shared LLM abstraction)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sage.irp.unified_history import to_unified


class LLMClientIRP:
    """IRP-contract wrapper around any LLMClient.

    The LLMClient must implement:
        chat(prompt, history=None, system_prompt=None, images_png=None,
             max_tokens=...) -> str

    (OllamaClient / ClaudeCLIClient / AnthropicClient in llm_dispatch.py
    all satisfy this post-B4.)

    The IRP contract (init_state / step / halt / get_response / energy /
    health_check) mirrors DaemonIRP so existing raising code needs no
    changes to swap backends.
    """

    def __init__(
        self,
        llm_client: Any,
        system_prompt: str = "",
        max_new_tokens: int = 300,
    ) -> None:
        """
        Args:
            llm_client: any object with a chat() method matching the
                LLMClient signature
            system_prompt: default system prompt if init_state doesn't
                override
            max_new_tokens: cap on generated tokens per step
        """
        self.client = llm_client
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        # For raising scripts that call health_check() on startup.
        # LLMClients are sync/subprocess/HTTP; we can't poll them cheaply,
        # so report "alive" optimistically. First chat() call will fail
        # informatively if the backend is down.
        self._alive = True

    def health_check(self) -> Dict[str, Any]:
        """Mirror DaemonIRP.health_check() shape.

        Real health would require calling the backend — deferred to first
        chat() call. Reports optimistic alive unless the caller has reason
        to believe otherwise.
        """
        return {
            "status": "alive" if self._alive else "error",
            "backend": type(self.client).__name__,
            "model": getattr(self.client, "model", "?"),
        }

    def init_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """IRP: build processing state from input context.

        Accepts DaemonIRP-compatible context:
            prompt: str — the new user turn
            memory: List — prior turns (any supported schema; normalized)
            system_prompt: str (optional) — override default
        """
        prompt = context.get("prompt", "")
        memory = context.get("memory", []) or []
        system = context.get("system_prompt") or self.system_prompt
        # Normalize memory to canonical {role, content}
        history = to_unified(memory)

        return {
            "prompt": prompt,
            "memory": history,                 # canonical form
            "system_prompt": system,
            "current_response": "",
            "iteration": 0,
            "max_iterations": 1,               # single-shot; no refinement loop
            "energy": 1.0,
        }

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """IRP: one generation step. Updates state in place + returns it."""
        prompt = state["prompt"]
        history = state.get("memory", [])
        system = state.get("system_prompt", "")

        try:
            response = self.client.chat(
                prompt,
                history=history if history else None,
                system_prompt=system or None,
                max_tokens=self.max_new_tokens,
            )
        except TypeError:
            # Backend LLMClient may not support history/system_prompt kwargs
            # (older signature). Fall back to single-shot.
            response = self.client.chat(prompt, max_tokens=self.max_new_tokens)
        except Exception as e:
            self._alive = False
            response = f"[LLMClientIRP error: {type(e).__name__}: {e}]"

        state["current_response"] = response
        state["energy"] = 0.0
        state["iteration"] = state.get("iteration", 0) + 1
        return state

    def halt(self, state: Dict[str, Any]) -> bool:
        """IRP: always halt after one step (single-shot generation)."""
        return state.get("iteration", 0) >= state.get("max_iterations", 1)

    def get_response(self, state: Dict[str, Any]) -> str:
        """IRP: extract final response."""
        return state.get("current_response", "")

    def energy(self, state: Dict[str, Any]) -> float:
        """IRP: current energy level (1.0 before step, 0.0 after)."""
        return state.get("energy", 1.0)
