#!/usr/bin/env python3
"""
Model Backend — abstract LLM interface with Ollama and interactive backends.

Part of SAGE Solver v11 modular architecture.

Backends:
  OllamaBackend         — calls Ollama /api/chat, optional vision via images[]
  ClaudeInteractiveBackend — prints context to stdout for Claude Code sessions
"""

import os
import sys
from abc import ABC, abstractmethod

import requests

# Use the canonical palette from arc_vision.py
sys.path.insert(0, os.path.dirname(__file__))
from arc_vision import ARC_PALETTE


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Vision-capable model families
_VISION_FAMILIES = ("gemma4", "llava", "llama3.2-vision", "minicpm-v")

# Thinking-capable model families (native CoT)
_THINKING_FAMILIES = ("gemma4",)


class ModelBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def query(self, prompt: str, max_tokens: int = -1,
              image_b64: str = None) -> str:
        """Send a prompt to the model and return the response text."""

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this backend can process images."""

    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this backend has native chain-of-thought."""

    @abstractmethod
    def warmup(self) -> bool:
        """Send a warmup request. Returns True if model is ready."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""


class OllamaBackend(ModelBackend):
    """Wraps Ollama /api/chat with optional vision via images[]."""

    def __init__(self, model: str = "", ollama_url: str = ""):
        self._url = ollama_url or OLLAMA_URL
        self._model = model
        if not self._model:
            self._auto_detect()

    def _auto_detect(self):
        """Auto-detect best available Ollama model."""
        try:
            resp = requests.get(f"{self._url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            models = []

        preferred = [
            "gemma4:e4b", "gemma3:12b", "phi4:14b",
            "qwen3.5:0.8b", "qwen2.5:3b", "qwen3.5:2b",
        ]
        for p in preferred:
            if p in models:
                self._model = p
                return

        # Fallback: first non-embedding model
        chat_models = [m for m in models if "embed" not in m]
        self._model = chat_models[0] if chat_models else "gemma4:e4b"

    def query(self, prompt: str, max_tokens: int = -1,
              image_b64: str = None) -> str:
        opts = {"temperature": 0.3}
        # Small models get a tighter generation cap
        if max_tokens == -1 and any(
                s in self._model for s in ["0.8b", "0.5b", "1b", "2b", "3b"]):
            opts["num_predict"] = 300
        elif max_tokens > 0:
            opts["num_predict"] = max_tokens

        msg = {"role": "user", "content": prompt}
        if image_b64 and self.supports_vision():
            msg["images"] = [image_b64]

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [msg],
            "options": opts,
        }
        if not self.supports_thinking():
            payload["think"] = False

        try:
            resp = requests.post(
                f"{self._url}/api/chat", json=payload, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    return content
                # Fallback to thinking field for thinking models
                return data.get("message", {}).get("thinking", "").strip()
        except Exception as e:
            return f"[error: {e}]"
        return ""

    def supports_vision(self) -> bool:
        return any(fam in self._model for fam in _VISION_FAMILIES)

    def supports_thinking(self) -> bool:
        return any(fam in self._model for fam in _THINKING_FAMILIES)

    def warmup(self) -> bool:
        result = self.query("ready")
        return "error" not in result.lower()

    def model_name(self) -> str:
        return self._model


class ClaudeInteractiveBackend(ModelBackend):
    """Interactive backend: prints context to stdout for Claude Code to read.

    Does NOT call an LLM. Claude Code (the interactive session) reads the
    printed context and responds via the session. query() returns empty string.
    """

    def query(self, prompt: str, max_tokens: int = -1,
              image_b64: str = None) -> str:
        print("\n" + "=" * 60)
        print("CONTEXT FOR CLAUDE:")
        print("=" * 60)
        print(prompt)
        if image_b64:
            print(f"\n[IMAGE: {len(image_b64)} bytes base64 — "
                  f"see /tmp/sage_solver/frame.png]")
        print("=" * 60)
        return ""

    def supports_vision(self) -> bool:
        # Claude Code can see images via the session
        return True

    def supports_thinking(self) -> bool:
        return True

    def warmup(self) -> bool:
        return True

    def model_name(self) -> str:
        return "claude-interactive"
