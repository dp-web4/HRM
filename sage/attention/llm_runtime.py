#!/usr/bin/env python3
"""
LLM Runtime Service (Tier 1)

On-demand LLM inference for deep reasoning during THINK state.
Supports multiple backends with hot/cold lifecycle management.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
import time
import asyncio


class RuntimeState(Enum):
    """LLM runtime lifecycle states"""
    COLD = "cold"        # Not loaded, minimal memory
    WARMING = "warming"  # Loading model into memory
    HOT = "hot"          # Ready for inference
    COOLING = "cooling"  # Unloading model


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    async def load(self) -> bool:
        """Load model into memory"""
        pass

    @abstractmethod
    async def unload(self) -> bool:
        """Unload model from memory"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available on this system"""
        pass


class OllamaBackend(LLMBackend):
    """Ollama backend for local inference"""

    def __init__(self, model_name: str = "qwen2.5:0.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.loaded = False

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    async def load(self) -> bool:
        """Load model (Ollama keeps models loaded)"""
        if not self.is_available():
            return False

        # Ollama doesn't need explicit loading, but we can check if model exists
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if self.model_name not in model_names:
                print(f"[OllamaBackend] Model {self.model_name} not found. Available: {model_names}")
                return False

            self.loaded = True
            return True
        except Exception as e:
            print(f"[OllamaBackend] Load error: {e}")
            return False

    async def unload(self) -> bool:
        """Unload model (no-op for Ollama)"""
        self.loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text using Ollama API"""
        if not self.loaded:
            await self.load()

        try:
            import requests

            payload = {
                'model': self.model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            }

            if stop_sequences:
                payload['options']['stop'] = stop_sequences

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )

            result = response.json()
            return result.get('response', '')

        except Exception as e:
            print(f"[OllamaBackend] Generate error: {e}")
            return f"[Error: {e}]"


class TransformersBackend(LLMBackend):
    """Transformers backend for local inference"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def is_available(self) -> bool:
        """Check if transformers is available"""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    async def load(self) -> bool:
        """Load model with transformers"""
        if self.loaded:
            return True

        if not self.is_available():
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"[TransformersBackend] Loading {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )

            self.loaded = True
            print(f"[TransformersBackend] Model loaded on {device}")
            return True

        except Exception as e:
            print(f"[TransformersBackend] Load error: {e}")
            return False

    async def unload(self) -> bool:
        """Unload model from memory"""
        if not self.loaded:
            return True

        try:
            import torch

            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.loaded = False
            print("[TransformersBackend] Model unloaded")
            return True

        except Exception as e:
            print(f"[TransformersBackend] Unload error: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text using transformers"""
        if not self.loaded:
            await self.load()

        if not self.model or not self.tokenizer:
            return "[Error: Model not loaded]"

        try:
            import torch

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]

            return generated_text.strip()

        except Exception as e:
            print(f"[TransformersBackend] Generate error: {e}")
            return f"[Error: {e}]"


class LLMRuntime:
    """
    LLM Runtime Service (Tier 1)

    Manages on-demand LLM inference with hot/cold lifecycle.
    Invoked by Tier 0 kernel during THINK state.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        self.state = RuntimeState.COLD
        self.backend: Optional[LLMBackend] = None

        # Configuration
        self.backend_type = config.get('backend', 'auto')  # 'auto', 'ollama', 'transformers'
        self.model_name = config.get('model_name', None)
        self.auto_unload_timeout = config.get('auto_unload_timeout', 300)  # 5 minutes

        # Lifecycle tracking
        self.last_use_time = 0
        self.total_invocations = 0
        self.total_tokens_generated = 0

        # Initialize backend
        self._init_backend()

    def _init_backend(self):
        """Initialize the appropriate backend"""
        if self.backend_type == 'auto':
            # Try Ollama first, fall back to Transformers
            ollama = OllamaBackend()
            if ollama.is_available():
                self.backend = ollama
                self.backend_type = 'ollama'
                print("[LLMRuntime] Using Ollama backend")
            else:
                transformers = TransformersBackend()
                if transformers.is_available():
                    self.backend = transformers
                    self.backend_type = 'transformers'
                    print("[LLMRuntime] Using Transformers backend")
                else:
                    print("[LLMRuntime] No backends available")
                    self.backend = None

        elif self.backend_type == 'ollama':
            model_name = self.model_name or "qwen2.5:0.5b"
            self.backend = OllamaBackend(model_name=model_name)

        elif self.backend_type == 'transformers':
            model_name = self.model_name or "Qwen/Qwen2.5-0.5B-Instruct"
            self.backend = TransformersBackend(model_name=model_name)

        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")

    async def warmup(self) -> bool:
        """Warm up runtime (load model)"""
        if self.state == RuntimeState.HOT:
            return True

        if not self.backend:
            return False

        self.state = RuntimeState.WARMING
        success = await self.backend.load()

        if success:
            self.state = RuntimeState.HOT
            print(f"[LLMRuntime] Warmed up ({self.backend_type})")
        else:
            self.state = RuntimeState.COLD
            print(f"[LLMRuntime] Warmup failed")

        return success

    async def cooldown(self) -> bool:
        """Cool down runtime (unload model)"""
        if self.state == RuntimeState.COLD:
            return True

        if not self.backend:
            return False

        self.state = RuntimeState.COOLING
        success = await self.backend.unload()

        if success:
            self.state = RuntimeState.COLD
            print(f"[LLMRuntime] Cooled down")
        else:
            print(f"[LLMRuntime] Cooldown failed")

        return success

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional stop sequences

        Returns:
            Generated text
        """
        # Ensure runtime is warm
        if self.state != RuntimeState.HOT:
            await self.warmup()

        if self.state != RuntimeState.HOT:
            return "[Error: Runtime failed to warm up]"

        # Generate
        start_time = time.time()
        result = await self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences
        )
        duration = time.time() - start_time

        # Update stats
        self.last_use_time = time.time()
        self.total_invocations += 1
        self.total_tokens_generated += len(result.split())

        print(f"[LLMRuntime] Generated {len(result.split())} tokens in {duration:.2f}s")

        return result

    async def check_auto_unload(self):
        """Check if runtime should auto-unload due to inactivity"""
        if self.state != RuntimeState.HOT:
            return

        if self.auto_unload_timeout <= 0:
            return  # Auto-unload disabled

        idle_time = time.time() - self.last_use_time
        if idle_time > self.auto_unload_timeout:
            print(f"[LLMRuntime] Auto-unloading after {idle_time:.0f}s idle")
            await self.cooldown()

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics"""
        return {
            'state': self.state.value,
            'backend': self.backend_type,
            'total_invocations': self.total_invocations,
            'total_tokens_generated': self.total_tokens_generated,
            'idle_seconds': time.time() - self.last_use_time if self.last_use_time > 0 else 0
        }
