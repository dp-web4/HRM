"""
Transformers backend for LLM Runtime

Uses HuggingFace Transformers for local GPU inference.
Implements true hot/cold lifecycle with model loading/unloading.
"""

import time
import torch
import gc
from typing import Dict, Any, Optional

from .base import LLMBackend, LLMRequest, LLMResponse, BackendState

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TransformersBackend(LLMBackend):
    """
    Transformers backend implementation

    Loads models from HuggingFace Hub or local cache.
    Supports quantization (4-bit, 8-bit) for memory efficiency.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Transformers backend

        Config:
            - model_name: HF model ID (e.g., 'Qwen/Qwen2.5-3B-Instruct')
            - device: 'cuda', 'cpu' (default: 'cuda')
            - dtype: 'float16', 'bfloat16', 'float32' (default: 'float16')
            - load_in_4bit: Enable 4-bit quantization (default: False)
            - load_in_8bit: Enable 8-bit quantization (default: False)
            - max_memory_gb: Max GPU memory to use (default: None = unlimited)
            - trust_remote_code: Trust remote code (default: False)
        """
        super().__init__(config)

        self.dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }

        self.dtype = self.dtype_map.get(
            config.get('dtype', 'float16'),
            torch.float16
        )
        self.load_in_4bit = config.get('load_in_4bit', False)
        self.load_in_8bit = config.get('load_in_8bit', False)
        self.max_memory_gb = config.get('max_memory_gb', None)
        self.trust_remote_code = config.get('trust_remote_code', False)

        self.model = None
        self.tokenizer = None
        self.last_inference_time = None
        self.error_message = None
        self.memory_usage_mb = 0

        if not TRANSFORMERS_AVAILABLE:
            self.state = BackendState.ERROR
            self.error_message = "transformers not installed"
        else:
            self.state = BackendState.COLD

    async def warm(self) -> bool:
        """
        Load model and tokenizer into memory

        Returns:
            True if loading successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            return False

        if self.state == BackendState.HOT:
            print(f"[TransformersBackend] Already hot: {self.model_name}")
            return True

        try:
            self.state = BackendState.WARMING
            print(f"[TransformersBackend] Loading {self.model_name}...")

            # Configure quantization if requested
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print(f"[TransformersBackend] Using 4-bit quantization")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )

            # Load model
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_name,
                'torch_dtype': self.dtype,
                'device_map': 'auto' if self.device == 'cuda' else None,
                'trust_remote_code': self.trust_remote_code,
            }

            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            elif self.load_in_8bit:
                model_kwargs['load_in_8bit'] = True

            if self.max_memory_gb and self.device == 'cuda':
                model_kwargs['max_memory'] = {0: f"{self.max_memory_gb}GB"}

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            # Move to device if not using device_map
            if self.device == 'cpu' or (self.device == 'cuda' and not model_kwargs.get('device_map')):
                self.model.to(self.device)

            # Set to eval mode
            self.model.eval()

            # Estimate memory usage
            if self.device == 'cuda' and torch.cuda.is_available():
                self.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024

            self.state = BackendState.HOT
            print(f"[TransformersBackend] Loaded {self.model_name} ({self.memory_usage_mb:.0f}MB)")
            return True

        except Exception as e:
            self.state = BackendState.ERROR
            self.error_message = str(e)
            print(f"[TransformersBackend] Failed to load: {e}")
            return False

    async def cool(self) -> bool:
        """
        Unload model and tokenizer, free GPU memory

        Returns:
            True if unloading successful, False otherwise
        """
        try:
            self.state = BackendState.COOLING
            print(f"[TransformersBackend] Unloading {self.model_name}...")

            # Delete model and tokenizer
            if self.model:
                del self.model
                self.model = None

            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if on GPU
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self.memory_usage_mb = 0
            self.state = BackendState.COLD
            print(f"[TransformersBackend] Unloaded {self.model_name}")
            return True

        except Exception as e:
            self.state = BackendState.ERROR
            self.error_message = str(e)
            print(f"[TransformersBackend] Failed to unload: {e}")
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response using Transformers model

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
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors='pt',
                truncation=True,
                max_length=2048  # Leave room for generation
            ).to(self.device)

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True if request.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode output (skip input tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            inference_time_ms = (time.time() - start_time) * 1000
            self.last_inference_time = time.time()

            tokens_generated = len(generated_ids)
            finish_reason = 'stop'  # TODO: detect length/stop properly

            return LLMResponse(
                text=text,
                finish_reason=finish_reason,
                tokens_generated=tokens_generated,
                inference_time_ms=inference_time_ms,
                metadata={
                    'model': self.model_name,
                    'device': self.device,
                    'dtype': str(self.dtype),
                    'input_tokens': input_length,
                }
            )

        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            self.error_message = str(e)
            print(f"[TransformersBackend] Generate failed: {e}")

            return LLMResponse(
                text='',
                finish_reason='error',
                tokens_generated=0,
                inference_time_ms=inference_time_ms,
                metadata={'error': str(e)}
            )

    def health_check(self) -> Dict[str, Any]:
        """Check Transformers backend health"""
        health = {
            'backend': 'transformers',
            'state': self.state.value,
            'model': self.model_name,
            'device': self.device,
            'dtype': str(self.dtype),
            'memory_mb': self.memory_usage_mb,
            'last_inference_time': self.last_inference_time,
            'error': self.error_message,
            'transformers_available': TRANSFORMERS_AVAILABLE,
        }

        # Add GPU stats if available
        if self.device == 'cuda' and torch.cuda.is_available():
            health['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            health['cuda_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return health
