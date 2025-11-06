"""
Qwen 7B IRP Plugin - Larger model for comparison
Testing if size matters for conversational learning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from sage.irp.base import IRPPlugin


class Qwen7BIRP(IRPPlugin):
    """
    Qwen 7B reasoning via IRP

    Testing hypothesis: Does 14x larger model (7B vs 0.5B) show:
    - Better conversational quality?
    - Different trust evolution patterns?
    - Worth the 7x memory cost?

    Characteristics:
    - 7B parameters (vs 0.5B in QwenAliveIRP)
    - Same instruct-tuned base
    - ~14GB RAM required (vs ~2GB for 0.5B)
    - Slower inference but potentially deeper reasoning
    """

    def __init__(
        self,
        model_path: str = "/home/dp/ai-workspace/HRM/model-zoo/sage/qwen2.5-7b-instruct",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        config: Dict[str, Any] = None
    ):
        super().__init__(config or {
            'max_iterations': 1,  # Single inference per call
            'halt_eps': 0.0,
            'entity_id': 'qwen_7b_reasoning'
        })

        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Model state
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')  # CPU for now, could use GPU

        # Inference state
        self.prompt = None
        self.response = None
        self.tokens_generated = 0
        self.inference_time = 0.0

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize - load model lazily on first use"""
        self.max_new_tokens = config.get('max_new_tokens', self.max_new_tokens)
        self.temperature = config.get('temperature', self.temperature)

        # Load model if not already loaded
        if self.model is None:
            self._load_model()

    def _load_model(self):
        """Load Qwen 7B model"""
        import time
        start = time.time()

        print(f"[Qwen7BIRP] Loading 7B model from {self.model_path}")
        print("[Qwen7BIRP] This will use ~14GB RAM...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        # Load model with GPU acceleration
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,  # Half precision for efficiency
            trust_remote_code=True,
            device_map='auto',  # Use GPU automatically
            low_cpu_mem_usage=True
        )
        self.model.eval()

        load_time = time.time() - start
        print(f"[Qwen7BIRP] 7B model loaded in {load_time:.1f}s")

    def preprocess(self, x: Any) -> Dict[str, Any]:
        """
        Convert input to prompt
        Input: str (question/task) or dict with 'prompt' key
        """
        if isinstance(x, str):
            self.prompt = x
        elif isinstance(x, dict):
            self.prompt = x.get('prompt', x.get('question', str(x)))
        else:
            self.prompt = str(x)

        return {'prompt': self.prompt}

    def step(self, x_t: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Execute Qwen 7B inference
        Single-shot per IRP call
        """
        import time

        if self.model is None:
            self._load_model()

        start = time.time()

        # Tokenize
        inputs = self.tokenizer(
            x_t['prompt'],
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part (after prompt)
        if x_t['prompt'] in full_output:
            self.response = full_output[len(x_t['prompt']):].strip()
        else:
            self.response = full_output.strip()

        self.tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        self.inference_time = time.time() - start

        return {
            'response': self.response,
            'tokens': self.tokens_generated,
            'time': self.inference_time
        }

    def energy(self, x_t: Dict[str, Any], t: int) -> float:
        """
        Energy for Qwen 7B

        Use same metric as 0.5B for fair comparison:
        - Response quality (length, coherence)
        - Conversational engagement
        """
        if self.response is None:
            return 1.0

        # Very short responses = high energy (failure mode)
        if len(self.response) < 10:
            return 0.9

        # Very long = might be rambling (moderate energy)
        if len(self.response) > 500:
            return 0.4

        # Check for engagement markers
        engagement_markers = [
            '?',  # Questions (curiosity)
            '!',  # Emphasis (emotion)
            'I ',  # First person (personal)
            'you',  # Second person (addressing user)
        ]

        engagement_count = sum(1 for marker in engagement_markers if marker in self.response)

        # More engagement = lower energy (better)
        if engagement_count >= 4:
            return 0.1  # Highly engaged
        elif engagement_count >= 2:
            return 0.2  # Moderately engaged
        elif engagement_count >= 1:
            return 0.4  # Some engagement
        else:
            return 0.6  # Low engagement (factual/dry)

    def halt(self, energies: List[float], t: int) -> bool:
        """Halt after single inference"""
        return t >= 1

    def get_result(self) -> Dict[str, Any]:
        """Return final result"""
        return {
            'response': self.response,
            'tokens_generated': self.tokens_generated,
            'inference_time_sec': self.inference_time,
            'model': 'Qwen2.5-7B-Instruct',
            'model_size': '7B parameters',
            'mode': 'instruct'
        }

    def get_cost(self) -> Dict[str, float]:
        """
        Resource cost metrics
        """
        return {
            'time_sec': self.inference_time,
            'tokens': self.tokens_generated,
            # Qwen 7B in fp16 ~14GB loaded, plus generation overhead
            'memory_mb_estimate': 14000 + (self.tokens_generated * 2.0)
        }


def create_qwen_7b_reasoning() -> Qwen7BIRP:
    """
    Create Qwen 7B reasoning plugin

    Returns:
        Qwen7BIRP instance (model loaded lazily)
    """
    return Qwen7BIRP()
