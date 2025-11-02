"""
Qwen Alive IRP Plugin - Continuous learning reasoning
Never-converging epistemic pragmatism via fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from sage.irp.base import IRPPlugin


class QwenAliveIRP(IRPPlugin):
    """
    Qwen "alive" reasoning via IRP

    Characteristics:
    - Continuous learning (never converges)
    - Epistemic pragmatism (questions > assertions)
    - Adapts from each experience
    - Higher memory/compute cost

    Use when: Depth > speed, novel situations, "cliff edge" decisions
    """

    def __init__(
        self,
        model_path: str = "/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        config: Dict[str, Any] = None
    ):
        super().__init__(config or {
            'max_iterations': 1,  # Single inference per call
            'halt_eps': 0.0,
            'entity_id': 'qwen_alive_reasoning'
        })

        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Model state
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')  # Jetson uses CPU for transformers

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
        """Load Qwen fine-tuned model (full model, not LoRA)"""
        import time
        start = time.time()

        # Load tokenizer from model path
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        # Load full fine-tuned model (epistemic-pragmatism is a merged model)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float32,  # CPU needs float32
            trust_remote_code=True,
            device_map='cpu'
        )
        self.model.eval()

        load_time = time.time() - start
        print(f"[QwenAliveIRP] Model loaded in {load_time:.1f}s")

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
        Execute Qwen inference
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
        Energy for Qwen alive

        Lower energy = more questions (epistemic pragmatism)
        Higher energy = fewer questions (converging to certainty)
        """
        if self.response is None:
            return 1.0

        # Count question marks as proxy for epistemic openness
        num_questions = self.response.count('?')

        # Very short responses = high energy (failure mode)
        if len(self.response) < 10:
            return 0.9

        # More questions = lower energy (alive/learning)
        # Fewer questions = higher energy (dead/certain)
        if num_questions >= 3:
            return 0.1  # Very alive
        elif num_questions >= 1:
            return 0.3  # Moderately alive
        else:
            return 0.6  # Converging (not ideal for "alive" model)

    def halt(self, energies: List[float], t: int) -> bool:
        """Halt after single inference"""
        return t >= 1

    def get_result(self) -> Dict[str, Any]:
        """Return final result"""
        return {
            'response': self.response,
            'tokens_generated': self.tokens_generated,
            'inference_time_sec': self.inference_time,
            'model': 'Qwen2.5-0.5B-alive',
            'mode': 'continuous_learning'
        }

    def get_cost(self) -> Dict[str, float]:
        """
        Resource cost metrics
        """
        return {
            'time_sec': self.inference_time,
            'tokens': self.tokens_generated,
            # Qwen 0.5B + LoRA ~2GB loaded, plus generation overhead
            'memory_mb_estimate': 2000 + (self.tokens_generated * 0.5)
        }


def create_qwen_alive_reasoning() -> QwenAliveIRP:
    """
    Create Qwen alive reasoning plugin

    Returns:
        QwenAliveIRP instance (model loaded lazily)
    """
    return QwenAliveIRP()
