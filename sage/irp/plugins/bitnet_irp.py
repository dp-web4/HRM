"""
BitNet IRP Plugin - Ultra-compressed edge reasoning
1.58-bit quantized inference via llama.cpp
"""

import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from sage.irp.base import IRPPlugin


class BitNetIRP(IRPPlugin):
    """
    BitNet reasoning via IRP

    Characteristics:
    - Ultra-fast (1.58-bit quantization)
    - Low memory footprint
    - CPU or GPU
    - Good for quick approximations

    Use when: Speed > depth, energy constrained, "flat ground" decisions
    """

    def __init__(
        self,
        model_path: str = "/home/dp/ai-workspace/BitNet/models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
        llama_bin: str = "/home/dp/ai-workspace/BitNet/build/bin/llama-cli",
        use_gpu: bool = False,
        max_tokens: int = 150,
        temperature: float = 0.7,
        config: Dict[str, Any] = None
    ):
        super().__init__(config or {
            'max_iterations': 1,  # Single-shot inference
            'halt_eps': 0.0,
            'entity_id': 'bitnet_reasoning'
        })

        self.model_path = Path(model_path)
        self.llama_bin = Path(llama_bin)
        self.use_gpu = use_gpu
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Verify resources
        if not self.model_path.exists():
            raise FileNotFoundError(f"BitNet model not found: {model_path}")
        if not self.llama_bin.exists():
            raise FileNotFoundError(f"llama-cli not found: {llama_bin}")

        # State
        self.prompt = None
        self.response = None
        self.tokens_generated = 0
        self.inference_time = 0.0

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize for specific task"""
        self.max_tokens = config.get('max_tokens', self.max_tokens)
        self.temperature = config.get('temperature', self.temperature)
        self.use_gpu = config.get('use_gpu', self.use_gpu)

    def preprocess(self, x: Any) -> Dict[str, Any]:
        """
        Convert input to prompt
        Input: str (question/task) or dict with 'prompt' key
        Output: dict with preprocessed prompt
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
        Execute BitNet inference
        Single-shot: t always 0
        """
        import time

        start = time.time()

        # Build llama-cli command
        cmd = [
            str(self.llama_bin),
            "-m", str(self.model_path),
            "-p", x_t['prompt'],
            "-n", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--repeat-penalty", "1.1",
            "-ngl", "99" if self.use_gpu else "0",
            "--no-mmap"
        ]

        try:
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output
            if result.returncode == 0:
                # Extract generated text (after prompt)
                output = result.stdout

                # Find the actual response (after prompt echo)
                # llama-cli echoes the prompt then generates
                if x_t['prompt'] in output:
                    self.response = output.split(x_t['prompt'], 1)[1].strip()
                else:
                    self.response = output.strip()

                # Extract token count from perf stats if available
                if "eval time" in output:
                    # Parse: "eval time = X ms / Y runs"
                    for line in output.split('\n'):
                        if 'eval time' in line and 'runs' in line:
                            try:
                                runs = int(line.split('/')[1].split('runs')[0].strip())
                                self.tokens_generated = runs
                            except:
                                pass
            else:
                self.response = f"[Error: {result.stderr[:200]}]"

        except subprocess.TimeoutExpired:
            self.response = "[Timeout]"
        except Exception as e:
            self.response = f"[Exception: {str(e)[:200]}]"

        self.inference_time = time.time() - start

        return {
            'response': self.response,
            'tokens': self.tokens_generated,
            'time': self.inference_time
        }

    def energy(self, x_t: Dict[str, Any], t: int) -> float:
        """
        Energy metric for BitNet

        BitNet is single-shot, so energy is based on:
        - Did it generate a response? (0.0 if yes, 1.0 if error)
        - Inference time (as proxy for confidence)
        """
        if self.response is None:
            return 1.0  # Not run yet

        # Error responses have high energy
        if self.response.startswith('[Error') or self.response.startswith('[Timeout'):
            return 0.9

        # Very short responses might indicate failure
        if len(self.response) < 10:
            return 0.7

        # Success - low energy
        # Could add temperature-based refinement, but BitNet is single-shot
        return 0.1

    def halt(self, energies: List[float], t: int) -> bool:
        """
        BitNet halts after first inference (single-shot)
        """
        return t >= 1  # Always halt after step 0

    def get_result(self) -> Dict[str, Any]:
        """Return final result"""
        return {
            'response': self.response,
            'tokens_generated': self.tokens_generated,
            'inference_time_sec': self.inference_time,
            'model': 'BitNet-1.58b-2.4B',
            'mode': 'GPU' if self.use_gpu else 'CPU'
        }

    def get_cost(self) -> Dict[str, float]:
        """
        Resource cost metrics
        Returns: {time, memory_estimate, tokens}
        """
        return {
            'time_sec': self.inference_time,
            'tokens': self.tokens_generated,
            # BitNet uses ~1.1GB model + inference overhead
            'memory_mb_estimate': 1100 + (self.tokens_generated * 0.1)
        }


# Convenience function for SAGE orchestrator
def create_bitnet_reasoning(use_gpu: bool = False) -> BitNetIRP:
    """
    Create BitNet reasoning plugin

    Args:
        use_gpu: Use GPU acceleration (requires fixed CUDA bug)

    Returns:
        BitNetIRP instance
    """
    return BitNetIRP(use_gpu=use_gpu)
