#!/usr/bin/env python3
"""
Qwen3.5-27B IRP Plugin with LoRA Training Support

Provides:
- HuggingFace transformers-based inference
- PEFT LoRA adapter training
- Sleep cycle learning integration
- Multimodal support (text + images + video)
- Experience-driven adaptation

This is the training-capable version for continuous learning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime


class Qwen35_27B_LoRA_IRP:
    """
    IRP-compliant plugin for Qwen3.5-27B with LoRA training.

    Architecture: Hybrid SSM+Attention (Gated DeltaNet)
    - Native multimodal: text + images + video
    - 262K context window
    - 27B parameters
    - LoRA fine-tuning for continuous learning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with IRP protocol compliance and LoRA config."""
        self.config = config or {}

        # Model paths
        default_path = str(Path(__file__).parent.parent.parent.parent / 'model-zoo' / 'qwen3.5-27b-transformers')
        self.model_path = self.config.get('model_path', default_path)

        # Instance directory for LoRA adapters and training state
        self.instance_dir = Path(self.config.get('instance_dir', '~/ai-workspace/SAGE/sage/instances/thor-qwen3.5-27b')).expanduser()
        self.lora_dir = self.instance_dir / 'lora_adapters'
        self.lora_dir.mkdir(parents=True, exist_ok=True)

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=self.config.get('lora_r', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            target_modules=self.config.get('lora_target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training state
        self.training_enabled = self.config.get('training_enabled', False)
        self.active_adapter = self.config.get('active_adapter', None)
        self.experience_buffer = []

        # Multimodal support
        self.multimodal_enabled = self.config.get('multimodal_enabled', True)

        # State management
        self.state = None
        self.conversation_memory = []
        self.refinement_history = []

        # IRP protocol tracking
        self.energy_history = []
        self.trust_score = 0.5

        # Device settings
        self.device = self._get_device()

        # Load model
        self._load_model()

        # Load training state if exists
        if self.training_enabled:
            self._load_training_state()

        print(f"Qwen3.5-27B LoRA IRP Plugin initialized")
        print(f"  Model: {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  LoRA: {'enabled' if self.training_enabled else 'disabled'}")
        print(f"  Multimodal: {'enabled' if self.multimodal_enabled else 'disabled'}")
        print(f"  Active adapter: {self.active_adapter or 'None'}")
        print(f"  Trust: {self.trust_score:.3f}")

    def _get_device(self):
        """Determine device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        """Load base model and optionally apply LoRA adapter."""
        print(f"Loading Qwen3.5-27B from {self.model_path}...")

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run download_transformers.py first."
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )

        # Load base model
        # Use BF16 on CUDA, FP32 on CPU
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        try:
            if self.multimodal_enabled:
                # Try to load as multimodal model
                print("Loading as multimodal Qwen2VL model...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            else:
                # Load as text-only model
                print("Loading as text-only model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"Multimodal loading failed: {e}")
            print("Falling back to text-only mode...")
            self.multimodal_enabled = False
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

        # Move to device if not using device_map="auto"
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        # Apply LoRA if training enabled
        if self.training_enabled:
            print("Preparing model for LoRA training...")
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, self.lora_config)

            # Load active adapter if specified
            if self.active_adapter:
                adapter_path = self.lora_dir / self.active_adapter
                if adapter_path.exists():
                    print(f"Loading LoRA adapter: {self.active_adapter}")
                    self.model.load_adapter(str(adapter_path))
                else:
                    print(f"Adapter not found: {self.active_adapter}, starting fresh")

            self.model.print_trainable_parameters()

        self.model.eval()
        print("Model loaded successfully")

    def _load_training_state(self):
        """Load training state from instance directory."""
        state_file = self.instance_dir / 'training' / 'state.json'
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self.active_adapter = state.get('active_adapter')
            print(f"Training state loaded: {state_file}")

    def _save_training_state(self):
        """Save training state to instance directory."""
        state_file = self.instance_dir / 'training' / 'state.json'
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
        else:
            state = {
                "training_enabled": True,
                "backend": "pytorch",
                "lora_adapters": [],
                "sleep_cycles": {
                    "total_cycles": 0,
                    "last_consolidation": None,
                    "pending_experiences": 0,
                },
            }

        # Update with current state
        state['active_adapter'] = self.active_adapter
        state['last_updated'] = datetime.now().isoformat()

        # Write back
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def init_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Initialize processing state.

        Args:
            context: Input context with conversation history, memory, etc.

        Returns:
            Initial state for iterative refinement
        """
        prompt = context.get('prompt', '')
        memory = context.get('memory', [])
        images = context.get('images', [])  # Multimodal support

        # Build full context
        full_context = self._build_context(prompt, memory)

        # Initialize state
        self.state = {
            'prompt': prompt,
            'full_context': full_context,
            'current_response': '',
            'iteration': 0,
            'energy': 1.0,
            'convergence_threshold': 0.1,
            'max_iterations': 3,  # Qwen3.5 is strong, fewer iterations needed
            'memory': memory,
            'images': images,
            'refinement_log': []
        }

        return self.state

    def _build_context(self, current_prompt: str, memory: List[Dict]) -> str:
        """Build conversation context from memory and current prompt."""
        context_parts = []

        # Add recent conversation history (last 5 turns)
        for turn in memory[-5:]:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            context_parts.append(f"{role}: {content}")

        # Add current prompt
        context_parts.append(f"user: {current_prompt}")

        return "\n".join(context_parts)

    def refine(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Perform one refinement iteration.

        Args:
            state: Current processing state

        Returns:
            Updated state after refinement
        """
        # Generate response
        response = self._generate(state['full_context'])
        state['current_response'] = response
        state['iteration'] += 1

        # Energy decay (Qwen3.5 is stable, less noise)
        state['energy'] *= 0.7

        # Log refinement
        state['refinement_log'].append({
            'iteration': state['iteration'],
            'energy': state['energy'],
            'response_length': len(response),
        })

        return state

    def _generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text response using the model."""
        # TEMPORARY: Use simple prompt formatting until CUDA issue is resolved
        # The Qwen3.5 model has tokenization issues with chat templates
        text = f"Q: {prompt}\nA:"

        print(f"Generating response for prompt length: {len(prompt)}")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        print(f"Input IDs shape: {inputs['input_ids'].shape}, max ID: {inputs['input_ids'].max().item()}, vocab size: {len(self.tokenizer)}")

        # Use greedy decoding only for now (no sampling to avoid CUDA errors)
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 100),  # Start with shorter generation
                    do_sample=False,  # Greedy only
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                print(f"Generation successful, output shape: {outputs.shape}")
            except Exception as e:
                print(f"Generation error: {type(e).__name__}: {e}")
                # Return error message instead of crashing
                return f"[Generation failed: {type(e).__name__}. This may be a tokenization or CUDA issue with this model.]"

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded response length: {len(response)}")

        # Remove the input prompt from response
        if response.startswith(text):
            response = response[len(text):].strip()

        return response if response else "[Empty response generated]"

    def converged(self, state: Dict[str, Any]) -> bool:
        """
        IRP Protocol: Check if refinement has converged.

        Args:
            state: Current processing state

        Returns:
            True if converged, False otherwise
        """
        # Qwen3.5 is strong - converge quickly
        if state['iteration'] >= 2:
            return True

        if state['energy'] < state['convergence_threshold']:
            return True

        return False

    def finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Finalize processing and return result.

        Args:
            state: Final processing state

        Returns:
            Final result with response and metadata
        """
        result = {
            'response': state['current_response'],
            'iterations': state['iteration'],
            'final_energy': state['energy'],
            'refinement_log': state['refinement_log'],
            'trust_score': self.trust_score,
        }

        # Update conversation memory
        self.conversation_memory.append({
            'role': 'user',
            'content': state['prompt']
        })
        self.conversation_memory.append({
            'role': 'assistant',
            'content': state['current_response']
        })

        # Track energy for trust adjustment
        self.energy_history.append(state['final_energy'])

        # Collect experience for training if enabled
        if self.training_enabled:
            self._collect_experience(state, result)

        return result

    def _collect_experience(self, state: Dict[str, Any], result: Dict[str, Any]):
        """Collect experience for sleep cycle learning."""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'prompt': state['prompt'],
            'response': result['response'],
            'iterations': result['iterations'],
            'energy': result['final_energy'],
            'context': state['full_context'],
        }
        self.experience_buffer.append(experience)

        # Trigger sleep cycle if threshold reached
        threshold = self.config.get('consolidation_threshold', 100)
        if len(self.experience_buffer) >= threshold:
            print(f"Sleep cycle threshold reached ({len(self.experience_buffer)} experiences)")
            # This would trigger dream bundle generation in the main loop

    def save_lora_adapter(self, adapter_name: str):
        """Save current LoRA adapter."""
        if not self.training_enabled:
            print("Training not enabled, cannot save adapter")
            return

        adapter_path = self.lora_dir / adapter_name
        self.model.save_pretrained(str(adapter_path))
        print(f"LoRA adapter saved: {adapter_path}")

        self.active_adapter = adapter_name
        self._save_training_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current status for monitoring."""
        return {
            'model': 'Qwen3.5-27B',
            'device': self.device,
            'training_enabled': self.training_enabled,
            'active_adapter': self.active_adapter,
            'multimodal': self.multimodal_enabled,
            'trust_score': self.trust_score,
            'conversation_turns': len(self.conversation_memory),
            'experiences_collected': len(self.experience_buffer),
        }


# IRP protocol compliance check
def validate_irp_protocol():
    """Verify that this plugin implements the IRP protocol correctly."""
    required_methods = ['init_state', 'refine', 'converged', 'finalize']
    for method in required_methods:
        assert hasattr(Qwen35_27B_LoRA_IRP, method), f"Missing IRP method: {method}"
    print("✓ IRP protocol validation passed")


if __name__ == "__main__":
    validate_irp_protocol()
    print("Qwen3.5-27B LoRA IRP plugin ready")
