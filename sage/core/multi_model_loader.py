"""
Multi-Model Loader for SAGE

Enables dynamic model selection based on task complexity.
Thor can load multiple models simultaneously (0.5B, 14B, 72B) and route
tasks to the appropriate model size.

Architecture:
- Small tasks → 0.5B (fast, efficient)
- Medium tasks → 14B (balanced)
- Complex tasks → 72B (maximum capability)

Memory management ensures models are loaded/unloaded as needed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Available model sizes"""
    SMALL = "0.5b"   # Qwen2.5-0.5B
    MEDIUM = "14b"   # Qwen2.5-14B
    LARGE = "72b"    # Qwen2.5-72B (future)


class TaskComplexity(Enum):
    """Task complexity levels for routing"""
    SIMPLE = "simple"       # Factual recall, simple questions
    MODERATE = "moderate"   # Reasoning, explanation
    COMPLEX = "complex"     # Multi-step reasoning, philosophy
    VERY_COMPLEX = "very_complex"  # Research-level reasoning


@dataclass
class ModelConfig:
    """Configuration for a loaded model"""
    size: ModelSize
    path: Path
    model: Optional[torch.nn.Module] = None
    tokenizer: Optional[AutoTokenizer] = None
    loaded: bool = False
    memory_usage_gb: float = 0.0


class MultiModelLoader:
    """
    Manages multiple models with dynamic loading/unloading.

    Thor instance can have:
    - 0.5B always loaded (fast, minimal memory)
    - 14B loaded on demand (primary H-Module)
    - 72B loaded for complex tasks (when available)
    """

    def __init__(
        self,
        model_zoo_path: Path,
        max_memory_gb: float = 100.0,
        default_size: ModelSize = ModelSize.MEDIUM
    ):
        """
        Initialize multi-model loader.

        Args:
            model_zoo_path: Path to model zoo directory
            max_memory_gb: Maximum memory to use for models (default 100GB on Thor)
            default_size: Default model size to use
        """
        self.model_zoo_path = Path(model_zoo_path)
        self.max_memory_gb = max_memory_gb
        self.default_size = default_size

        # Model registry
        self.models: Dict[ModelSize, ModelConfig] = {}

        # Define model paths
        self._init_model_paths()

    def _init_model_paths(self):
        """Initialize model path configurations"""

        # 0.5B model (epistemic-pragmatism variant - same as Sprout uses)
        self.models[ModelSize.SMALL] = ModelConfig(
            size=ModelSize.SMALL,
            path=self.model_zoo_path / "epistemic-stances" / "qwen2.5-0.5b" / "epistemic-pragmatism"
        )

        # 14B model
        self.models[ModelSize.MEDIUM] = ModelConfig(
            size=ModelSize.MEDIUM,
            path=self.model_zoo_path / "epistemic-stances" / "qwen2.5-14b" / "base-instruct"
        )

        # 72B model (future)
        self.models[ModelSize.LARGE] = ModelConfig(
            size=ModelSize.LARGE,
            path=self.model_zoo_path / "epistemic-stances" / "qwen2.5-72b" / "base-instruct"
        )

    def load_model(
        self,
        size: ModelSize,
        force_reload: bool = False
    ) -> ModelConfig:
        """
        Load a model into memory.

        Args:
            size: Model size to load
            force_reload: Force reload even if already loaded

        Returns:
            ModelConfig with loaded model and tokenizer
        """

        config = self.models[size]

        # Already loaded?
        if config.loaded and not force_reload:
            logger.info(f"Model {size.value} already loaded")
            return config

        # Check if model exists
        if not config.path.exists():
            raise FileNotFoundError(
                f"Model {size.value} not found at {config.path}. "
                f"Run download script first."
            )

        logger.info(f"Loading model {size.value} from {config.path}")

        # Load tokenizer
        config.tokenizer = AutoTokenizer.from_pretrained(
            str(config.path),
            trust_remote_code=True
        )

        # Load model
        config.model = AutoModelForCausalLM.from_pretrained(
            str(config.path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        config.loaded = True

        # Estimate memory usage (rough approximation)
        if size == ModelSize.SMALL:
            config.memory_usage_gb = 1.0
        elif size == ModelSize.MEDIUM:
            config.memory_usage_gb = 28.0
        elif size == ModelSize.LARGE:
            config.memory_usage_gb = 72.0

        logger.info(
            f"✅ Loaded {size.value} model "
            f"(~{config.memory_usage_gb:.1f}GB memory)"
        )

        return config

    def unload_model(self, size: ModelSize):
        """
        Unload a model from memory.

        Args:
            size: Model size to unload
        """

        config = self.models[size]

        if not config.loaded:
            logger.info(f"Model {size.value} not loaded")
            return

        logger.info(f"Unloading model {size.value}")

        # Free memory
        del config.model
        del config.tokenizer
        config.model = None
        config.tokenizer = None
        config.loaded = False
        config.memory_usage_gb = 0.0

        # Force garbage collection
        torch.cuda.empty_cache()

        logger.info(f"✅ Unloaded {size.value} model")

    def get_model_for_task(
        self,
        complexity: TaskComplexity,
        auto_load: bool = True
    ) -> ModelConfig:
        """
        Get appropriate model for task complexity.

        Args:
            complexity: Task complexity level
            auto_load: Automatically load model if not loaded

        Returns:
            ModelConfig for appropriate model size

        Routing logic:
        - SIMPLE → 0.5B (fast, efficient)
        - MODERATE → 14B (balanced, default)
        - COMPLEX → 14B (can handle most)
        - VERY_COMPLEX → 72B (if available, else 14B)

        With fallback: If requested model unavailable, use next larger size
        """

        # Determine model size
        if complexity == TaskComplexity.SIMPLE:
            size = ModelSize.SMALL
        elif complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]:
            size = ModelSize.MEDIUM
        else:  # VERY_COMPLEX
            # Use 72B if available, else 14B
            if self.models[ModelSize.LARGE].path.exists():
                size = ModelSize.LARGE
            else:
                size = ModelSize.MEDIUM
                logger.info("72B not available, using 14B for very complex task")

        # Fallback logic: if requested model doesn't exist, use next larger
        config = self.models[size]
        if not config.path.exists():
            logger.warning(f"Model {size.value} not found at {config.path}")

            # Fallback chain: SMALL → MEDIUM → LARGE
            fallback_sizes = []
            if size == ModelSize.SMALL:
                fallback_sizes = [ModelSize.MEDIUM, ModelSize.LARGE]
            elif size == ModelSize.MEDIUM:
                fallback_sizes = [ModelSize.LARGE]

            for fallback_size in fallback_sizes:
                fallback_config = self.models[fallback_size]
                if fallback_config.path.exists():
                    logger.info(f"Using {fallback_size.value} model as fallback")
                    config = fallback_config
                    size = fallback_size
                    break
            else:
                # No fallback available
                raise FileNotFoundError(
                    f"No models available. Requested {size.value}, "
                    f"no fallbacks found. Check model-zoo directory."
                )

        # Load if needed
        if auto_load and not config.loaded:
            self.load_model(size)

        return config

    def get_total_memory_usage(self) -> float:
        """Get total memory used by all loaded models."""
        return sum(
            config.memory_usage_gb
            for config in self.models.values()
            if config.loaded
        )

    def ensure_memory_available(
        self,
        required_gb: float
    ) -> bool:
        """
        Ensure enough memory is available, unloading models if needed.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if memory can be made available
        """

        current_usage = self.get_total_memory_usage()
        available = self.max_memory_gb - current_usage

        if available >= required_gb:
            return True

        # Need to free memory - unload smallest impact models first
        logger.info(
            f"Need {required_gb:.1f}GB, only {available:.1f}GB available. "
            f"Unloading models..."
        )

        # Unload in order: LARGE → MEDIUM → keep SMALL
        for size in [ModelSize.LARGE, ModelSize.MEDIUM]:
            if self.models[size].loaded:
                self.unload_model(size)
                current_usage = self.get_total_memory_usage()
                available = self.max_memory_gb - current_usage

                if available >= required_gb:
                    return True

        # If we still don't have enough, we have a problem
        return available >= required_gb

    def generate(
        self,
        prompt: str,
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using appropriate model for task complexity.

        Args:
            prompt: Input prompt
            complexity: Task complexity level
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """

        # Get appropriate model
        config = self.get_model_for_task(complexity, auto_load=True)

        if not config.model or not config.tokenizer:
            raise RuntimeError(f"Model {config.size.value} not loaded")

        # Tokenize
        inputs = config.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(config.model.device)

        # Generate
        with torch.no_grad():
            outputs = config.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )

        # Decode
        response = config.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response

    def get_status(self) -> Dict:
        """Get current loader status"""
        memory_used = self.get_total_memory_usage()
        return {
            "models": {
                size.value: {
                    "loaded": config.loaded,
                    "memory_gb": config.memory_usage_gb,
                    "path": str(config.path),
                    "exists": config.path.exists()
                }
                for size, config in self.models.items()
            },
            "total_memory_gb": memory_used,
            "memory_used_gb": memory_used,  # Alias for compatibility
            "max_memory_gb": self.max_memory_gb,
            "default_size": self.default_size.value
        }


# Convenience function for Thor SAGE
def create_thor_loader(
    model_zoo_path: Path = Path("model-zoo/sage"),
    preload_default: bool = True
) -> MultiModelLoader:
    """
    Create multi-model loader configured for Thor.

    Args:
        model_zoo_path: Path to model zoo
        preload_default: Preload 14B model (recommended)

    Returns:
        Configured MultiModelLoader
    """

    loader = MultiModelLoader(
        model_zoo_path=model_zoo_path,
        max_memory_gb=100.0,  # Leave 22GB headroom
        default_size=ModelSize.MEDIUM  # 14B default for Thor
    )

    if preload_default:
        logger.info("Preloading default 14B model for Thor...")
        loader.load_model(ModelSize.MEDIUM)

    return loader
