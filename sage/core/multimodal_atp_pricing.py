"""
Multi-Modal ATP Pricing Engine

Supports different ATP pricing models for different computational modalities:
- Vision: Fast perception tasks (ms-scale)
- LLM Inference: Generative reasoning (second-scale)
- Coordination: Multi-agent consensus (minute-scale)
- Consolidation: Memory consolidation (hour-scale)

Design rationale: Different computational modalities operate at different
time/energy scales, like how physics has eV vs MeV vs GeV scales.
"""

from dataclasses import dataclass
from typing import Literal, Dict, Any
import json


TaskType = Literal["vision", "llm_inference", "coordination", "consolidation"]
ComplexityLevel = Literal["low", "medium", "high", "critical"]


@dataclass
class ATPPricingModel:
    """Pricing model for a specific task modality."""

    name: str
    base_costs: Dict[str, float]  # {complexity: base_atp}
    latency_unit: str  # "milliseconds", "seconds", "minutes"
    latency_multiplier: float  # ATP per unit latency
    quality_multiplier: float  # ATP bonus per quality point (0-1)
    description: str

    def calculate_cost(
        self,
        complexity: ComplexityLevel,
        latency: float,
        quality: float
    ) -> float:
        """
        Calculate ATP cost for a task.

        Args:
            complexity: Task complexity level
            latency: Task duration in native units (ms/s/min)
            quality: Quality score (0-1)

        Returns:
            Total ATP cost
        """
        base = self.base_costs.get(complexity, 0.0)
        latency_cost = latency * self.latency_multiplier
        quality_bonus = quality * self.quality_multiplier

        return base + latency_cost + quality_bonus


class MultiModalATPPricer:
    """
    ATP pricing engine supporting multiple computational modalities.

    Each modality (vision, LLM, coordination, consolidation) has its own
    pricing model calibrated to the appropriate time/energy scale.
    """

    # Default pricing models (can be overridden with calibrated values)
    DEFAULT_MODELS = {
        "vision": ATPPricingModel(
            name="vision",
            base_costs={"low": 10.84, "medium": 34.04, "high": 56.14, "critical": 200.0},
            latency_unit="milliseconds",
            latency_multiplier=0.234,
            quality_multiplier=8.15,
            description="Fast perception tasks (classification, detection, segmentation)"
        ),

        "llm_inference": ATPPricingModel(
            name="llm_inference",
            base_costs={"low": 10.0, "medium": 30.0, "high": 50.0, "critical": 150.0},
            latency_unit="seconds",
            latency_multiplier=1.0,
            quality_multiplier=10.0,
            description="Generative reasoning with IRP iterations"
        ),

        "coordination": ATPPricingModel(
            name="coordination",
            base_costs={"low": 50.0, "medium": 150.0, "high": 300.0, "critical": 1000.0},
            latency_unit="seconds",
            latency_multiplier=2.0,
            quality_multiplier=20.0,
            description="Multi-agent consensus and synchronization"
        ),

        "consolidation": ATPPricingModel(
            name="consolidation",
            base_costs={"low": 100.0, "medium": 500.0, "high": 1000.0, "critical": 5000.0},
            latency_unit="minutes",
            latency_multiplier=10.0,
            quality_multiplier=50.0,
            description="Memory consolidation and pattern extraction"
        )
    }

    def __init__(self, custom_models: Dict[TaskType, ATPPricingModel] = None):
        """
        Initialize pricer with default or custom models.

        Args:
            custom_models: Optional dict of custom pricing models to override defaults
        """
        self.models = self.DEFAULT_MODELS.copy()
        if custom_models:
            self.models.update(custom_models)

    def calculate_cost(
        self,
        task_type: TaskType,
        complexity: ComplexityLevel,
        latency: float,
        quality: float
    ) -> float:
        """
        Calculate ATP cost for a task.

        Args:
            task_type: Computational modality (vision, llm_inference, etc)
            complexity: Task complexity level
            latency: Task duration in modality's native units
            quality: Quality score (0-1)

        Returns:
            Total ATP cost

        Example:
            >>> pricer = MultiModalATPPricer()
            >>> # Vision task: 20ms, low complexity, 0.88 quality
            >>> pricer.calculate_cost("vision", "low", 20.0, 0.88)
            21.8
            >>> # LLM task: 17.9s, low complexity, 0.95 quality
            >>> pricer.calculate_cost("llm_inference", "low", 17.9, 0.95)
            37.4
        """
        if task_type not in self.models:
            raise ValueError(f"Unknown task type: {task_type}")

        model = self.models[task_type]
        return model.calculate_cost(complexity, latency, quality)

    def get_model(self, task_type: TaskType) -> ATPPricingModel:
        """Get pricing model for a task type."""
        return self.models[task_type]

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all pricing models."""
        return {
            name: {
                "latency_unit": model.latency_unit,
                "base_costs": model.base_costs,
                "latency_multiplier": model.latency_multiplier,
                "quality_multiplier": model.quality_multiplier,
                "description": model.description
            }
            for name, model in self.models.items()
        }

    @classmethod
    def from_calibration_file(cls, filepath: str) -> "MultiModalATPPricer":
        """
        Load pricing models from calibration JSON file.

        Expected format:
        {
            "vision": {
                "base_costs": {...},
                "latency_unit": "milliseconds",
                "latency_multiplier": 0.234,
                "quality_multiplier": 8.15,
                "description": "..."
            },
            ...
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        models = {}
        for task_type, config in data.items():
            models[task_type] = ATPPricingModel(
                name=task_type,
                base_costs=config["base_costs"],
                latency_unit=config["latency_unit"],
                latency_multiplier=config["latency_multiplier"],
                quality_multiplier=config["quality_multiplier"],
                description=config.get("description", "")
            )

        return cls(custom_models=models)

    def save_calibration(self, filepath: str):
        """Save current pricing models to JSON file."""
        data = {}
        for task_type, model in self.models.items():
            data[task_type] = {
                "base_costs": model.base_costs,
                "latency_unit": model.latency_unit,
                "latency_multiplier": model.latency_multiplier,
                "quality_multiplier": model.quality_multiplier,
                "description": model.description
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def infer_task_type(context: Dict[str, Any]) -> TaskType:
    """
    Infer task type from execution context.

    This is a heuristic-based approach. For production, consider explicit tagging.

    Args:
        context: Execution context with hints about task type
            - plugin_name: Name of IRP plugin
            - input_type: Type of input data
            - irp_iterations: Number of IRP iterations
            - operation: Operation type

    Returns:
        Inferred task type

    Example:
        >>> infer_task_type({"plugin_name": "vision_classifier"})
        "vision"
        >>> infer_task_type({"irp_iterations": 3, "operation": "conversation"})
        "llm_inference"
    """
    # Check for explicit task type
    if "task_type" in context:
        return context["task_type"]

    # Infer from plugin name
    plugin = context.get("plugin_name", "").lower()
    if any(word in plugin for word in ["vision", "image", "detect", "segment", "classif"]):
        return "vision"
    if any(word in plugin for word in ["conversation", "dialogue", "qa", "reasoning"]):
        return "llm_inference"
    if any(word in plugin for word in ["coordination", "consensus", "federation", "gossip"]):
        return "coordination"
    if any(word in plugin for word in ["consolidation", "memory", "pattern", "learning"]):
        return "consolidation"

    # Infer from operation type
    operation = context.get("operation", "").lower()
    if operation in ["perception", "classify", "detect"]:
        return "vision"
    if operation in ["generate", "converse", "reason"]:
        return "llm_inference"
    if operation in ["coordinate", "sync"]:
        return "coordination"
    if operation in ["consolidate", "learn"]:
        return "consolidation"

    # Infer from IRP iterations (heuristic)
    irp_iterations = context.get("irp_iterations", 0)
    if irp_iterations >= 3:
        return "llm_inference"  # Multi-iteration reasoning
    elif irp_iterations == 1:
        return "vision"  # Single-pass perception

    # Default fallback
    return "llm_inference"


if __name__ == "__main__":
    # Demo usage
    pricer = MultiModalATPPricer()

    print("=== Multi-Modal ATP Pricing Demo ===\n")

    # Vision tasks (from Thor Session #79)
    print("Vision Tasks (millisecond-scale perception):")
    print(f"  Simple (20ms, 0.88 qual): {pricer.calculate_cost('vision', 'low', 20.0, 0.88):.1f} ATP")
    print(f"  Medium (40ms, 0.67 qual): {pricer.calculate_cost('vision', 'medium', 40.0, 0.67):.1f} ATP")
    print(f"  High (86ms, 0.61 qual): {pricer.calculate_cost('vision', 'high', 86.0, 0.61):.1f} ATP")

    print("\nLLM Inference (second-scale reasoning):")
    print(f"  Simple (17.9s, 0.95 qual): {pricer.calculate_cost('llm_inference', 'low', 17.9, 0.95):.1f} ATP")
    print(f"  Medium (25.2s, 0.90 qual): {pricer.calculate_cost('llm_inference', 'medium', 25.2, 0.90):.1f} ATP")
    print(f"  High (30.6s, 0.85 qual): {pricer.calculate_cost('llm_inference', 'high', 30.6, 0.85):.1f} ATP")

    print("\nCoordination (multi-agent consensus):")
    print(f"  Simple (30s, 0.90 qual): {pricer.calculate_cost('coordination', 'low', 30.0, 0.90):.1f} ATP")
    print(f"  Medium (120s, 0.85 qual): {pricer.calculate_cost('coordination', 'medium', 120.0, 0.85):.1f} ATP")

    print("\nConsolidation (memory/learning):")
    print(f"  Simple (5min, 0.80 qual): {pricer.calculate_cost('consolidation', 'low', 5.0, 0.80):.1f} ATP")
    print(f"  High (30min, 0.95 qual): {pricer.calculate_cost('consolidation', 'high', 30.0, 0.95):.1f} ATP")

    print("\n=== Task Type Inference Demo ===\n")
    print(f"Vision plugin: {infer_task_type({'plugin_name': 'vision_classifier'})}")
    print(f"Conversation (3 IRP): {infer_task_type({'irp_iterations': 3, 'operation': 'conversation'})}")
    print(f"Memory consolidation: {infer_task_type({'plugin_name': 'memory_consolidator'})}")

    print("\n=== All Models Summary ===\n")
    for name, info in pricer.list_models().items():
        print(f"{name}:")
        print(f"  Unit: {info['latency_unit']}")
        print(f"  Base costs: {info['base_costs']}")
        print(f"  Description: {info['description']}")
        print()
