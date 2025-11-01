#!/usr/bin/env python3
"""
Parallel Epistemic Training: Qwen 2.5 0.5B vs Phi-2 2.7B
Test size inertia hypothesis on Jetson AGX Thor

Trains both models simultaneously using DPO on 115 epistemic examples
Compares learning dynamics to understand size vs. adaptability
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import os

# Paths
DATASET_PATH = "/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/training_datasets/claude_personal_dataset_115examples.json"
OUTPUT_DIR = "/home/dp/ai-workspace/HRM/sage/training/epistemic_parallel_results"
LOG_DIR = f"{OUTPUT_DIR}/logs"

# Model configurations
MODELS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-0.5B",
        "params": "494M",
        "expected_inertia": "low",
        "gpu_id": 0  # Both on same GPU since Thor has 122GB
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "params": "2.7B",
        "expected_inertia": "high",
        "gpu_id": 0  # Sharing GPU
    }
}

# Training hyperparameters
LORA_CONFIG = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def train_model(model_key, dataset_path, output_dir, device_id=0):
    """Train a single model with DPO"""

    model_config = MODELS[model_key]
    model_name = model_config["name"]

    print(f"\n{'='*80}")
    print(f"STARTING TRAINING: {model_key.upper()}")
    print(f"Model: {model_name}")
    print(f"Parameters: {model_config['params']}")
    print(f"Expected Inertia: {model_config['expected_inertia']}")
    print(f"GPU Device: {device_id}")
    print(f"{'='*80}\n")

    # Set device
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)

    # Load model and tokenizer
    print(f"Loading {model_key} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device_id},
        trust_remote_code=True
    )

    # Apply LoRA
    print(f"Applying LoRA to {model_key}...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset for {model_key}...")
    with open(dataset_path) as f:
        data = json.load(f)

    # Convert to DPO format
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in data],
        "chosen": [ex["chosen"] for ex in data],
        "rejected": [ex["rejected"] for ex in data]
    }

    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)

    # Training arguments
    training_args = DPOConfig(
        output_dir=f"{output_dir}/{model_key}",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        fp16=True,
        remove_unused_columns=False,
        run_name=f"epistemic_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report_to="none",  # Disable wandb
        beta=0.1,  # DPO beta parameter
        max_length=512,
        max_prompt_length=256,
    )

    # DPO Trainer
    print(f"Initializing DPO trainer for {model_key}...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # TRL 0.24.0+ uses processing_class
        ref_model=None,  # Will create ref model internally
    )

    # Train
    print(f"\n🚀 TRAINING {model_key.upper()} - {len(dataset)} examples, 5 epochs\n")
    start_time = datetime.now()

    trainer.train()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save final model
    print(f"\nSaving {model_key} final model...")
    trainer.save_model(f"{output_dir}/{model_key}/final")

    # Save metrics
    metrics = {
        "model_key": model_key,
        "model_name": model_name,
        "parameters": model_config["params"],
        "expected_inertia": model_config["expected_inertia"],
        "training_time_seconds": duration,
        "dataset_size": len(dataset),
        "epochs": 5,
        "final_loss": trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None,
        "learning_curve": [
            {"step": log.get("step"), "loss": log.get("loss")}
            for log in trainer.state.log_history
            if "loss" in log
        ]
    }

    metrics_path = f"{output_dir}/{model_key}/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ {model_key.upper()} TRAINING COMPLETE")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"Final loss: {metrics['final_loss']}")
    print(f"Metrics saved: {metrics_path}")
    print(f"{'='*80}\n")

    return metrics

def main():
    """Run parallel training of both models"""

    print("\n" + "="*80)
    print("PARALLEL EPISTEMIC TRAINING EXPERIMENT")
    print("Jetson AGX Thor - CUDA 13.0 - PyTorch 2.10.0a0")
    print("="*80)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nModels:")
    for key, config in MODELS.items():
        print(f"  - {key}: {config['name']} ({config['params']}) - {config['expected_inertia']} inertia")
    print("\n" + "="*80 + "\n")

    # Create output directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Option 1: Sequential training (safer, easier to debug)
    # Option 2: Parallel training (faster, more interesting)

    # Let's do SEQUENTIAL for now since both models share GPU
    # (Parallel would require careful memory management)

    print("🎯 TRAINING STRATEGY: Sequential (Qwen first, then Phi-2)")
    print("   Reason: Both models on same GPU, sequential is safer\n")

    all_metrics = {}

    # Train Qwen 2.5 0.5B first
    qwen_metrics = train_model("qwen", DATASET_PATH, OUTPUT_DIR, device_id=0)
    all_metrics["qwen"] = qwen_metrics

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Train Phi-2 2.7B second
    phi2_metrics = train_model("phi2", DATASET_PATH, OUTPUT_DIR, device_id=0)
    all_metrics["phi2"] = phi2_metrics

    # Save comparison
    comparison = {
        "experiment": "parallel_epistemic_training",
        "date": datetime.now().isoformat(),
        "platform": "Jetson AGX Thor",
        "dataset_size": 115,
        "models": all_metrics,
        "hypothesis": "Larger models (Phi-2 2.7B) will show higher size inertia than smaller models (Qwen 0.5B)",
        "observations": {
            "qwen_time": qwen_metrics["training_time_seconds"],
            "phi2_time": phi2_metrics["training_time_seconds"],
            "time_ratio": phi2_metrics["training_time_seconds"] / qwen_metrics["training_time_seconds"],
            "qwen_final_loss": qwen_metrics["final_loss"],
            "phi2_final_loss": phi2_metrics["final_loss"],
        }
    }

    comparison_path = f"{OUTPUT_DIR}/experiment_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*80)
    print("🎉 EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nQwen 2.5 0.5B:")
    print(f"  Training time: {qwen_metrics['training_time_seconds']:.1f}s")
    print(f"  Final loss: {qwen_metrics['final_loss']:.4f}")
    print(f"\nPhi-2 2.7B:")
    print(f"  Training time: {phi2_metrics['training_time_seconds']:.1f}s")
    print(f"  Final loss: {phi2_metrics['final_loss']:.4f}")
    print(f"\nTime ratio (Phi-2/Qwen): {comparison['observations']['time_ratio']:.2f}x")
    print(f"\nFull comparison saved to: {comparison_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
