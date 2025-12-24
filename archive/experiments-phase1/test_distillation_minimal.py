"""
Minimal Knowledge Distillation Test
Tests if student model can learn from teacher with just a few examples
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import time

def create_test_dataset(teacher_model, teacher_tokenizer, examples):
    """Generate teacher responses for examples"""

    print(f"📝 Creating training dataset...")
    print(f"   Teacher generating responses for {len(examples)} examples\n")

    training_pairs = []

    for i, prompt in enumerate(examples):
        print(f"   {i+1}/{len(examples)}: {prompt[:50]}...")

        inputs = teacher_tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )

        response = teacher_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        full_text = f"Question: {prompt}\nAnswer: {response}"
        training_pairs.append(full_text)

    print(f"\n✓ Dataset created\n")
    return training_pairs


def evaluate_student(model, tokenizer, test_prompts):
    """Evaluate student model on test prompts"""

    print("📊 Evaluating student model...")
    model.eval()

    responses = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=0.7
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        responses.append((prompt, response))
        print(f"   Q: {prompt[:40]}...")
        print(f"   A: {response[:60]}...\n")

    return responses


def test_minimal_distillation():
    """Test distillation with minimal setup"""

    print("🧪 Minimal Knowledge Distillation Test\n")
    print("="*80)
    print("SETUP")
    print("="*80 + "\n")

    # Use tiny models for fast testing
    teacher_name = "Qwen/Qwen2-0.5B"  # "Teacher" (will use for student too, just to test mechanics)
    student_name = "Qwen/Qwen2-0.5B"

    print(f"Teacher: {teacher_name}")
    print(f"Student: {student_name}")
    print(f"(Using same model to test distillation mechanics)\n")

    # Training examples
    training_examples = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is supervised learning?",
        "How do transformers work?",
        "What is reinforcement learning?"
    ]

    test_examples = [
        "What is deep learning?",
        "Explain AI in one sentence."
    ]

    print(f"Training examples: {len(training_examples)}")
    print(f"Test examples: {len(test_examples)}\n")

    # Load teacher
    print("=" * 80)
    print("LOADING TEACHER")
    print("=" * 80 + "\n")

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    teacher_model.eval()
    print(f"✓ Teacher loaded\n")

    # Generate training data
    print("=" * 80)
    print("GENERATING TRAINING DATA")
    print("=" * 80 + "\n")

    training_texts = create_test_dataset(
        teacher_model,
        teacher_tokenizer,
        training_examples
    )

    # Free teacher model to save GPU memory
    print("Freeing teacher model...")
    del teacher_model
    torch.cuda.empty_cache()
    print(f"✓ Teacher freed\n")

    # Load student
    print("=" * 80)
    print("LOADING STUDENT")
    print("=" * 80 + "\n")

    student_tokenizer = AutoTokenizer.from_pretrained(student_name)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.float32,  # Use FP32 for training
        device_map="cuda"
    )
    print(f"✓ Student loaded\n")

    # Evaluate BEFORE training
    print("=" * 80)
    print("EVALUATION BEFORE DISTILLATION")
    print("=" * 80 + "\n")

    responses_before = evaluate_student(student_model, student_tokenizer, test_examples)

    # Prepare dataset
    print("=" * 80)
    print("PREPARING DATASET")
    print("=" * 80 + "\n")

    tokenized = student_tokenizer(
        training_texts,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone()
    })

    print(f"✓ Dataset prepared: {len(dataset)} examples\n")

    # Train
    print("=" * 80)
    print("TRAINING (DISTILLATION)")
    print("=" * 80 + "\n")

    training_args = TrainingArguments(
        output_dir="./distill_test_output",
        num_train_epochs=2,  # Very short
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        fp16=False,  # Use FP32
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...\n")
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    print(f"\n✓ Training complete")
    print(f"   Time: {training_time:.1f}s")
    print(f"   Final loss: {train_result.training_loss:.4f}\n")

    # Evaluate AFTER training
    print("=" * 80)
    print("EVALUATION AFTER DISTILLATION")
    print("=" * 80 + "\n")

    responses_after = evaluate_student(student_model, student_tokenizer, test_examples)

    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80 + "\n")

    print("Did responses change?\n")

    for i, (prompt, resp_before) in enumerate(responses_before):
        _, resp_after = responses_after[i]

        print(f"Prompt: {prompt}")
        print(f"Before: {resp_before[:70]}...")
        print(f"After:  {resp_after[:70]}...")

        if resp_before != resp_after:
            print(f"✓ Response changed")
        else:
            print(f"⚠️  Response unchanged")

        print()

    return {
        'training_loss': train_result.training_loss,
        'training_time': training_time,
        'responses_changed': any(
            responses_before[i][1] != responses_after[i][1]
            for i in range(len(responses_before))
        )
    }


if __name__ == "__main__":
    print("🚀 Minimal Knowledge Distillation Test\n")

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("⚠️  No GPU detected\n")

    try:
        results = test_minimal_distillation()

        print("=" * 80)
        print("RESULTS")
        print("=" * 80 + "\n")

        print(f"Training loss: {results['training_loss']:.4f}")
        print(f"Training time: {results['training_time']:.1f}s")
        print(f"Responses changed: {'✓ Yes' if results['responses_changed'] else '✗ No'}")

        print("\n🎯 Conclusion:")
        if results['training_loss'] < 3.0 and results['responses_changed']:
            print("   ✅ Distillation mechanics working!")
            print("   Model learned from training data")
        elif results['training_loss'] < 3.0:
            print("   ⚠️  Loss decreased but responses unchanged")
            print("   May need more epochs or different prompts")
        else:
            print("   ⚠️  Training may not have converged")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
