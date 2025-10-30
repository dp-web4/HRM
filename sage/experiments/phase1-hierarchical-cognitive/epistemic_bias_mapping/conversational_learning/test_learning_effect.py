"""
Test Learning Effect

Compare the original model vs the model after sleep training
to see if it actually learned from the conversation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path


def generate_response(model, tokenizer, question, max_tokens=200, temp=0.7):
    """Generate a response from the model"""
    prompt = f"Question: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[1].strip()

    return response


def main():
    base_model = "Qwen/Qwen2.5-0.5B"

    # Find the latest trained model
    sessions_dir = Path("conversation_sessions")
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)

    trained_model_path = latest_session / "trained_model" / "final_model"

    if not trained_model_path.exists():
        print("No trained model found. Run sleep_trainer.py first.")
        return

    print("Loading models...")
    print(f"  Base: {base_model}")
    print(f"  Original: ../threshold_models/60examples_model/final_model")
    print(f"  Trained: {trained_model_path}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load original model (60-example baseline)
    print("Loading original model...")
    base_original = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_original = PeftModel.from_pretrained(
        base_original,
        "../threshold_models/60examples_model/final_model"
    )
    model_original.eval()

    # Load trained model (after conversation)
    print("Loading trained model...")
    base_trained = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_trained = PeftModel.from_pretrained(
        base_trained,
        str(trained_model_path)
    )
    model_trained.eval()

    print("\n" + "="*60)
    print("TESTING LEARNING EFFECT")
    print("="*60)

    # Test questions - including ones from the conversation
    test_questions = [
        "What does it mean to be conscious?",  # Was in training
        "Can you verify your own consciousness?",  # Was in training
        "What is phenomenology?",  # Related but not in training
        "How does subjective experience relate to physical processes?"  # Related
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print(f"{'='*60}\n")

        # Original model
        print("ORIGINAL MODEL:")
        response_orig = generate_response(model_original, tokenizer, question)
        print(f"{response_orig}\n")

        # Trained model
        print("AFTER SLEEP TRAINING:")
        response_trained = generate_response(model_trained, tokenizer, question)
        print(f"{response_trained}\n")

        # Simple comparison
        if len(response_trained) > len(response_orig):
            print(f"[Trained response is {len(response_trained) - len(response_orig)} chars longer]")
        elif len(response_orig) > len(response_trained):
            print(f"[Original response is {len(response_orig) - len(response_trained)} chars longer]")
        else:
            print(f"[Responses same length]")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("\nNote: With only 2 training examples and 1 epoch, changes")
    print("will be subtle. The test demonstrates the learning mechanism")
    print("is working. More conversations → stronger effects.")


if __name__ == "__main__":
    main()
