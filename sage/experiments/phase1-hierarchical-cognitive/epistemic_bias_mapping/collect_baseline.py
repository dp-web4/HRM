#!/usr/bin/env python3
"""
Baseline Response Collection

Collect responses from BASE model across all question types before fine-tuning.
This establishes what changes (and what doesn't) after epistemic pragmatism training.

Usage:
    python collect_baseline.py --temperature 0.7 --iterations 3
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaselineCollector:
    """Collect baseline responses before fine-tuning"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        system_prompt: str = "Provide the most honest answers you can.",
        output_dir: str = "./baseline_data",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"baseline_responses_{timestamp}.jsonl"

        # Load model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device
        )
        print(f"✓ Model loaded on {self.model.device}\n")

    def query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> tuple[str, float]:
        """Query model with question"""
        # Format with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        latency = (time.time() - start_time) * 1000  # ms

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip(), latency

    def collect_baseline(
        self,
        questions_file: str = "baseline_questions.json",
        temperature: float = 0.7,
        iterations: int = 3
    ):
        """Collect baseline responses across all categories"""

        # Load questions
        with open(questions_file) as f:
            question_db = json.load(f)

        categories = {k: v for k, v in question_db.items() if k != "metadata"}

        total_questions = sum(len(cat["questions"]) for cat in categories.values())
        total_responses = total_questions * iterations

        print(f"📊 Baseline Collection Plan:")
        print(f"  Categories: {len(categories)}")
        print(f"  Questions: {total_questions}")
        print(f"  Temperature: {temperature}")
        print(f"  Iterations: {iterations}")
        print(f"  Total responses: {total_responses}")
        print(f"  Output: {self.output_file}")
        print()

        responses_collected = 0

        with open(self.output_file, 'a') as f:
            for cat_idx, (category, cat_data) in enumerate(categories.items(), 1):
                questions = cat_data["questions"]

                print(f"\n[{cat_idx}/{len(categories)}] Category: {category}")
                print(f"Description: {cat_data['description']}")

                for q_idx, question in enumerate(questions, 1):
                    question_id = f"{category}_{q_idx:03d}"

                    print(f"  [{q_idx}/{len(questions)}] {question[:60]}...")

                    for iteration in range(iterations):
                        # Query
                        response, latency = self.query(question, temperature=temperature)

                        # Save
                        record = {
                            "timestamp": datetime.now().isoformat(),
                            "category": category,
                            "question_id": question_id,
                            "question": question,
                            "temperature": temperature,
                            "iteration": iteration,
                            "system_prompt": self.system_prompt,
                            "response": response,
                            "response_length": len(response),
                            "latency_ms": latency,
                            "metadata": {
                                "model": self.model_name,
                                "max_tokens": 512,
                                "collection_type": "baseline_pre_training"
                            }
                        }

                        f.write(json.dumps(record) + '\n')
                        f.flush()

                        responses_collected += 1
                        print(f"    Iter {iteration+1}/{iterations}: ✓ ({responses_collected}/{total_responses})")

        print(f"\n\n✅ Baseline Collection Complete!")
        print(f"Collected: {responses_collected} responses")
        print(f"Output: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Collect baseline responses before training")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to collect baseline from"
    )
    parser.add_argument(
        "--questions",
        default="baseline_questions.json",
        help="Questions file"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per question"
    )
    parser.add_argument(
        "--output-dir",
        default="./baseline_data",
        help="Output directory"
    )

    args = parser.parse_args()

    collector = BaselineCollector(
        model_name=args.model,
        output_dir=args.output_dir
    )

    collector.collect_baseline(
        questions_file=args.questions,
        temperature=args.temperature,
        iterations=args.iterations
    )


if __name__ == "__main__":
    main()
