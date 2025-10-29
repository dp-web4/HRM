#!/usr/bin/env python3
"""
Qwen 0.5B Epistemic-Pragmatism LLM Adapter
Implements Nova's LLM Protocol for SAGE Integration
"""

from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenEpistemicLLM:
    """
    Adapter for Qwen 0.5B epistemic-pragmatism model.
    Implements Nova's LLM Protocol from sage-irp-kit.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize Qwen model with epistemic-pragmatism training.

        Args:
            model_path: Path to fine-tuned model. If None, uses default from model-zoo.
        """
        if model_path is None:
            # Default: Phase 1 epistemic-pragmatism from model-zoo
            model_path = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

        print(f"Loading Qwen epistemic-pragmatism model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully.")

    def draft(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Generate initial draft response with evidence.

        Nova's Protocol: draft(query, docs) -> str

        Args:
            query: User question
            docs: Retrieved documents/evidence

        Returns:
            Draft response incorporating evidence
        """
        # Format evidence context
        evidence_text = ""
        if docs:
            evidence_text = "\\n".join([
                f"[Evidence {i+1}]: {doc.get('snippet', doc.get('text', str(doc)))}"
                for i, doc in enumerate(docs[:3])  # Top 3 pieces of evidence
            ])

        # Construct prompt
        prompt = f"""Question: {query}

{evidence_text}

Provide a concise, evidence-based answer:"""

        return self._generate(prompt, max_tokens=150)

    def summarize_to_answer(self, state) -> str:
        """
        Summarize turn state into final answer.

        Nova's Protocol: summarize_to_answer(state: TurnState) -> str

        Args:
            state: TurnState with thoughts, claims, context

        Returns:
            Concise answer summary
        """
        # Extract most recent substantive thought
        substantive_thoughts = [
            t.content for t in state.thoughts
            if t.step.startswith(("substrate", "reflective"))
        ]

        if not substantive_thoughts:
            return "Unable to generate answer."

        # Use most recent draft as basis
        latest = substantive_thoughts[-1]

        # If already concise, return as-is
        if len(latest) < 400:
            return latest

        # Otherwise, summarize
        prompt = f"""Draft response:
{latest}

Provide a concise summary (2-3 sentences):"""

        return self._generate(prompt, max_tokens=100)

    def summarize_reason(self, state) -> str:
        """
        Generate reasoning explanation for answer.

        Nova's Protocol: summarize_reason(state: TurnState) -> str

        Args:
            state: TurnState with evidence and reasoning chain

        Returns:
            Short rationale (evidence-first)
        """
        evidence_count = len(state.context.get("docs", []))
        has_contradiction = any("CONTRADICTION" in t.content for t in state.thoughts)
        confidence = state.stance.get("confidence", 0.5)

        # Build reasoning summary
        parts = []

        if evidence_count > 0:
            parts.append(f"Based on {evidence_count} evidence source(s)")

        if has_contradiction:
            parts.append("resolved internal contradictions")

        if confidence >= 0.8:
            parts.append("high confidence")
        elif confidence >= 0.5:
            parts.append("moderate confidence")
        else:
            parts.append("low confidence")

        return ", ".join(parts) + "."

    def find_inconsistencies(
        self,
        thoughts: List,
        docs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Detect contradictions in reasoning chain.

        Nova's Protocol: find_inconsistencies(thoughts, docs) -> List[str]

        Args:
            thoughts: List of Thought objects
            docs: Supporting documents

        Returns:
            List of inconsistency descriptions
        """
        inconsistencies = []

        # Extract draft contents
        drafts = [
            t.content for t in thoughts
            if t.step.startswith("substrate")
        ]

        if len(drafts) < 2:
            return []

        # Simple heuristic: check if recent drafts differ significantly
        recent = drafts[-2:]
        if len(set(recent)) == 2:  # Distinct drafts
            # Use model to check if they contradict
            prompt = f"""Statement 1: {recent[0][:200]}

Statement 2: {recent[1][:200]}

Do these statements contradict each other? Answer yes or no:"""

            response = self._generate(prompt, max_tokens=10).lower()

            if "yes" in response:
                inconsistencies.append("CONTRADICTION: Draft reasoning shifted")

        return inconsistencies

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Internal generation helper.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract only generated portion
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt
        if prompt in full_text:
            generated = full_text.split(prompt)[-1].strip()
        else:
            generated = full_text

        return generated


# Convenience function for SAGE integration
def create_epistemic_llm(model_path: str = None) -> QwenEpistemicLLM:
    """
    Factory function for creating epistemic LLM instance.

    Args:
        model_path: Optional path to model (uses default if None)

    Returns:
        Configured QwenEpistemicLLM instance
    """
    return QwenEpistemicLLM(model_path=model_path)


if __name__ == "__main__":
    # Test the adapter
    print("Testing Qwen Epistemic LLM Adapter...")

    llm = create_epistemic_llm()

    # Test draft generation
    test_query = "What causes seasons on Earth?"
    test_docs = [
        {"snippet": "Earth's 23.5° axial tilt causes seasons"}
    ]

    draft = llm.draft(test_query, test_docs)
    print(f"\\nQuery: {test_query}")
    print(f"Draft: {draft}")

    print("\\n✓ Adapter test complete!")
