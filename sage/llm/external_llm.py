"""
External LLM Integration for SAGE
Provides language-based conceptual understanding for reasoning tasks.

Key insight: Without language to think with, models can't understand abstract concepts.
This module integrates a small (2-7B) LLM to provide the "inner monologue" for reasoning.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

# Optional import for quantization
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
import logging

logger = logging.getLogger(__name__)


class ExternalLLMInterface(nn.Module):
    """
    Interface to external language model for conceptual understanding.
    Uses small, efficient models like Gemma-2B or Phi-2 for edge deployment.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",  # 2.7B params, good for edge
        device: str = "cuda",
        quantize: bool = True,  # Use INT4 quantization for efficiency
        max_context_length: int = 512,
        temperature: float = 0.7,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.max_context_length = max_context_length
        self.temperature = temperature
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        if quantize and device == "cuda" and HAS_BITSANDBYTES:
            # INT4 quantization for efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif quantize and device == "cuda" and not HAS_BITSANDBYTES:
            logger.warning("BitsAndBytes not available, loading without quantization")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                self.llm = self.llm.to(device)
        
        # Freeze LLM weights - we're using it as a tool, not training it
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Learned projection to align LLM outputs with SAGE hidden dimensions
        self.output_projection = None  # Will be set when we know SAGE dims
        
        logger.info(f"Loaded {model_name} for conceptual understanding")
    
    def set_sage_dimensions(self, sage_hidden_dim: int):
        """Configure projection layer to match SAGE dimensions."""
        llm_hidden_dim = self.llm.config.hidden_size
        self.output_projection = nn.Linear(llm_hidden_dim, sage_hidden_dim)
        self.output_projection.to(self.device)
    
    def understand_pattern(
        self,
        visual_input: torch.Tensor,
        task_description: Optional[str] = None,
        return_reasoning: bool = True
    ) -> Tuple[torch.Tensor, str]:
        """
        Use LLM to understand what pattern is being shown.
        
        Args:
            visual_input: Tensor representation of visual pattern
            task_description: Optional textual description of the task
            return_reasoning: Whether to return the reasoning process
            
        Returns:
            context_embedding: Tensor for SAGE H-module
            reasoning: String explanation of the pattern
        """
        # Convert visual input to text description
        # In real implementation, this would use a vision encoder
        # For now, we'll create a simple textual representation
        pattern_description = self._visual_to_text(visual_input)
        
        # Construct prompt for pattern understanding
        prompt = self._build_understanding_prompt(pattern_description, task_description)
        
        # Get LLM's understanding
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Generate reasoning
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=150,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
            # Extract hidden states from last layer
            if hasattr(outputs, 'hidden_states'):
                # Get last token's hidden state from final generation step
                last_hidden = outputs.hidden_states[-1][-1][:, -1, :]
            else:
                # Fallback: run forward pass to get hidden states
                model_outputs = self.llm(
                    outputs.sequences,
                    output_hidden_states=True
                )
                last_hidden = model_outputs.hidden_states[-1][:, -1, :]
        
        # Decode reasoning
        reasoning = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Project to SAGE dimensions if configured
        if self.output_projection is not None:
            context_embedding = self.output_projection(last_hidden)
        else:
            context_embedding = last_hidden
        
        if return_reasoning:
            return context_embedding, reasoning
        return context_embedding, ""
    
    def _visual_to_text(self, visual_input: torch.Tensor) -> str:
        """
        Convert visual tensor to text description.
        This is a placeholder - in real implementation would use vision encoder.
        """
        # For ARC tasks, we'd convert the grid to a textual representation
        if len(visual_input.shape) == 3 and visual_input.shape[0] <= 10:
            # Assume it's a color grid (ARC-like)
            grid_str = "Grid pattern:\n"
            for row in visual_input:
                row_str = " ".join([str(int(cell.item())) for cell in row[0]])
                grid_str += f"  {row_str}\n"
            return grid_str
        else:
            return f"Visual input with shape {visual_input.shape}"
    
    def _build_understanding_prompt(
        self,
        pattern: str,
        task: Optional[str] = None
    ) -> str:
        """Build prompt for pattern understanding."""
        base_prompt = (
            "You are analyzing patterns to understand their underlying rules.\n\n"
            f"Pattern:\n{pattern}\n\n"
        )
        
        if task:
            base_prompt += f"Task: {task}\n\n"
        
        base_prompt += (
            "What transformation or pattern do you observe? "
            "Describe it concisely in terms of geometric operations, "
            "color mappings, or logical rules:"
        )
        
        return base_prompt
    
    def generate_hypothesis(
        self,
        context: torch.Tensor,
        previous_attempts: Optional[list] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate a hypothesis for solving the task based on context.
        
        Args:
            context: Current understanding from H-module
            previous_attempts: List of previous failed attempts
            
        Returns:
            hypothesis_embedding: Refined context for next attempt
            hypothesis_text: Textual description of the hypothesis
        """
        # Build prompt including previous attempts
        prompt = "Based on the pattern analysis"
        
        if previous_attempts:
            prompt += " and previous attempts:\n"
            for i, attempt in enumerate(previous_attempts[-3:]):  # Last 3 attempts
                prompt += f"  Attempt {i+1}: {attempt}\n"
            prompt += "\nWhat new approach should we try?"
        else:
            prompt += ", what is the solution approach?"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=self.temperature + 0.1,  # Slightly higher for creativity
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
            # Extract hidden states
            if hasattr(outputs, 'hidden_states'):
                last_hidden = outputs.hidden_states[-1][-1][:, -1, :]
            else:
                model_outputs = self.llm(
                    outputs.sequences,
                    output_hidden_states=True
                )
                last_hidden = model_outputs.hidden_states[-1][:, -1, :]
        
        hypothesis_text = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Combine with input context
        if self.output_projection is not None:
            hypothesis_embedding = self.output_projection(last_hidden) + context
        else:
            hypothesis_embedding = last_hidden
        
        return hypothesis_embedding, hypothesis_text
    
    def verify_solution(
        self,
        predicted: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to verify if a solution makes sense.
        
        Args:
            predicted: Predicted output
            expected: Expected output (if available)
            context: Context from H-module
            
        Returns:
            Dictionary with verification results
        """
        pred_text = self._visual_to_text(predicted)
        
        prompt = f"Does this solution follow the identified pattern?\n\nSolution:\n{pred_text}\n\n"
        
        if expected is not None:
            exp_text = self._visual_to_text(expected)
            prompt += f"Expected:\n{exp_text}\n\n"
            prompt += "Are they equivalent? Explain briefly:"
        else:
            prompt += "Does this look correct? Explain briefly:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,  # Lower temperature for verification
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        verification = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Simple heuristic for success detection
        success_indicators = ["yes", "correct", "equivalent", "matches", "right"]
        is_valid = any(word in verification.lower() for word in success_indicators)
        
        return {
            "is_valid": is_valid,
            "reasoning": verification,
            "confidence": 0.8 if is_valid else 0.2
        }


class LLMGuidedSAGE(nn.Module):
    """
    SAGE enhanced with external LLM for conceptual understanding.
    This addresses the core issue: without language, there's no abstract thought.
    """
    
    def __init__(
        self,
        sage_core: nn.Module,  # The existing SAGE model
        llm_model: str = "microsoft/phi-2",
        use_llm_always: bool = False,  # Whether to always use LLM
        llm_threshold: float = 0.7,  # SNARC score threshold for LLM activation
    ):
        super().__init__()
        
        self.sage = sage_core
        self.llm = ExternalLLMInterface(model_name=llm_model)
        self.use_llm_always = use_llm_always
        self.llm_threshold = llm_threshold
        
        # Configure LLM projection to match SAGE dimensions
        sage_hidden_dim = sage_core.config.hidden_size
        self.llm.set_sage_dimensions(sage_hidden_dim)
        
        # Gating mechanism to decide when to use LLM
        self.llm_gate = nn.Sequential(
            nn.Linear(sage_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info("Initialized LLM-guided SAGE")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task_description: Optional[str] = None,
        **kwargs
    ):
        """
        Forward pass with optional LLM guidance.
        
        The LLM provides conceptual understanding when:
        1. SNARC scores indicate high novelty/surprise
        2. The task requires abstract reasoning
        3. Previous attempts have failed
        """
        # Get SAGE's initial processing
        sage_outputs = self.sage(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Check if we should use LLM
        if self.use_llm_always or self._should_use_llm(sage_outputs):
            # Get LLM's understanding
            context, reasoning = self.llm.understand_pattern(
                visual_input=input_ids,
                task_description=task_description
            )
            
            # Inject LLM context into H-module
            if hasattr(self.sage, 'h_module'):
                # Add LLM context to H-module's understanding
                sage_outputs = self._inject_llm_context(
                    sage_outputs,
                    context,
                    reasoning
                )
                
                logger.debug(f"LLM reasoning: {reasoning[:100]}...")
        
        # Add loss computation if labels provided
        if labels is not None:
            # Compute loss (this would be task-specific)
            loss = self._compute_loss(sage_outputs, labels)
            sage_outputs['loss'] = loss
        
        return sage_outputs
    
    def _should_use_llm(self, sage_outputs: Dict) -> bool:
        """Decide whether to engage the LLM based on SNARC scores."""
        if 'snarc_scores' in sage_outputs:
            scores = sage_outputs['snarc_scores']
            # High novelty or surprise triggers LLM
            if scores.get('novelty', 0) > self.llm_threshold:
                return True
            if scores.get('surprise', 0) > self.llm_threshold:
                return True
        
        # Could also check for repeated failures, high uncertainty, etc.
        return False
    
    def _inject_llm_context(
        self,
        sage_outputs: Dict,
        llm_context: torch.Tensor,
        reasoning: str
    ) -> Dict:
        """Inject LLM understanding into SAGE's processing."""
        # This would integrate with SAGE's H-module
        # For now, we just add it to the outputs
        sage_outputs['llm_context'] = llm_context
        sage_outputs['llm_reasoning'] = reasoning
        
        # In real implementation, would modify H-module hidden states
        if 'h_hidden' in sage_outputs:
            # Blend LLM context with H-module understanding
            alpha = 0.3  # Blending factor
            sage_outputs['h_hidden'] = (
                (1 - alpha) * sage_outputs['h_hidden'] + 
                alpha * llm_context
            )
        
        return sage_outputs
    
    def _compute_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        # This would be implemented based on the specific task
        # For now, return a dummy loss
        if 'logits' in outputs:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(outputs['logits'].view(-1, outputs['logits'].size(-1)), labels.view(-1))
        return torch.tensor(0.0, requires_grad=True)


def create_llm_guided_sage(
    sage_config,
    llm_model: str = "microsoft/phi-2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> LLMGuidedSAGE:
    """
    Factory function to create LLM-guided SAGE.
    
    Args:
        sage_config: Configuration for SAGE model
        llm_model: Name of the LLM to use
        device: Device to run on
        
    Returns:
        LLM-guided SAGE model
    """
    # Import here to avoid circular dependency
    from sage.core.sage_core import SAGECore
    
    # Create base SAGE
    sage = SAGECore(sage_config).to(device)
    
    # Wrap with LLM guidance
    guided_sage = LLMGuidedSAGE(
        sage_core=sage,
        llm_model=llm_model
    ).to(device)
    
    return guided_sage


if __name__ == "__main__":
    # Test the LLM interface
    print("Testing External LLM Interface...")
    
    # Use CPU for testing to avoid CUDA memory issues
    llm = ExternalLLMInterface(
        model_name="microsoft/phi-2",
        device="cpu",
        quantize=False
    )
    
    # Test pattern understanding
    test_pattern = torch.tensor([
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    ])
    
    llm.set_sage_dimensions(768)  # Example SAGE dimension
    
    context, reasoning = llm.understand_pattern(
        test_pattern,
        task_description="Identify the pattern in this grid"
    )
    
    print(f"Context shape: {context.shape}")
    print(f"Reasoning: {reasoning}")
    
    print("\n✅ LLM Interface ready for SAGE integration!")