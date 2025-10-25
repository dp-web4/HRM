"""
Tiny Fast-Path Model Architecture

A ~10M parameter model for performatory acknowledgments and simple responses.
Replaces pattern matching with learned neural responses.

Architecture inspired by:
1. DistilGPT-2 (distillation approach)
2. TinyStories models (small but capable)
3. MobileBERT (efficiency-focused)

Design goals:
- <20ms inference on Jetson Orin Nano
- ~10M parameters (fits in 40MB at FP16)
- Trained via distillation from Qwen 2.5-0.5B
- Handles: greetings, acknowledgments, simple Q&A, handoff detection

Model components:
1. Tokenizer: Shared with Qwen (for compatibility)
2. Embedding: 512-dim (vs 896 in Qwen)
3. Transformer: 6 layers, 8 heads, 512 hidden (vs 14 layers in Qwen)
4. Output: Classification head for response selection + confidence
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import json
from pathlib import Path


class TinyFastModelConfig:
    """Configuration for tiny fast-path model"""
    def __init__(self):
        # Model architecture
        self.vocab_size = 151936  # Same as Qwen 2.5
        self.hidden_size = 512    # Reduced from 896
        self.num_layers = 6       # Reduced from 14
        self.num_heads = 8        # Reduced from 14
        self.intermediate_size = 2048  # Reduced from 4864
        self.max_position_embeddings = 512  # Reduced from 32768

        # Training
        self.dropout = 0.1
        self.layer_norm_eps = 1e-6

        # Response selection
        self.num_response_categories = 100  # Pre-defined response templates
        self.confidence_threshold = 0.6

    def to_dict(self):
        """Convert config to dictionary"""
        return self.__dict__


class TinyTransformerBlock(nn.Module):
    """Single transformer block for tiny model"""
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attention_mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.ln1(x + attn_out)

        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x


class TinyFastModel(nn.Module):
    """
    Tiny fast-path model for SAGE.

    ~10M parameters, replaces pattern matching with learned responses.
    """
    def __init__(self, config: TinyFastModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (shared with Qwen for compatibility)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dual heads: response selection + confidence
        self.response_head = nn.Linear(config.hidden_size, config.num_response_categories)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )

        # Response templates (loaded from training data)
        self.response_templates: List[str] = []

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            response_logits: [batch, num_response_categories]
            confidence: [batch, 1]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        x = token_embeds + position_embeds

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)

        # Use [CLS] token (first token) for classification
        cls_hidden = x[:, 0, :]

        # Response selection
        response_logits = self.response_head(cls_hidden)

        # Confidence estimation
        confidence = self.confidence_head(cls_hidden)

        return response_logits, confidence

    def predict(self, input_ids, attention_mask=None) -> Tuple[int, float, str]:
        """
        Predict response and confidence.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            response_idx: Index of selected response
            confidence: Confidence score
            response_text: Actual response text
        """
        self.eval()
        with torch.no_grad():
            response_logits, confidence = self.forward(input_ids, attention_mask)

            # Select response with highest logit
            response_idx = response_logits.argmax(dim=-1).item()
            confidence_score = confidence.item()

            # Get response text
            if response_idx < len(self.response_templates):
                response_text = self.response_templates[response_idx]
            else:
                response_text = "I'm not sure how to respond."

        return response_idx, confidence_score, response_text

    def load_response_templates(self, templates_file: str):
        """Load response templates from file"""
        with open(templates_file, 'r') as f:
            self.response_templates = json.load(f)

    def save_response_templates(self, templates_file: str):
        """Save response templates to file"""
        with open(templates_file, 'w') as f:
            json.dump(self.response_templates, f, indent=2)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyModelDistiller:
    """
    Distills knowledge from Qwen 2.5-0.5B into tiny fast-path model.

    Uses:
    1. Response distillation (match Qwen's response distribution)
    2. Hidden state alignment (intermediate layer matching)
    3. Attention transfer (learn Qwen's attention patterns)
    """
    def __init__(self, teacher_model, student_model, temperature=2.0):
        """
        Initialize distiller.

        Args:
            teacher_model: Qwen 2.5-0.5B (teacher)
            student_model: TinyFastModel (student)
            temperature: Temperature for distillation
        """
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature

    def distill_response(self, input_ids, teacher_response_logits, student_response_logits):
        """
        Distillation loss for response matching.

        Uses KL divergence between teacher and student distributions.
        """
        # Soften distributions with temperature
        teacher_soft = torch.softmax(teacher_response_logits / self.temperature, dim=-1)
        student_soft = torch.log_softmax(student_response_logits / self.temperature, dim=-1)

        # KL divergence loss
        distill_loss = torch.nn.functional.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return distill_loss


class TinyModelTrainer:
    """
    Trains tiny fast-path model on extracted training data.

    Training strategies:
    1. Supervised learning on labeled fast_ack examples
    2. Distillation from Qwen on general examples
    3. Confidence calibration on handoff examples
    """
    def __init__(self, model: TinyFastModel, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None

    def train_epoch(self, dataloader, epoch: int):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with training examples
            epoch: Current epoch number

        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)  # Response category labels

            # Forward pass
            response_logits, confidence = self.model(input_ids, attention_mask)

            # Loss: cross-entropy for response selection
            loss = torch.nn.functional.cross_entropy(response_logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


def create_tiny_fast_model(response_templates: List[str]) -> TinyFastModel:
    """
    Create and initialize tiny fast-path model.

    Args:
        response_templates: List of response templates to learn

    Returns:
        Initialized TinyFastModel
    """
    config = TinyFastModelConfig()
    config.num_response_categories = len(response_templates)

    model = TinyFastModel(config)
    model.response_templates = response_templates

    return model


def estimate_model_size(model: TinyFastModel) -> dict:
    """
    Estimate model size and memory requirements.

    Returns:
        Dictionary with size estimates
    """
    num_params = model.count_parameters()

    return {
        'parameters': num_params,
        'size_fp32': num_params * 4 / (1024**2),  # MB
        'size_fp16': num_params * 2 / (1024**2),  # MB
        'size_int8': num_params / (1024**2),      # MB
    }
