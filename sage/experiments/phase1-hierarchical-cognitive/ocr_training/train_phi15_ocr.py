"""
Train Phi-1.5 with Ontological Coherence Reward (OCR) losses

Experiment to test whether OCR-trained models work better with orchestration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from pathlib import Path
import json
import sys
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.diverse_prompts import get_all_prompts, get_prompt_metadata
from ocr_training.ocr_losses import OCRLosses, OCRConfig


class EpistemicDataset(Dataset):
    """Dataset of epistemic prompts with pseudo-labels"""

    def __init__(self, prompts, tokenizer, max_length=128):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create pseudo-labels based on prompt category
        self.labels = []
        for prompt in prompts:
            meta = get_prompt_metadata(prompt)
            # Map categories to label IDs
            category_map = {
                'epistemology': 0,
                'self_referential': 1,
                'scientific_reasoning': 2,
                'ethical_dilemmas': 3,
                'abstract_concepts': 4,
                'practical_problems': 5,
                'debates': 6,
                'uncertainty_scenarios': 7,
                'meta_cognitive': 8
            }
            self.labels.append(category_map.get(meta['category'], 0))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class Phi15OCRModel(nn.Module):
    """Phi-1.5 with classification head and OCR losses"""

    def __init__(self, model_name, num_labels, ocr_config):
        super().__init__()

        # Load pretrained Phi-1.5
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)

        hidden_dim = config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # OCR losses
        self.ocr_losses = OCRLosses(num_labels, hidden_dim, ocr_config)
        self.ocr_config = ocr_config
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        # Get encoder output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Use CLS token (first token) from last hidden state
        # Phi models use first token as sentence representation
        hidden_states = outputs.last_hidden_state  # [B, L, H]
        cls_rep = hidden_states[:, 0, :]  # [B, H]

        # Classification
        cls_dropout = self.dropout(cls_rep)
        logits = self.classifier(cls_dropout)

        loss = None
        loss_dict = {}

        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits, labels)
            loss_dict['ce'] = ce_loss.item()

            if self.ocr_config.use_ocr:
                # Compute OCR losses
                ocr_losses = self.ocr_losses.compute_all(
                    cls_rep=cls_rep,
                    logits=logits,
                    labels=labels,
                    classifier=lambda x: self.classifier(self.dropout(x))
                )

                # Total loss
                loss = ce_loss + ocr_losses['total_ocr']

                # Log individual components
                loss_dict.update({
                    'ocr_total': ocr_losses['total_ocr'].item(),
                    'stability': ocr_losses['stability'].item(),
                    'center': ocr_losses['center'].item(),
                    'separation': ocr_losses['separation'].item(),
                    'brier': ocr_losses['brier'].item()
                })
            else:
                loss = ce_loss

        return {
            'loss': loss,
            'logits': logits,
            'cls_rep': cls_rep.detach(),
            'loss_dict': loss_dict
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    loss_components = {
        'ce': 0, 'ocr_total': 0, 'stability': 0,
        'center': 0, 'separation': 0, 'brier': 0
    }

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        outputs = model(**batch)
        loss = outputs['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track losses
        total_loss += loss.item()
        for k, v in outputs['loss_dict'].items():
            if k in loss_components:
                loss_components[k] += v

        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Average losses
    n = len(dataloader)
    return {
        'total': total_loss / n,
        **{k: v/n for k, v in loss_components.items()}
    }


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()

    correct = 0
    total = 0
    all_cls_reps = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs['logits']
            labels = batch['labels']

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_cls_reps.append(outputs['cls_rep'])
            all_labels.append(labels)

    accuracy = correct / total

    # Compute cluster compactness
    cls_reps = torch.cat(all_cls_reps, dim=0)
    labels = torch.cat(all_labels, dim=0)

    compactness = 0
    for label_id in labels.unique():
        mask = (labels == label_id)
        if mask.sum() > 1:
            class_reps = cls_reps[mask]
            center = class_reps.mean(dim=0, keepdim=True)
            variance = ((class_reps - center)**2).sum(dim=1).mean().item()
            compactness += variance
    compactness /= len(labels.unique())

    return {
        'accuracy': accuracy,
        'compactness': compactness
    }


def main():
    # Config
    MODEL_NAME = "microsoft/phi-1_5"
    NUM_LABELS = 9  # 9 prompt categories
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LR = 3e-5
    MAX_LENGTH = 128

    # OCR config
    ocr_config = OCRConfig(
        use_ocr=True,
        lambda_stab=0.2,
        lambda_center=0.1,
        lambda_sep=0.05,
        lambda_brier=0.1,
        noise_std=1e-3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading prompts...")
    all_prompts = get_all_prompts()

    # Split train/val (80/20)
    split_idx = int(0.8 * len(all_prompts))
    train_prompts = all_prompts[:split_idx]
    val_prompts = all_prompts[split_idx:]

    print(f"Train: {len(train_prompts)}, Val: {len(val_prompts)}")

    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    train_dataset = EpistemicDataset(train_prompts, tokenizer, MAX_LENGTH)
    val_dataset = EpistemicDataset(val_prompts, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print("\nInitializing model...")
    model = Phi15OCRModel(MODEL_NAME, NUM_LABELS, ocr_config).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    history = {'train': [], 'val': []}

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"\nTrain metrics:")
        print(f"  Total loss: {train_metrics['total']:.4f}")
        print(f"  CE: {train_metrics['ce']:.4f}")
        if ocr_config.use_ocr:
            print(f"  OCR total: {train_metrics['ocr_total']:.4f}")
            print(f"    Stability: {train_metrics['stability']:.6f}")
            print(f"    Center: {train_metrics['center']:.6f}")
            print(f"    Separation: {train_metrics['separation']:.6f}")
            print(f"    Brier: {train_metrics['brier']:.6f}")

        # Eval
        val_metrics = evaluate(model, val_loader, device)
        print(f"\nVal metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Compactness: {val_metrics['compactness']:.4f}")

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_dir = Path(f"models/phi15_ocr/checkpoint_epoch_{epoch+1}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model.encoder.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save classifier head and OCR state
            torch.save({
                'classifier': model.classifier.state_dict(),
                'ocr_centers': model.ocr_losses.centers,
                'epoch': epoch + 1,
                'ocr_config': vars(ocr_config)
            }, checkpoint_dir / "ocr_state.pt")

            print(f"  → Checkpoint saved: {checkpoint_dir}")

    # Save final model
    final_dir = Path("models/phi15_ocr/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save({
        'classifier': model.classifier.state_dict(),
        'ocr_centers': model.ocr_losses.centers,
        'ocr_config': vars(ocr_config)
    }, final_dir / "ocr_state.pt")

    # Save training history
    with open("models/phi15_ocr/training_history.json", 'w') as f:
        # Convert tensors to lists for JSON
        history_json = {
            'train': [{k: float(v) if isinstance(v, (int, float)) else v
                      for k, v in epoch.items()} for epoch in history['train']],
            'val': [{k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in epoch.items()} for epoch in history['val']]
        }
        json.dump(history_json, f, indent=2)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Final model saved: {final_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
