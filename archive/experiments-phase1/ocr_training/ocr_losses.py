"""
Ontological Coherence Reward (OCR) Losses

Implements three auxiliary losses for geometric training:
1. Stability: Penalizes sensitivity to perturbations (Lipschitz-like)
2. Center loss: Encourages compact class clusters
3. Separation loss: Pushes class centroids apart
4. Calibration: Brier score for well-calibrated predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCRLosses(nn.Module):
    """
    Ontological Coherence Reward losses for geometric training.

    Creates well-structured latent spaces with:
    - Compact clusters (center loss)
    - Well-separated clusters (separation loss)
    - Smooth manifolds (stability loss)
    - Calibrated boundaries (Brier loss)
    """

    def __init__(self, num_labels, hidden_dim, config):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.config = config

        # Class centroids (learned via EMA)
        self.register_buffer("centers", torch.zeros(num_labels, hidden_dim))
        self.register_buffer("centers_initialized", torch.tensor(0))

    def stability_loss(self, cls_rep, logits_clean, classifier):
        """
        Penalize sensitivity to small perturbations (Lipschitz-like constraint).

        Args:
            cls_rep: Clean CLS representations [B, H]
            logits_clean: Logits from clean representations [B, C]
            classifier: Function mapping cls_rep -> logits

        Returns:
            Stability loss (scalar)
        """
        # Add small noise to CLS representation
        noise = torch.randn_like(cls_rep) * self.config.noise_std
        cls_noisy = cls_rep + noise

        # Get logits from noisy representation
        logits_noisy = classifier(cls_noisy)

        # Penalize large changes in logits
        stability = ((logits_noisy - logits_clean)**2).mean()

        return stability

    def center_loss(self, cls_rep, labels):
        """
        Pull features toward their class centroids.

        Args:
            cls_rep: CLS representations [B, H]
            labels: Class labels [B]

        Returns:
            Center loss (scalar)
        """
        # Get centers for this batch
        centers_batch = self.centers[labels]  # [B, H]

        # L2 distance to centers
        center = ((cls_rep - centers_batch)**2).mean()

        return center

    def separation_loss(self):
        """
        Push different class centroids apart.

        Returns:
            Separation loss (scalar)
        """
        # Compute pairwise distances between centers
        dist = torch.cdist(self.centers, self.centers, p=2)  # [C, C]

        # Create identity matrix to ignore diagonal
        I = torch.eye(self.num_labels, device=self.centers.device)

        # Penalize small inter-center distances
        # exp(-dist) is large when dist is small
        sep = torch.exp(-dist + I).sum() - torch.exp(torch.tensor(0.0, device=self.centers.device)) * self.num_labels

        return sep

    def brier_loss(self, logits, labels):
        """
        Calibration loss via Brier score.

        Args:
            logits: Model predictions [B, C]
            labels: Ground truth [B]

        Returns:
            Brier score (scalar)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # [B, C]

        # One-hot encode labels
        one_hot = F.one_hot(labels, num_classes=self.num_labels).float()  # [B, C]

        # Brier score: sum of squared differences
        brier = ((probs - one_hot)**2).sum(dim=1).mean()

        return brier

    def update_centers(self, cls_rep, labels, momentum=0.97):
        """
        Update class centroids via exponential moving average.

        Args:
            cls_rep: CLS representations [B, H]
            labels: Class labels [B]
            momentum: EMA momentum (default 0.97)
        """
        with torch.no_grad():
            if self.centers_initialized.item() == 0:
                # Initialize centers as class means from first batch
                for i in range(self.num_labels):
                    mask = (labels == i)
                    if mask.any():
                        self.centers[i] = cls_rep[mask].mean(dim=0)
                self.centers_initialized.fill_(1)
            else:
                # EMA update
                for i in range(self.num_labels):
                    mask = (labels == i)
                    if mask.any():
                        batch_mean = cls_rep[mask].mean(dim=0)
                        self.centers[i] = momentum * self.centers[i] + (1 - momentum) * batch_mean

    def compute_all(self, cls_rep, logits, labels, classifier):
        """
        Compute all OCR losses and update centers.

        Args:
            cls_rep: CLS representations [B, H]
            logits: Clean logits [B, C]
            labels: Ground truth [B]
            classifier: Function cls_rep -> logits

        Returns:
            Dictionary of losses
        """
        # Update centers first
        self.update_centers(cls_rep, labels)

        # Compute individual losses
        stab = self.stability_loss(cls_rep, logits, classifier)
        center = self.center_loss(cls_rep, labels)
        sep = self.separation_loss()
        brier = self.brier_loss(logits, labels)

        # Weighted combination
        total_ocr = (self.config.lambda_stab * stab +
                     self.config.lambda_center * center +
                     self.config.lambda_sep * sep +
                     self.config.lambda_brier * brier)

        return {
            'total_ocr': total_ocr,
            'stability': stab,
            'center': center,
            'separation': sep,
            'brier': brier
        }


class OCRConfig:
    """Configuration for OCR losses"""

    def __init__(self,
                 use_ocr=True,
                 lambda_stab=0.2,
                 lambda_center=0.1,
                 lambda_sep=0.05,
                 lambda_brier=0.1,
                 noise_std=1e-3):
        self.use_ocr = use_ocr
        self.lambda_stab = lambda_stab
        self.lambda_center = lambda_center
        self.lambda_sep = lambda_sep
        self.lambda_brier = lambda_brier
        self.noise_std = noise_std
