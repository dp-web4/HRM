#!/usr/bin/env python3
"""
Selective Expert Loader for SAGE

Loads Qwen3-Omni experts on-demand based on:
- Metabolic state (determines how many experts to load)
- SNARC salience scores (influences expert selection)
- Trust scores (learned from convergence behavior)
- Router logits (standard MoE routing)

This implements SAGE's core thesis: selective resource loading
based on attention needs rather than loading everything.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertTrustRecord:
    """Trust tracking for individual experts"""

    def __init__(self, expert_id: int, layer_id: int, component: str):
        self.expert_id = expert_id
        self.layer_id = layer_id
        self.component = component

        # Trust metrics (learned over time)
        self.convergence_rate = 0.5  # How quickly it helps reduce energy
        self.stability = 0.5  # Consistency across similar inputs
        self.efficiency = 0.5  # Quality per computation cost

        # Usage statistics
        self.activation_count = 0
        self.success_count = 0
        self.last_used = 0.0

        # Context
        self.works_best_with: List[int] = []  # Other expert IDs
        self.metabolic_preferences: Dict[str, float] = {}

    def update_from_outcome(self, energy_decrease: float, stability_score: float):
        """Update trust based on observed behavior"""
        self.activation_count += 1

        # Update convergence rate (exponential moving average)
        alpha = 0.1
        self.convergence_rate = (1 - alpha) * self.convergence_rate + alpha * energy_decrease

        # Update stability
        self.stability = (1 - alpha) * self.stability + alpha * stability_score

        # Success if energy decreased and stable
        if energy_decrease > 0 and stability_score > 0.5:
            self.success_count += 1

        # Update efficiency (success rate)
        self.efficiency = self.success_count / self.activation_count if self.activation_count > 0 else 0.5

        self.last_used = time.time()

    @property
    def overall_trust(self) -> float:
        """Combined trust score (0-1)"""
        return (self.convergence_rate * 0.4 +
                self.stability * 0.3 +
                self.efficiency * 0.3)


class SelectiveExpertLoader:
    """
    Load experts on-demand for SAGE integration

    Manages:
    - Expert memory (load/unload based on need)
    - Router evaluation (determine which experts to activate)
    - Trust-based eviction (keep high-trust experts in memory)
    - SNARC-weighted selection (salience influences routing)
    """

    def __init__(
        self,
        extraction_dir: str,
        component: str = "thinker",
        device: str = "cuda",
        max_loaded_experts: int = 16
    ):
        """
        Args:
            extraction_dir: Path to extracted experts directory
            component: "thinker" or "talker"
            device: Where to load experts
            max_loaded_experts: Maximum experts to keep in memory
        """
        self.extraction_dir = Path(extraction_dir)
        self.component = component
        self.device = device
        self.max_loaded_experts = max_loaded_experts

        # Load manifest
        manifest_path = self.extraction_dir / "extraction_manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Paths
        self.experts_dir = self.extraction_dir / "experts"
        self.routers_dir = self.extraction_dir / "routers"

        # Loaded experts (layer_id -> {expert_id -> weights})
        self.loaded_experts: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

        # Loaded routers (layer_id -> weights)
        self.loaded_routers: Dict[int, torch.Tensor] = {}

        # Trust records (layer_id -> {expert_id -> ExpertTrustRecord})
        self.trust_records: Dict[int, Dict[int, ExpertTrustRecord]] = {}

        # LRU tracking
        self.expert_access_times: Dict[Tuple[int, int], float] = {}  # (layer, expert) -> timestamp

        print(f"✅ Selective Expert Loader initialized")
        print(f"   Component: {component}")
        print(f"   Max loaded experts: {max_loaded_experts}")
        print(f"   Device: {device}")

    def load_router(self, layer_id: int) -> torch.Tensor:
        """
        Load router for a layer (lightweight, keep all in memory)

        Returns:
            Router weight tensor [num_experts, hidden_size]
        """
        if layer_id in self.loaded_routers:
            return self.loaded_routers[layer_id]

        # Load from file
        router_file = self.routers_dir / f"{self.component}_router_layer_{layer_id:02d}.safetensors"

        with safetensors.safe_open(router_file, framework="pt") as f:
            router_key = f"{self.component}.model.layers.{layer_id}.mlp.gate.weight"
            # Convert to float32 for compatibility
            router_weight = f.get_tensor(router_key).to(self.device).float()

        self.loaded_routers[layer_id] = router_weight
        return router_weight

    def load_expert(self, expert_id: int, layer_id: int) -> Dict[str, torch.Tensor]:
        """
        Load a specific expert into memory

        Returns:
            Expert weights dict {proj_name: tensor}
        """
        # Check if already loaded
        if layer_id in self.loaded_experts and expert_id in self.loaded_experts[layer_id]:
            self.expert_access_times[(layer_id, expert_id)] = time.time()
            return self.loaded_experts[layer_id][expert_id]

        # Enforce max loaded experts (evict if needed)
        if self._count_loaded_experts() >= self.max_loaded_experts:
            self._evict_expert()

        # Load from file
        expert_file = self.experts_dir / f"{self.component}_expert_{expert_id:03d}_layer_{layer_id:02d}.safetensors"

        # Check if expert exists (handle sparse layers)
        if not expert_file.exists():
            print(f"⚠️  Expert {expert_id} not found in layer {layer_id} (sparse layer)")
            return None  # Return None for missing experts in sparse layers

        expert_weights = {}
        with safetensors.safe_open(expert_file, framework="pt") as f:
            for key in f.keys():
                # Convert to float32 for compatibility
                expert_weights[key] = f.get_tensor(key).to(self.device).float()

        # Store
        if layer_id not in self.loaded_experts:
            self.loaded_experts[layer_id] = {}

        self.loaded_experts[layer_id][expert_id] = expert_weights

        # Initialize trust record if needed
        if layer_id not in self.trust_records:
            self.trust_records[layer_id] = {}
        if expert_id not in self.trust_records[layer_id]:
            self.trust_records[layer_id][expert_id] = ExpertTrustRecord(expert_id, layer_id, self.component)

        # Update access time
        self.expert_access_times[(layer_id, expert_id)] = time.time()

        return expert_weights

    def select_experts_snarc(
        self,
        hidden_states: torch.Tensor,
        layer_id: int,
        num_experts: int = 8,
        snarc_salience: Optional[Dict[str, float]] = None,
        metabolic_state: str = "FOCUS"
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Select experts using SNARC-augmented routing

        Args:
            hidden_states: Input to layer [batch, seq, hidden]
            layer_id: Which layer
            num_experts: How many to activate (from metabolic state)
            snarc_salience: 5D salience scores (surprise, novelty, arousal, reward, conflict)
            metabolic_state: Current state (affects selection strategy)

        Returns:
            (selected_expert_ids, router_weights)
        """
        # Load router
        router = self.load_router(layer_id)

        # Compute router logits
        # router: [num_experts, hidden_size]
        # hidden_states: [batch, seq, hidden]
        # We'll use mean over sequence for expert selection
        pooled_hidden = hidden_states.mean(dim=1)  # [batch, hidden]

        router_logits = F.linear(pooled_hidden, router)  # [batch, num_experts]
        router_logits = router_logits.mean(dim=0)  # [num_experts] - average across batch

        # Standard MoE: just use top-k of router logits
        if snarc_salience is None:
            top_k_values, top_k_indices = torch.topk(router_logits, k=num_experts)
            return top_k_indices.tolist(), top_k_values

        # SNARC-augmented selection
        # Weight router logits by salience + trust
        snarc_scores = torch.zeros_like(router_logits)

        for expert_id in range(len(router_logits)):
            # Get trust record
            trust = 0.5  # Default
            if layer_id in self.trust_records and expert_id in self.trust_records[layer_id]:
                trust = self.trust_records[layer_id][expert_id].overall_trust

            # Combine router logits, trust, and salience
            base_score = router_logits[expert_id].item()

            # Weight by salience dimensions
            surprise_weight = snarc_salience.get('surprise', 0.5) * 0.2
            reward_weight = snarc_salience.get('reward', 0.5) * trust * 0.2
            novelty_weight = snarc_salience.get('novelty', 0.5) * 0.1

            # Augmented score
            augmented_score = (
                base_score * 0.5 +  # Base router
                surprise_weight +  # Prefer novel experts for surprising inputs
                reward_weight +  # Prefer proven experts for important tasks
                novelty_weight  # Explore new experts
            )

            snarc_scores[expert_id] = augmented_score

        # Select top-k with SNARC augmentation
        top_k_values, top_k_indices = torch.topk(snarc_scores, k=num_experts)

        return top_k_indices.tolist(), top_k_values

    def _count_loaded_experts(self) -> int:
        """Count total loaded experts across all layers"""
        return sum(len(experts) for experts in self.loaded_experts.values())

    def _evict_expert(self):
        """
        Evict least valuable expert from memory

        Uses LRU + trust-weighted eviction:
        - Keep high-trust experts longer
        - Evict rarely-used experts first
        """
        if self._count_loaded_experts() == 0:
            return

        # Find expert with highest eviction score
        best_candidate = None
        best_score = -float('inf')

        for layer_id, experts in self.loaded_experts.items():
            for expert_id in experts.keys():
                # Get trust
                trust = 0.5
                if layer_id in self.trust_records and expert_id in self.trust_records[layer_id]:
                    trust_record = self.trust_records[layer_id][expert_id]
                    trust = trust_record.overall_trust

                # Get last access time
                last_access = self.expert_access_times.get((layer_id, expert_id), 0)
                time_since_use = time.time() - last_access

                # Eviction score (higher = evict sooner)
                eviction_score = (
                    time_since_use * 0.4 +  # Recency
                    (1 - trust) * 0.6  # Keep high-trust experts
                )

                if eviction_score > best_score:
                    best_score = eviction_score
                    best_candidate = (layer_id, expert_id)

        # Evict
        if best_candidate:
            layer_id, expert_id = best_candidate
            del self.loaded_experts[layer_id][expert_id]
            if len(self.loaded_experts[layer_id]) == 0:
                del self.loaded_experts[layer_id]

            print(f"🗑️  Evicted expert {expert_id} from layer {layer_id} (trust: {trust:.3f})")

    def update_expert_trust(
        self,
        expert_id: int,
        layer_id: int,
        energy_decrease: float,
        stability_score: float
    ):
        """Update trust record after using an expert"""
        if layer_id not in self.trust_records:
            self.trust_records[layer_id] = {}
        if expert_id not in self.trust_records[layer_id]:
            self.trust_records[layer_id][expert_id] = ExpertTrustRecord(expert_id, layer_id, self.component)

        self.trust_records[layer_id][expert_id].update_from_outcome(energy_decrease, stability_score)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        expert_memory = 0
        for layer_experts in self.loaded_experts.values():
            for expert_weights in layer_experts.values():
                for tensor in expert_weights.values():
                    expert_memory += tensor.numel() * tensor.element_size()

        router_memory = 0
        for router in self.loaded_routers.values():
            router_memory += router.numel() * router.element_size()

        return {
            "experts_mb": expert_memory / 1024**2,
            "routers_mb": router_memory / 1024**2,
            "total_mb": (expert_memory + router_memory) / 1024**2,
            "num_loaded_experts": self._count_loaded_experts(),
            "num_loaded_routers": len(self.loaded_routers)
        }

    def save_trust_records(self, output_path: str):
        """Save learned trust records for persistence"""
        trust_data = {}

        for layer_id, experts in self.trust_records.items():
            trust_data[layer_id] = {}
            for expert_id, record in experts.items():
                trust_data[layer_id][expert_id] = {
                    "convergence_rate": record.convergence_rate,
                    "stability": record.stability,
                    "efficiency": record.efficiency,
                    "activation_count": record.activation_count,
                    "success_count": record.success_count,
                    "overall_trust": record.overall_trust
                }

        with open(output_path, 'w') as f:
            json.dump(trust_data, f, indent=2)

        print(f"✅ Trust records saved: {output_path}")
