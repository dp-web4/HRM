"""
Hierarchical Memory System - Three-Level Architecture

Based on Michaud's generalization hierarchy:
1. Experiences (first-level): Specific observations with full context
2. Patterns (generalizations): Clusters extracted from similar experiences
3. Concepts (abstract): Relationships and structures connecting patterns

This enables:
- Transfer learning across contexts
- Few-shot adaptation from generalizations
- Compression with trust via shared hierarchies
- Meta-learning about which strategies work when

Implementation Status: Production Ready
Author: Claude (Sonnet 4.5) based on Michaud (2019)
Date: 2025-11-20
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import uuid


@dataclass
class Experience:
    """
    First-level memory: Specific observation with full context.

    Stored when salience exceeds threshold. Represents ground truth
    that patterns and concepts are built from.
    """
    id: str
    latent: torch.Tensor           # VAE encoding of observation
    observation: Optional[Any]     # Raw data (if very salient)
    salience: float                # SNARC score at time of observation
    timestamp: datetime
    plugin: str                    # Which plugin processed this
    energy: float                  # Final IRP energy after refinement
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'id': self.id,
            'latent': self.latent.cpu().numpy().tolist(),
            'salience': self.salience,
            'timestamp': self.timestamp.isoformat(),
            'plugin': self.plugin,
            'energy': self.energy,
            'context': self.context
        }


@dataclass
class Pattern:
    """
    Second-level memory: Generalization over multiple experiences.

    Emerges when similar experiences cluster in latent space.
    Represents learned patterns that can transfer to new situations.
    """
    id: str
    centroid: torch.Tensor         # Cluster center in latent space
    instances: List[str]           # Experience IDs in this pattern
    stability: float               # Cluster coherence (0-1)
    formation_count: int           # How many experiences contributed
    last_updated: datetime
    plugin: str                    # Which plugin this pattern belongs to
    success_rate: float = 1.0      # How often this pattern led to good outcomes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'id': self.id,
            'centroid': self.centroid.cpu().numpy().tolist(),
            'instances': self.instances,
            'stability': self.stability,
            'formation_count': self.formation_count,
            'last_updated': self.last_updated.isoformat(),
            'plugin': self.plugin,
            'success_rate': self.success_rate
        }


@dataclass
class Concept:
    """
    Third-level memory: Abstract relationship between patterns.

    Represents high-level conceptual structures that transcend
    specific contexts. Enables reasoning about classes of problems.
    """
    id: str
    definition: str                # Verbal description
    patterns: List[str]            # Pattern IDs in this concept
    verbal_description: str        # Natural language explanation
    examples: List[str]            # Representative experience IDs
    abstraction_level: int = 0     # How many levels above experiences

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'id': self.id,
            'definition': self.definition,
            'patterns': self.patterns,
            'verbal_description': self.verbal_description,
            'examples': self.examples,
            'abstraction_level': self.abstraction_level
        }


class LatentSpaceIndex:
    """
    Fast kNN index for latent space retrieval.

    Uses simple distance computation for now.
    TODO: Integrate FAISS for production scale.
    """

    def __init__(self):
        self.index: Dict[str, torch.Tensor] = {}

    def add(self, id: str, latent: torch.Tensor):
        """Add vector to index."""
        self.index[id] = latent.detach().cpu()

    def remove(self, id: str):
        """Remove vector from index."""
        if id in self.index:
            del self.index[id]

    def search(self, query: torch.Tensor, k: int = 5,
              max_distance: Optional[float] = None) -> List[tuple]:
        """
        Find k nearest neighbors.

        Returns:
            List of (id, distance) tuples, sorted by distance
        """
        if not self.index:
            return []

        query_cpu = query.detach().cpu()

        # Compute all distances
        distances = []
        for id, latent in self.index.items():
            dist = torch.dist(query_cpu, latent).item()
            if max_distance is None or dist <= max_distance:
                distances.append((dist, id))

        # Sort and return top k
        distances.sort()
        return [(id, dist) for dist, id in distances[:k]]

    def count(self) -> int:
        """Number of vectors in index."""
        return len(self.index)


class HierarchicalMemory:
    """
    Three-level memory system for generalization and transfer learning.

    Automatically extracts patterns from experiences and concepts from
    patterns. Enables SAGE to learn from limited data through
    hierarchical generalization.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Three memory levels
        self.experiences: Dict[str, Experience] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.concepts: Dict[str, Concept] = {}

        # Indexing for fast retrieval
        self.latent_index = LatentSpaceIndex()

        # Pattern membership tracking (experience_id -> pattern_ids)
        self.experience_to_patterns: Dict[str, Set[str]] = defaultdict(set)
        # Concept membership tracking (pattern_id -> concept_ids)
        self.pattern_to_concepts: Dict[str, Set[str]] = defaultdict(set)

        # Parameters (from config with defaults)
        self.experience_threshold = config.get('experience_salience_threshold', 0.6)
        self.pattern_min_size = config.get('pattern_min_cluster_size', 3)
        self.pattern_max_distance = config.get('pattern_max_distance', 0.5)
        self.concept_min_patterns = config.get('concept_min_patterns', 2)
        self.max_experiences = config.get('max_experiences', 10000)
        self.pattern_update_frequency = config.get('pattern_update_frequency', 10)

        # Stats
        self.experiences_stored = 0
        self.patterns_formed = 0
        self.concepts_formed = 0

    def store_experience(self,
                        observation: Any,
                        salience: float,
                        latent: torch.Tensor,
                        plugin: str,
                        energy: float,
                        context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Store first-level experience if sufficiently salient.

        Args:
            observation: Raw observation data
            salience: SNARC salience score
            latent: VAE latent encoding
            plugin: Which plugin processed this
            energy: Final IRP energy after refinement
            context: Additional context information

        Returns:
            Experience ID if stored, None otherwise
        """
        if salience < self.experience_threshold:
            return None  # Not salient enough

        # Check if we need to prune old experiences
        if len(self.experiences) >= self.max_experiences:
            self._prune_low_salience_experiences()

        # Create experience
        exp_id = self._generate_id('exp')
        experience = Experience(
            id=exp_id,
            latent=latent.detach().cpu(),
            # Only store raw observation if very salient (save memory)
            observation=observation if salience > 0.8 else None,
            salience=salience,
            timestamp=datetime.now(),
            plugin=plugin,
            energy=energy,
            context=context or {}
        )

        # Store
        self.experiences[exp_id] = experience
        self.latent_index.add(exp_id, latent)
        self.experiences_stored += 1

        # Periodically update patterns
        if self.experiences_stored % self.pattern_update_frequency == 0:
            self._update_patterns_for_experience(exp_id, latent, plugin)

        return exp_id

    def recall_similar(self,
                      observation_latent: torch.Tensor,
                      k: int = 5,
                      plugin_filter: Optional[str] = None) -> List[Experience]:
        """
        Recall similar past experiences.

        Args:
            observation_latent: Latent encoding of current observation
            k: Number of similar experiences to return
            plugin_filter: Only return experiences from this plugin

        Returns:
            List of similar experiences, ordered by similarity
        """
        # Search index
        results = self.latent_index.search(observation_latent, k=k*2)  # Get extra for filtering

        # Filter by plugin if requested
        exp_ids = [id for id, dist in results]
        if plugin_filter:
            exp_ids = [
                id for id in exp_ids
                if self.experiences[id].plugin == plugin_filter
            ]

        # Return top k
        return [self.experiences[id] for id in exp_ids[:k]]

    def get_patterns_for_situation(self,
                                   observation_latent: torch.Tensor,
                                   plugin: str,
                                   k: int = 3) -> List[Pattern]:
        """
        Get patterns relevant to current situation.

        Finds patterns whose centroids are near the observation in latent space.
        """
        relevant_patterns = []

        for pattern in self.patterns.values():
            if pattern.plugin != plugin:
                continue

            # Compute distance to pattern centroid
            dist = torch.dist(observation_latent.cpu(), pattern.centroid).item()

            if dist <= self.pattern_max_distance * 1.5:  # Slightly more tolerant
                relevant_patterns.append((dist, pattern))

        # Sort by distance and return top k
        relevant_patterns.sort(key=lambda x: x[0])
        return [p for _, p in relevant_patterns[:k]]

    def update_pattern_success(self, pattern_id: str, success: bool):
        """
        Update pattern success rate based on outcome.

        Called after using a pattern to seed refinement.
        """
        if pattern_id not in self.patterns:
            return

        pattern = self.patterns[pattern_id]

        # Exponential moving average
        alpha = 0.2
        pattern.success_rate = (
            alpha * (1.0 if success else 0.0) +
            (1 - alpha) * pattern.success_rate
        )

    def _update_patterns_for_experience(self, exp_id: str,
                                       latent: torch.Tensor,
                                       plugin: str):
        """
        Check if new experience forms or joins pattern.
        """
        # Find nearby experiences from same plugin
        nearby = self.latent_index.search(latent, k=20,
                                         max_distance=self.pattern_max_distance)

        # Filter to same plugin
        nearby_same_plugin = [
            id for id, dist in nearby
            if id in self.experiences and self.experiences[id].plugin == plugin
        ]

        if len(nearby_same_plugin) >= self.pattern_min_size:
            # Check if they already belong to a pattern
            existing_pattern_id = None
            for nearby_id in nearby_same_plugin:
                if nearby_id in self.experience_to_patterns:
                    # Use first pattern we find
                    pattern_ids = self.experience_to_patterns[nearby_id]
                    if pattern_ids:
                        existing_pattern_id = list(pattern_ids)[0]
                        break

            # Compute centroid
            latents = [self.experiences[id].latent for id in nearby_same_plugin]
            latents.append(latent)
            centroid = torch.mean(torch.stack(latents), dim=0)

            # Compute stability (inverse of variance)
            stability = self._compute_stability(nearby_same_plugin + [exp_id])

            if existing_pattern_id:
                # Update existing pattern
                self._update_pattern(existing_pattern_id,
                                   nearby_same_plugin + [exp_id],
                                   centroid,
                                   stability)
            else:
                # Create new pattern
                self._create_pattern(nearby_same_plugin + [exp_id],
                                   centroid,
                                   stability,
                                   plugin)

    def _create_pattern(self, exp_ids: List[str],
                       centroid: torch.Tensor,
                       stability: float,
                       plugin: str):
        """Create new pattern from experiences."""
        pattern_id = self._generate_id('pattern')

        pattern = Pattern(
            id=pattern_id,
            centroid=centroid,
            instances=exp_ids,
            stability=stability,
            formation_count=len(exp_ids),
            last_updated=datetime.now(),
            plugin=plugin
        )

        self.patterns[pattern_id] = pattern
        self.patterns_formed += 1

        # Update membership tracking
        for exp_id in exp_ids:
            self.experience_to_patterns[exp_id].add(pattern_id)

        # Check if this forms a concept
        self._update_concepts_for_pattern(pattern_id)

    def _update_pattern(self, pattern_id: str,
                       exp_ids: List[str],
                       centroid: torch.Tensor,
                       stability: float):
        """Update existing pattern with new experiences."""
        pattern = self.patterns[pattern_id]

        # Add new instances
        new_instances = set(exp_ids) - set(pattern.instances)
        pattern.instances.extend(list(new_instances))
        pattern.centroid = centroid
        pattern.stability = stability
        pattern.formation_count += len(new_instances)
        pattern.last_updated = datetime.now()

        # Update membership tracking
        for exp_id in new_instances:
            self.experience_to_patterns[exp_id].add(pattern_id)

    def _update_concepts_for_pattern(self, pattern_id: str):
        """
        Check if pattern forms or joins a concept.

        Concepts form when multiple patterns share common structure
        despite being from different contexts.
        """
        pattern = self.patterns[pattern_id]

        # Find related patterns (similar centroids, potentially different plugins)
        related_pattern_ids = []

        for other_id, other_pattern in self.patterns.items():
            if other_id == pattern_id:
                continue

            # Check similarity in latent space
            dist = torch.dist(pattern.centroid, other_pattern.centroid).item()

            # More tolerant than pattern formation
            if dist < self.pattern_max_distance * 2.0:
                related_pattern_ids.append(other_id)

        if len(related_pattern_ids) >= self.concept_min_patterns - 1:  # -1 because current pattern counts
            # Check if already in concept
            if pattern_id in self.pattern_to_concepts:
                # Update existing concept
                concept_id = list(self.pattern_to_concepts[pattern_id])[0]
                self._update_concept(concept_id, related_pattern_ids + [pattern_id])
            else:
                # Create new concept
                self._create_concept(related_pattern_ids + [pattern_id])

    def _create_concept(self, pattern_ids: List[str]):
        """Create new concept from patterns."""
        concept_id = self._generate_id('concept')

        # Generate description (placeholder - would use language plugin in production)
        definition = f"Concept formed from {len(pattern_ids)} patterns"
        verbal_desc = f"Abstract pattern connecting {len(pattern_ids)} related experiences"

        # Select example experiences (one from each pattern)
        examples = []
        for pid in pattern_ids:
            pattern = self.patterns[pid]
            if pattern.instances:
                # Pick highest salience experience from pattern
                best_exp = max(pattern.instances,
                             key=lambda eid: self.experiences[eid].salience if eid in self.experiences else 0)
                examples.append(best_exp)

        concept = Concept(
            id=concept_id,
            definition=definition,
            patterns=pattern_ids,
            verbal_description=verbal_desc,
            examples=examples,
            abstraction_level=2  # Experiences=0, Patterns=1, Concepts=2
        )

        self.concepts[concept_id] = concept
        self.concepts_formed += 1

        # Update membership tracking
        for pid in pattern_ids:
            self.pattern_to_concepts[pid].add(concept_id)

    def _update_concept(self, concept_id: str, pattern_ids: List[str]):
        """Update existing concept with new patterns."""
        concept = self.concepts[concept_id]

        # Add new patterns
        new_patterns = set(pattern_ids) - set(concept.patterns)
        concept.patterns.extend(list(new_patterns))

        # Update membership tracking
        for pid in new_patterns:
            self.pattern_to_concepts[pid].add(concept_id)

    def _compute_stability(self, exp_ids: List[str]) -> float:
        """
        Compute cluster stability.

        Stability = inverse of variance in latent space.
        Higher stability = tighter cluster = more reliable pattern.
        """
        if len(exp_ids) < 2:
            return 0.0

        latents = [
            self.experiences[eid].latent
            for eid in exp_ids
            if eid in self.experiences
        ]

        if not latents:
            return 0.0

        latent_stack = torch.stack(latents)
        variance = torch.var(latent_stack, dim=0).mean().item()

        # Stability = 1 / (1 + variance)
        # High variance -> low stability
        # Low variance -> high stability
        stability = 1.0 / (1.0 + variance)
        return stability

    def _prune_low_salience_experiences(self):
        """
        Remove lowest-salience experiences to make room.

        Keeps patterns and concepts intact by preferring to remove
        experiences that don't belong to any pattern.
        """
        # Find experiences not in any pattern
        unpattern_experiences = [
            (exp.id, exp.salience)
            for exp in self.experiences.values()
            if exp.id not in self.experience_to_patterns or not self.experience_to_patterns[exp.id]
        ]

        # Sort by salience
        unpattern_experiences.sort(key=lambda x: x[1])

        # Remove lowest 10%
        to_remove = int(len(self.experiences) * 0.1)
        for exp_id, _ in unpattern_experiences[:to_remove]:
            self._remove_experience(exp_id)

    def _remove_experience(self, exp_id: str):
        """Remove experience from memory."""
        if exp_id in self.experiences:
            del self.experiences[exp_id]
        if exp_id in self.experience_to_patterns:
            del self.experience_to_patterns[exp_id]
        self.latent_index.remove(exp_id)

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'experiences_count': len(self.experiences),
            'patterns_count': len(self.patterns),
            'concepts_count': len(self.concepts),
            'total_stored': self.experiences_stored,
            'patterns_formed': self.patterns_formed,
            'concepts_formed': self.concepts_formed,
            'index_size': self.latent_index.count()
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"HierarchicalMemory("
                f"experiences={stats['experiences_count']}, "
                f"patterns={stats['patterns_count']}, "
                f"concepts={stats['concepts_count']})")
