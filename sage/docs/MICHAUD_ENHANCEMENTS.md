# SAGE Enhancements Based on Michaud's Neurolinguistics

**Date:** 2025-11-20
**Status:** Implementation Proposals
**Priority:** High - Addresses core consciousness capabilities

---

## Overview

This document provides **actionable implementation proposals** for enhancing SAGE based on insights from André Michaud's research on the mechanics of conceptual thinking. Each enhancement is prioritized, specified in detail, and includes implementation guidance.

---

## Table of Contents

1. [Priority Matrix](#priority-matrix)
2. [Enhancement 1: Three-System Architecture Completion](#enhancement-1-three-system-architecture-completion)
3. [Enhancement 2: Hierarchical Memory System](#enhancement-2-hierarchical-memory-system)
4. [Enhancement 3: Attention-Driven Resource Allocation](#enhancement-3-attention-driven-resource-allocation)
5. [Enhancement 4: Emotional Energy Functions](#enhancement-4-emotional-energy-functions)
6. [Enhancement 5: Cogitation Plugin](#enhancement-5-cogitation-plugin)
7. [Enhancement 6: Cross-Modal VAE Enhancement](#enhancement-6-cross-modal-vae-enhancement)
8. [Enhancement 7: Memory Integration into Main Loop](#enhancement-7-memory-integration-into-main-loop)
9. [Enhancement 8: Meta-Learning System](#enhancement-8-meta-learning-system)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Priority Matrix

| Enhancement | Impact | Complexity | Priority | Timeline |
|------------|--------|------------|----------|----------|
| Memory Integration (E7) | Critical | Medium | **P0** | 1-2 weeks |
| Hierarchical Memory (E2) | High | Medium | **P1** | 2-3 weeks |
| Attention Manager (E3) | High | Low | **P1** | 1 week |
| Emotional Energy (E4) | Medium | Low | **P2** | 1 week |
| Cogitation Plugin (E5) | High | Medium | **P2** | 2 weeks |
| Mathematical Reasoning (E1) | High | High | **P3** | 4-6 weeks |
| Cross-Modal VAE (E6) | Medium | High | **P3** | 3-4 weeks |
| Meta-Learning (E8) | Medium | Medium | **P3** | 2-3 weeks |

**Priority Definitions:**
- **P0 (Critical):** Blocking other enhancements, core functionality
- **P1 (High):** Major capability improvement, should do soon
- **P2 (Medium):** Significant enhancement, do after P1
- **P3 (Future):** Important but can be deferred

---

## Enhancement 1: Three-System Architecture Completion

**Status:** Partial (2/3 complete)
**Priority:** P3
**Effort:** 4-6 weeks

### Current State

- ✅ **First System (Sensory):** Vision, Audio plugins operational
- ✅ **Second System (Verbal):** Language plugin with Qwen 2.5-0.5B
- ❌ **Third System (Mathematical):** Not implemented

### Proposed Implementation

#### 1.1 Mathematical Reasoning Plugin

**File:** `/HRM/sage/irp/mathematical_reasoning.py`

```python
"""
Mathematical Reasoning Plugin - Third System of Signalization

Implements Michaud's third system: nonverbal symbolic thinking
in separate areas from language processing.

Based on Amalric & Dehaene (2016) findings that mathematical
thinking occurs in distinct neocortex regions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from .base import IRPPlugin, IRPState


class SymbolicRepresentation:
    """
    Symbolic mathematical representation.

    Encodes equations, geometric forms, and logical statements
    in structured format for manipulation.
    """

    def __init__(self, expression: str, domain: str = 'algebra'):
        self.expression = expression
        self.domain = domain  # 'algebra', 'geometry', 'calculus', etc.
        self.parse_tree = self._parse(expression)
        self.constraints = []

    def _parse(self, expression: str):
        """Parse expression into symbolic tree."""
        # Use sympy or custom parser
        import sympy as sp
        return sp.sympify(expression)

    def apply_transformation(self, rule: str):
        """Apply mathematical transformation rule."""
        # Implement symbolic manipulation
        pass

    def evaluate_coherence(self) -> float:
        """Check internal consistency."""
        # Check for contradictions, undefined terms, etc.
        pass


class MathematicalReasoningPlugin(IRPPlugin):
    """
    IRP plugin for mathematical and symbolic reasoning.

    Operates independently of language plugin but can translate
    to/from verbal descriptions via cross-modal VAE.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.symbolic_engine = SymbolicEngine()
        self.geometric_visualizer = GeometricVisualizer()
        self.proof_search = ProofSearchEngine()

    def can_handle(self, observation: Any) -> float:
        """
        Determine if observation is mathematical in nature.

        Returns confidence [0, 1] that this plugin should handle it.
        """
        # Check for mathematical symbols, equations, geometric shapes
        if self._contains_equations(observation):
            return 0.9
        if self._contains_geometric_elements(observation):
            return 0.8
        if self._contains_logical_statements(observation):
            return 0.7
        return 0.1

    def init_state(self, observation: Any) -> IRPState:
        """
        Initialize symbolic representation.

        Args:
            observation: Problem statement (text, equation, diagram)

        Returns:
            Initial state with symbolic form and solution space
        """
        # Parse observation into symbolic representation
        symbolic_form = self._parse_to_symbols(observation)

        # Initialize geometric visualization if applicable
        geometric_viz = None
        if self._has_geometric_aspects(symbolic_form):
            geometric_viz = self.geometric_visualizer.init(symbolic_form)

        # Set up solution space
        solution_space = self._initialize_solution_space(symbolic_form)

        return IRPState(
            data={
                'symbolic_form': symbolic_form,
                'geometric_viz': geometric_viz,
                'solution_space': solution_space,
                'derivation_steps': [],
                'current_energy': float('inf'),
                'stuck_count': 0
            },
            metadata={
                'plugin': 'mathematical_reasoning',
                'domain': symbolic_form.domain,
                'complexity': self._estimate_complexity(symbolic_form)
            }
        )

    def step(self, state: IRPState) -> IRPState:
        """
        One step of symbolic manipulation.

        Applies mathematical transformations to refine solution.
        """
        symbolic_form = state.data['symbolic_form']
        solution_space = state.data['solution_space']

        # Select most promising transformation
        transformation = self._select_transformation(
            symbolic_form,
            solution_space,
            state.data['derivation_steps']
        )

        # Apply transformation
        new_symbolic_form = symbolic_form.apply_transformation(transformation)

        # Update geometric visualization if exists
        new_geometric_viz = state.data['geometric_viz']
        if new_geometric_viz:
            new_geometric_viz = self.geometric_visualizer.update(
                new_geometric_viz,
                new_symbolic_form
            )

        # Update solution space
        new_solution_space = self._refine_solution_space(
            solution_space,
            new_symbolic_form
        )

        # Record derivation step
        new_steps = state.data['derivation_steps'] + [transformation]

        # Detect if stuck
        stuck_count = state.data['stuck_count']
        if self._is_stuck(new_steps):
            stuck_count += 1
        else:
            stuck_count = 0

        # Compute new energy
        new_energy = self.energy(IRPState(
            data={
                'symbolic_form': new_symbolic_form,
                'geometric_viz': new_geometric_viz,
                'solution_space': new_solution_space,
                'derivation_steps': new_steps,
                'current_energy': 0,  # Will be set below
                'stuck_count': stuck_count
            },
            metadata=state.metadata
        ))

        return IRPState(
            data={
                'symbolic_form': new_symbolic_form,
                'geometric_viz': new_geometric_viz,
                'solution_space': new_solution_space,
                'derivation_steps': new_steps,
                'current_energy': new_energy,
                'stuck_count': stuck_count
            },
            metadata=state.metadata
        )

    def energy(self, state: IRPState) -> float:
        """
        Coherence of symbolic representation.

        Lower energy = more coherent, simpler, more elegant solution.
        """
        symbolic_form = state.data['symbolic_form']

        # Inconsistency measure
        inconsistency = symbolic_form.evaluate_coherence()

        # Complexity penalty (simpler is better)
        complexity = self._measure_complexity(symbolic_form)

        # Elegance bonus (recognize beautiful solutions)
        elegance = self._measure_elegance(symbolic_form)

        # Solution completeness (how close to answer)
        completeness = self._measure_completeness(
            symbolic_form,
            state.data['solution_space']
        )

        return (
            10.0 * inconsistency +      # Inconsistency is bad
            0.1 * complexity -           # Complexity is slightly bad
            0.5 * elegance +             # Elegance is good
            5.0 * (1.0 - completeness)   # Incompleteness is bad
        )

    def should_halt(self, energy_history: List[float]) -> bool:
        """
        Stop when solution found or genuinely stuck.
        """
        if super().should_halt(energy_history):
            return True

        # Also halt if solution is complete
        if energy_history[-1] < 0.1:  # Very low energy = good solution
            return True

        # Or if stuck for too long
        if len(energy_history) > 100:  # Max iterations
            return True

        return False

    def _select_transformation(self, symbolic_form, solution_space, history):
        """
        Choose next symbolic manipulation.

        Uses heuristics like:
        - Simplification rules
        - Substitution patterns
        - Proof strategies
        - Analogy to known solutions
        """
        candidates = self.symbolic_engine.get_applicable_rules(symbolic_form)

        # Score each candidate
        scores = []
        for rule in candidates:
            score = self._score_transformation(rule, symbolic_form, history)
            scores.append((score, rule))

        # Select best (with some randomness for exploration)
        scores.sort(reverse=True)
        return scores[0][1]  # Return best rule

    def _measure_elegance(self, symbolic_form) -> float:
        """
        Recognize mathematical beauty.

        Elegant solutions have:
        - Symmetry
        - Simplicity
        - Unexpected connections
        - Universal patterns
        """
        elegance = 0.0

        # Symmetry detection
        if self._has_symmetry(symbolic_form):
            elegance += 0.3

        # Uses fundamental constants (π, e, etc.)
        if self._uses_fundamental_constants(symbolic_form):
            elegance += 0.2

        # Connects different domains
        if self._crosses_domains(symbolic_form):
            elegance += 0.3

        # Generalizes known results
        if self._generalizes_known(symbolic_form):
            elegance += 0.2

        return elegance


class SymbolicEngine:
    """Symbolic manipulation engine."""

    def get_applicable_rules(self, symbolic_form):
        """Return mathematical rules applicable to current form."""
        # Implement rule library
        pass


class GeometricVisualizer:
    """Geometric visualization engine."""

    def init(self, symbolic_form):
        """Create initial geometric representation."""
        pass

    def update(self, current_viz, new_symbolic_form):
        """Update visualization based on new symbolic form."""
        pass


class ProofSearchEngine:
    """Automated theorem proving."""

    def search_proof(self, goal, axioms):
        """Search for proof of goal from axioms."""
        pass
```

#### 1.2 Mathematical VAE

**File:** `/HRM/sage/compression/mathematical_vae.py`

```python
"""
Mathematical VAE - Compression for symbolic representations.

Maps symbolic expressions to latent space for:
- Cross-modal translation
- Similarity search
- Pattern recognition
"""

import torch
import torch.nn as nn


class MathematicalVAE(nn.Module):
    """
    VAE for mathematical/symbolic content.

    Latent dimension: 128D (between vision 64D and language 256D)
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: symbolic → latent
        self.encoder = SymbolicEncoder(latent_dim)

        # Decoder: latent → symbolic
        self.decoder = SymbolicDecoder(latent_dim)

    def encode(self, symbolic_form):
        """Encode symbolic expression to latent."""
        # Convert symbolic form to tensor representation
        tensor_rep = self._symbolize_to_tensor(symbolic_form)

        # Encode to latent space
        mu, log_var = self.encoder(tensor_rep)
        return mu, log_var

    def decode(self, latent):
        """Decode latent to symbolic expression."""
        tensor_rep = self.decoder(latent)

        # Convert tensor back to symbolic form
        symbolic_form = self._tensor_to_symbolic(tensor_rep)
        return symbolic_form

    def _symbolize_to_tensor(self, symbolic_form):
        """Convert symbolic representation to tensor."""
        # Use tree encoding or sequence encoding
        pass

    def _tensor_to_symbolic(self, tensor_rep):
        """Convert tensor back to symbolic form."""
        # Parse tensor to tree or sequence
        pass


class SymbolicEncoder(nn.Module):
    """Encoder for symbolic expressions."""

    def __init__(self, latent_dim):
        super().__init__()
        # Tree-LSTM or Transformer encoder
        self.tree_encoder = TreeLSTM(hidden_dim=256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, tensor_rep):
        hidden = self.tree_encoder(tensor_rep)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var


class SymbolicDecoder(nn.Module):
    """Decoder for symbolic expressions."""

    def __init__(self, latent_dim):
        super().__init__()
        # Tree generation or sequence generation
        self.fc = nn.Linear(latent_dim, 256)
        self.tree_decoder = TreeLSTMDecoder(hidden_dim=256)

    def forward(self, latent):
        hidden = self.fc(latent)
        tensor_rep = self.tree_decoder(hidden)
        return tensor_rep
```

#### 1.3 Integration

**File:** `/HRM/sage/core/sage.py` (modifications)

```python
def _initialize_plugins(self):
    """Load all IRP plugins including mathematical reasoning."""
    self.plugins = {
        'vision': VisionPlugin(self.config['vision']),
        'audio': AudioPlugin(self.config['audio']),
        'language': LanguagePlugin(self.config['language']),
        'mathematical': MathematicalReasoningPlugin(  # NEW
            self.config['mathematical']
        ),
        'control': ControlPlugin(self.config['control']),
        # ... other plugins
    }
```

### Success Criteria

- [ ] Mathematical reasoning plugin can solve basic algebra problems
- [ ] Geometric visualizer can represent 2D shapes
- [ ] Mathematical VAE achieves >0.85 trust on test set
- [ ] Cross-modal translation: equation ↔ verbal explanation works
- [ ] Plugin integrates into main SAGE loop

### Dependencies

- sympy for symbolic computation
- matplotlib for geometric visualization
- Tree-LSTM or Transformer for symbolic encoding

---

## Enhancement 2: Hierarchical Memory System

**Status:** Not implemented
**Priority:** P1
**Effort:** 2-3 weeks

### Motivation

Michaud's generalization hierarchy (first-level → patterns → concepts) enables:
- Transfer learning across contexts
- Few-shot adaptation from generalizations
- Compression with trust via shared hierarchies

### Proposed Implementation

#### 2.1 Hierarchical Memory

**File:** `/HRM/sage/memory/hierarchical_memory.py`

```python
"""
Hierarchical Memory System

Three-level structure matching Michaud's framework:
1. Experiences (first-level): Specific observations
2. Patterns (generalizations): Clusters from experiences
3. Concepts (abstract): Relationships between patterns
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Experience:
    """First-level memory: specific observation."""
    id: str
    latent: torch.Tensor      # VAE encoding
    observation: Any          # Raw data (if salient enough to store)
    salience: float           # SNARC score
    timestamp: datetime
    plugin: str               # Which plugin processed this
    energy: float             # Final IRP energy


@dataclass
class Pattern:
    """Second-level memory: generalization over experiences."""
    id: str
    centroid: torch.Tensor    # Cluster center in latent space
    instances: List[str]      # Experience IDs in this pattern
    stability: float          # How stable is this cluster
    formation_count: int      # How many experiences contributed
    last_updated: datetime


@dataclass
class Concept:
    """Third-level memory: abstract relationship between patterns."""
    id: str
    definition: str           # Verbal description
    patterns: List[str]       # Pattern IDs in this concept
    verbal_description: str   # Language plugin generated
    examples: List[str]       # Experience IDs exemplifying concept


class HierarchicalMemory:
    """
    Three-level memory system for generalization and transfer.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Three memory levels
        self.experiences: Dict[str, Experience] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.concepts: Dict[str, Concept] = {}

        # Indexing for fast retrieval
        self.latent_index = LatentSpaceIndex()  # kNN in latent space

        # Parameters
        self.experience_threshold = config.get('experience_salience_threshold', 0.6)
        self.pattern_min_size = config.get('pattern_min_cluster_size', 3)
        self.pattern_max_distance = config.get('pattern_max_distance', 0.5)
        self.concept_min_patterns = config.get('concept_min_patterns', 2)

    def store_experience(self, observation: Any, salience: float,
                        latent: torch.Tensor, plugin: str, energy: float):
        """
        Store first-level experience if sufficiently salient.

        Args:
            observation: Raw observation data
            salience: SNARC salience score
            latent: VAE latent encoding
            plugin: Which plugin processed this
            energy: Final IRP energy after refinement
        """
        if salience < self.experience_threshold:
            return  # Not salient enough to remember

        # Create experience
        exp_id = self._generate_id('exp')
        experience = Experience(
            id=exp_id,
            latent=latent,
            observation=observation if salience > 0.8 else None,  # Save space
            salience=salience,
            timestamp=datetime.now(),
            plugin=plugin,
            energy=energy
        )

        # Store
        self.experiences[exp_id] = experience
        self.latent_index.add(exp_id, latent)

        # Check if this generalizes to existing pattern
        self._update_patterns(exp_id, latent, plugin)

    def _update_patterns(self, exp_id: str, latent: torch.Tensor, plugin: str):
        """
        Check if new experience fits existing pattern or forms new one.
        """
        # Find nearby experiences in latent space
        nearby_exp_ids = self.latent_index.search(latent, k=10)

        # Filter to same plugin (patterns are plugin-specific)
        nearby_same_plugin = [
            eid for eid in nearby_exp_ids
            if self.experiences[eid].plugin == plugin
        ]

        if len(nearby_same_plugin) >= self.pattern_min_size:
            # Check if they're close enough to form pattern
            nearby_latents = [
                self.experiences[eid].latent
                for eid in nearby_same_plugin
            ]

            # Compute cluster coherence
            centroid = torch.mean(torch.stack(nearby_latents + [latent]), dim=0)
            max_dist = max([
                torch.dist(lat, centroid).item()
                for lat in nearby_latents + [latent]
            ])

            if max_dist < self.pattern_max_distance:
                # Form or update pattern
                self._create_or_update_pattern(
                    nearby_same_plugin + [exp_id],
                    centroid,
                    plugin
                )

    def _create_or_update_pattern(self, exp_ids: List[str],
                                 centroid: torch.Tensor, plugin: str):
        """
        Create new pattern or update existing one.
        """
        # Check if any exp_ids already belong to pattern
        existing_pattern_id = None
        for pid, pattern in self.patterns.items():
            if any(eid in pattern.instances for eid in exp_ids):
                existing_pattern_id = pid
                break

        if existing_pattern_id:
            # Update existing pattern
            pattern = self.patterns[existing_pattern_id]
            pattern.instances = list(set(pattern.instances + exp_ids))
            pattern.centroid = centroid
            pattern.formation_count += 1
            pattern.last_updated = datetime.now()
            pattern.stability = self._compute_stability(pattern.instances)
        else:
            # Create new pattern
            pattern_id = self._generate_id('pattern')
            pattern = Pattern(
                id=pattern_id,
                centroid=centroid,
                instances=exp_ids,
                stability=self._compute_stability(exp_ids),
                formation_count=len(exp_ids),
                last_updated=datetime.now()
            )
            self.patterns[pattern_id] = pattern

        # Check if this pattern forms concept
        self._update_concepts(pattern.id)

    def _update_concepts(self, pattern_id: str):
        """
        Check if patterns form abstract concepts.
        """
        # Find related patterns (similar centroids but from different contexts)
        pattern = self.patterns[pattern_id]
        related_patterns = self._find_related_patterns(pattern)

        if len(related_patterns) >= self.concept_min_patterns:
            # Check if they share common structure
            if self._has_common_structure(related_patterns):
                # Form or update concept
                self._create_or_update_concept(related_patterns)

    def _find_related_patterns(self, pattern: Pattern) -> List[str]:
        """Find patterns with similar structure but different contexts."""
        related = []

        for pid, other_pattern in self.patterns.items():
            if pid == pattern.id:
                continue

            # Check similarity in latent space
            dist = torch.dist(pattern.centroid, other_pattern.centroid).item()

            if dist < self.pattern_max_distance * 1.5:  # Slightly more tolerant
                related.append(pid)

        return related

    def _has_common_structure(self, pattern_ids: List[str]) -> bool:
        """Check if patterns share common abstract structure."""
        # Analyze experiences in each pattern for structural similarities
        # This is domain-specific and would need sophisticated implementation
        # For now, simple heuristic: similar salience profiles
        patterns = [self.patterns[pid] for pid in pattern_ids]

        # Compute salience profiles
        profiles = []
        for pattern in patterns:
            salinces = [self.experiences[eid].salience
                       for eid in pattern.instances]
            profiles.append(np.mean(salinces))

        # Check variance
        variance = np.var(profiles)
        return variance < 0.1  # Similar salience patterns

    def _create_or_update_concept(self, pattern_ids: List[str]):
        """
        Create or update abstract concept from patterns.
        """
        # Check if concept already exists
        existing_concept_id = None
        for cid, concept in self.concepts.items():
            if any(pid in concept.patterns for pid in pattern_ids):
                existing_concept_id = cid
                break

        # Generate verbal description using language plugin
        verbal_desc = self._generate_concept_description(pattern_ids)

        if existing_concept_id:
            # Update existing concept
            concept = self.concepts[existing_concept_id]
            concept.patterns = list(set(concept.patterns + pattern_ids))
            concept.verbal_description = verbal_desc
        else:
            # Create new concept
            concept_id = self._generate_id('concept')

            # Collect example experiences
            example_ids = []
            for pid in pattern_ids:
                pattern = self.patterns[pid]
                # Pick most salient experience from pattern
                best_exp = max(pattern.instances,
                             key=lambda eid: self.experiences[eid].salience)
                example_ids.append(best_exp)

            concept = Concept(
                id=concept_id,
                definition=verbal_desc,
                patterns=pattern_ids,
                verbal_description=verbal_desc,
                examples=example_ids
            )
            self.concepts[concept_id] = concept

    def _generate_concept_description(self, pattern_ids: List[str]) -> str:
        """
        Generate verbal description of concept.

        Would use language plugin in full implementation.
        """
        # Placeholder - would invoke language plugin
        return f"Concept formed from {len(pattern_ids)} patterns"

    def _compute_stability(self, exp_ids: List[str]) -> float:
        """Compute how stable/coherent a cluster is."""
        if len(exp_ids) < 2:
            return 0.0

        latents = [self.experiences[eid].latent for eid in exp_ids]
        latent_stack = torch.stack(latents)

        # Compute variance
        variance = torch.var(latent_stack, dim=0).mean().item()

        # Lower variance = higher stability
        stability = 1.0 / (1.0 + variance)
        return stability

    def recall_similar(self, observation_latent: torch.Tensor,
                      k: int = 5) -> List[Experience]:
        """
        Recall similar past experiences.

        Args:
            observation_latent: Latent encoding of current observation
            k: Number of similar experiences to return

        Returns:
            List of similar experiences, ordered by similarity
        """
        exp_ids = self.latent_index.search(observation_latent, k=k)
        return [self.experiences[eid] for eid in exp_ids]

    def get_pattern_for_experience(self, exp_id: str) -> Optional[Pattern]:
        """Get pattern that this experience belongs to."""
        for pattern in self.patterns.values():
            if exp_id in pattern.instances:
                return pattern
        return None

    def get_concept_for_pattern(self, pattern_id: str) -> Optional[Concept]:
        """Get concept that this pattern belongs to."""
        for concept in self.concepts.values():
            if pattern_id in concept.patterns:
                return concept
        return None

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"


class LatentSpaceIndex:
    """kNN index for fast latent space retrieval."""

    def __init__(self):
        self.index = {}  # id -> latent
        # In production, use FAISS or similar

    def add(self, id: str, latent: torch.Tensor):
        """Add vector to index."""
        self.index[id] = latent

    def search(self, query: torch.Tensor, k: int = 5) -> List[str]:
        """Find k nearest neighbors."""
        if not self.index:
            return []

        # Compute distances
        distances = []
        for id, latent in self.index.items():
            dist = torch.dist(query, latent).item()
            distances.append((dist, id))

        # Sort and return top k
        distances.sort()
        return [id for _, id in distances[:k]]
```

#### 2.2 Integration into SAGE

**File:** `/HRM/sage/core/sage.py` (modifications)

```python
def __init__(self, config: Dict[str, Any]):
    # ... existing initialization ...

    # Initialize hierarchical memory
    self.hierarchical_memory = HierarchicalMemory(
        config.get('hierarchical_memory', {})
    )

def _process_observation(self, observation):
    """Process observation with hierarchical memory integration."""

    # Encode to latent space
    latent = self.vae.encode(observation)

    # Recall similar past experiences
    similar_past = self.hierarchical_memory.recall_similar(latent)

    # Compute salience (with context from memory)
    salience = self.snarc.compute(observation, context=similar_past)

    # ... rest of processing ...

    # After refinement, store if salient
    self.hierarchical_memory.store_experience(
        observation=observation,
        salience=salience,
        latent=latent,
        plugin=selected_plugin.name,
        energy=final_energy
    )
```

### Success Criteria

- [ ] Experiences automatically cluster into patterns
- [ ] Patterns form concepts with verbal descriptions
- [ ] Transfer learning: pattern from one context applies to another
- [ ] Few-shot: New experience matched to pattern with < 3 examples
- [ ] Hierarchical retrieval faster than flat memory

---

## Enhancement 3: Attention-Driven Resource Allocation

**Status:** Partial (ATP exists, not fully dynamic)
**Priority:** P1
**Effort:** 1 week

### Motivation

Michaud's attention mechanism = localized wave of excitation + inhibition of everything else. SAGE needs dynamic ATP allocation based on metabolic state and salience.

### Proposed Implementation

**File:** `/HRM/sage/core/attention_manager.py`

```python
"""
Attention Manager - Dynamic Resource Allocation

Implements Michaud's attention mechanism:
- Focus: Narrow attention, most resources to single target
- Wake: Distributed attention across moderate targets
- Rest: Minimal attention, consolidation focus
- Dream: Random exploration
"""

from enum import Enum
from typing import Dict, List
import numpy as np


class MetabolicState(Enum):
    """Metabolic states for SAGE."""
    WAKE = "wake"       # Normal processing
    FOCUS = "focus"     # Intense concentration
    REST = "rest"       # Low activity, consolidation
    DREAM = "dream"     # Random exploration
    CRISIS = "crisis"   # Emergency response


class AttentionManager:
    """
    Manages dynamic ATP allocation based on attention focus.
    """

    def __init__(self, total_atp: float = 100.0):
        self.total_atp = total_atp
        self.current_state = MetabolicState.WAKE
        self.attention_history = []

    def allocate_attention(self, salience_map: Dict[str, float],
                          metabolic_state: MetabolicState) -> Dict[str, float]:
        """
        Compute ATP allocation across targets.

        Args:
            salience_map: {target_id: salience_score}
            metabolic_state: Current metabolic state

        Returns:
            {target_id: atp_allocation}
        """
        self.current_state = metabolic_state

        if metabolic_state == MetabolicState.FOCUS:
            return self._focus_allocation(salience_map)

        elif metabolic_state == MetabolicState.WAKE:
            return self._wake_allocation(salience_map)

        elif metabolic_state == MetabolicState.REST:
            return self._rest_allocation()

        elif metabolic_state == MetabolicState.DREAM:
            return self._dream_allocation(salience_map)

        elif metabolic_state == MetabolicState.CRISIS:
            return self._crisis_allocation(salience_map)

    def _focus_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        FOCUS: 80% to highest salience, 15% to second, 5% background.

        Mimics Michaud's "localized wave of overexcitement" with
        inhibition of everything else.
        """
        if not salience_map:
            return {}

        # Sort by salience
        sorted_targets = sorted(salience_map.items(),
                              key=lambda x: x[1],
                              reverse=True)

        allocation = {}

        if len(sorted_targets) >= 1:
            allocation[sorted_targets[0][0]] = 0.8 * self.total_atp

        if len(sorted_targets) >= 2:
            allocation[sorted_targets[1][0]] = 0.15 * self.total_atp

        # Distribute remaining 5% among rest
        if len(sorted_targets) > 2:
            remaining_atp = 0.05 * self.total_atp
            per_target = remaining_atp / (len(sorted_targets) - 2)
            for target_id, _ in sorted_targets[2:]:
                allocation[target_id] = per_target

        return allocation

    def _wake_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        WAKE: Distribute proportional to salience with some spreading.

        Mimics normal waking awareness - attentive but not hyper-focused.
        """
        if not salience_map:
            return {}

        # Normalize salience to sum to 1
        total_salience = sum(salience_map.values())
        if total_salience == 0:
            # Equal distribution if all zero salience
            per_target = self.total_atp / len(salience_map)
            return {tid: per_target for tid in salience_map.keys()}

        # Distribute proportionally with some flattening (spread_factor)
        spread_factor = 0.5  # 0 = pure proportional, 1 = equal
        allocation = {}

        for target_id, salience in salience_map.items():
            proportional = (salience / total_salience) * self.total_atp
            equal = self.total_atp / len(salience_map)
            allocation[target_id] = (
                (1 - spread_factor) * proportional +
                spread_factor * equal
            )

        return allocation

    def _rest_allocation(self) -> Dict[str, float]:
        """
        REST: Minimal monitoring, most resources to consolidation.

        Mimics sleep state - minimal sensory processing, memory
        consolidation active.
        """
        return {
            'memory_consolidation': 0.7 * self.total_atp,
            'minimal_monitoring': 0.3 * self.total_atp
        }

    def _dream_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        DREAM: Random exploration, pattern discovery.

        Mimics REM sleep - random activation for novel connections.
        """
        if not salience_map:
            return {}

        # Random allocation with some bias toward recent high-salience
        allocation = {}
        remaining_atp = self.total_atp

        for target_id in salience_map.keys():
            # Random between 0 and remaining
            amount = np.random.uniform(0, remaining_atp / 2)
            allocation[target_id] = amount
            remaining_atp -= amount

        # Distribute remaining
        if remaining_atp > 0:
            per_target = remaining_atp / len(salience_map)
            for target_id in allocation.keys():
                allocation[target_id] += per_target

        return allocation

    def _crisis_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        CRISIS: All resources to highest priority threat.

        Mimics fight-or-flight - maximum resources to survival.
        """
        if not salience_map:
            return {}

        # ALL ATP to highest salience target
        highest_salience_target = max(salience_map.items(),
                                     key=lambda x: x[1])[0]

        return {highest_salience_target: self.total_atp}

    def transition_state(self, salience_map: Dict[str, float],
                        time_in_state: float) -> MetabolicState:
        """
        Determine metabolic state transitions.

        Args:
            salience_map: Current salience scores
            time_in_state: How long in current state (seconds)

        Returns:
            New metabolic state
        """
        max_salience = max(salience_map.values()) if salience_map else 0.0

        current = self.current_state

        # Crisis: Any very high salience triggers
        if max_salience > 0.95:
            return MetabolicState.CRISIS

        # From Crisis -> Focus (after threat passes)
        if current == MetabolicState.CRISIS and max_salience < 0.8:
            return MetabolicState.FOCUS

        # From Focus -> Wake (when task completes or salience drops)
        if current == MetabolicState.FOCUS:
            if max_salience < 0.6 or time_in_state > 300:  # 5 minutes
                return MetabolicState.WAKE

        # From Wake -> Focus (high salience sustained)
        if current == MetabolicState.WAKE and max_salience > 0.8:
            return MetabolicState.FOCUS

        # From Wake -> Rest (low salience sustained)
        if current == MetabolicState.WAKE:
            if max_salience < 0.3 and time_in_state > 60:
                return MetabolicState.REST

        # From Rest -> Dream (random trigger)
        if current == MetabolicState.REST:
            if time_in_state > 120 and np.random.random() < 0.1:
                return MetabolicState.DREAM

        # From Dream -> Wake (any moderate salience)
        if current == MetabolicState.DREAM and max_salience > 0.4:
            return MetabolicState.WAKE

        # From Dream -> Rest (after exploration period)
        if current == MetabolicState.DREAM and time_in_state > 60:
            return MetabolicState.REST

        return current  # No transition
```

### Success Criteria

- [ ] Metabolic state transitions work as specified
- [ ] ATP allocation matches state (80% focus in FOCUS mode, etc.)
- [ ] State transitions respond appropriately to salience changes
- [ ] CRISIS mode activates for high-threat situations
- [ ] DREAM mode enables exploration

---

## Enhancement 4: Emotional Energy Functions

**Status:** Not implemented
**Priority:** P2
**Effort:** 1 week

### Motivation

Michaud: Emotions provide evaluation function that drives behavior. Energy functions can encode same drives computationally.

### Proposed Implementation

**File:** `/HRM/sage/irp/emotional_energy.py`

```python
"""
Emotional Energy Functions - Intrinsic Motivation

Emotions = evolved energy functions that add drives beyond
task-specific metrics.

Implements:
- Curiosity (novelty-seeking)
- Mastery (competence-building)
- Completion (goal achievement)
- Frustration (stuck avoidance)
"""

import torch
from typing import Any, List
from .base import IRPPlugin, IRPState


class EmotionalEnergyMixin:
    """
    Mixin for IRP plugins to add emotional drives.

    Usage:
        class MyPlugin(EmotionalEnergyMixin, IRPPlugin):
            def energy(self, state):
                return (
                    self.task_energy(state) +
                    self.emotional_energy(state)
                )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Emotional drive weights (tunable per plugin)
        self.curiosity_weight = kwargs.get('curiosity_weight', 0.3)
        self.mastery_weight = kwargs.get('mastery_weight', 0.2)
        self.completion_weight = kwargs.get('completion_weight', 0.4)
        self.frustration_weight = kwargs.get('frustration_weight', 0.5)

        # State for tracking
        self.competence_history = []
        self.progress_history = []

    def emotional_energy(self, state: IRPState) -> float:
        """
        Combined emotional drives.

        Lower energy = more motivated to pursue this state.
        """
        return (
            -self.curiosity_weight * self._curiosity_drive(state) +
            -self.mastery_weight * self._mastery_drive(state) +
            -self.completion_weight * self._completion_drive(state) +
            self.frustration_weight * self._frustration_cost(state)
        )

    def _curiosity_drive(self, state: IRPState) -> float:
        """
        Seek novelty and surprise.

        Returns: [0, 1] where 1 = very novel/surprising
        """
        # Novelty: How different from past experiences
        novelty = self._measure_novelty(state)

        # Surprise: How unexpected given predictions
        surprise = self._measure_surprise(state)

        return novelty * surprise

    def _measure_novelty(self, state: IRPState) -> float:
        """
        Measure novelty of current state.

        Check similarity to past experiences in memory.
        """
        if not hasattr(self, 'memory') or self.memory is None:
            return 0.5  # Unknown novelty

        # Get state representation
        state_repr = self._get_state_representation(state)

        # Query memory for similar past states
        similar = self.memory.recall_similar(state_repr, k=5)

        if not similar:
            return 1.0  # Completely novel

        # Compute average similarity
        similarities = [
            self._compute_similarity(state_repr, s.latent)
            for s in similar
        ]
        avg_similarity = sum(similarities) / len(similarities)

        # Novelty = 1 - similarity
        return 1.0 - avg_similarity

    def _measure_surprise(self, state: IRPState) -> float:
        """
        Measure surprise - prediction error.

        How different is actual observation from prediction?
        """
        if not hasattr(state.data, 'prediction'):
            return 0.5  # No prediction available

        if not hasattr(state.data, 'observation'):
            return 0.5  # No observation available

        # Compute prediction error
        error = torch.dist(
            state.data['prediction'],
            state.data['observation']
        ).item()

        # Normalize to [0, 1]
        surprise = min(1.0, error / 10.0)  # Assuming max error ~ 10
        return surprise

    def _mastery_drive(self, state: IRPState) -> float:
        """
        Seek improvement and skill development.

        Returns: [0, 1] where 1 = high learning potential
        """
        # Current competence level
        competence = self._estimate_competence(state)
        self.competence_history.append(competence)

        # Growth potential (am I getting better?)
        if len(self.competence_history) < 2:
            growth = 0.5
        else:
            recent_growth = (
                self.competence_history[-1] -
                self.competence_history[-10:]
            )
            growth = max(0.0, min(1.0, recent_growth + 0.5))

        return competence * growth

    def _estimate_competence(self, state: IRPState) -> float:
        """
        Estimate current competence level.

        Based on:
        - How quickly energy decreases
        - How low final energy gets
        - Success rate on similar tasks
        """
        if not hasattr(state.data, 'energy_history'):
            return 0.5

        energy_history = state.data['energy_history']
        if len(energy_history) < 2:
            return 0.5

        # Fast convergence = high competence
        convergence_speed = (
            energy_history[0] - energy_history[-1]
        ) / len(energy_history)

        # Low final energy = high competence
        final_energy = energy_history[-1]

        competence = (
            0.5 * min(1.0, convergence_speed / 0.1) +  # Normalize
            0.5 * (1.0 - min(1.0, final_energy))
        )

        return competence

    def _completion_drive(self, state: IRPState) -> float:
        """
        Seek goal achievement.

        Returns: [0, 1] where 1 = very close to completion
        """
        # Progress toward goal
        progress = self._estimate_progress(state)
        self.progress_history.append(progress)

        # Proximity to goal
        proximity = self._estimate_proximity(state)

        # Extra motivation when close to finishing
        completion_bonus = 0.0
        if proximity > 0.8:
            completion_bonus = 0.3

        return progress * proximity + completion_bonus

    def _estimate_progress(self, state: IRPState) -> float:
        """
        Estimate progress toward goal.

        Based on energy reduction over time.
        """
        if not hasattr(state.data, 'energy_history'):
            return 0.0

        energy_history = state.data['energy_history']
        if len(energy_history) < 2:
            return 0.0

        # Progress = relative energy reduction
        initial_energy = energy_history[0]
        current_energy = energy_history[-1]

        if initial_energy == 0:
            return 1.0

        progress = (initial_energy - current_energy) / initial_energy
        return max(0.0, min(1.0, progress))

    def _estimate_proximity(self, state: IRPState) -> float:
        """
        Estimate proximity to goal.

        Based on current energy level.
        """
        current_energy = state.data.get('current_energy', float('inf'))

        # Proximity = inverse of energy
        if current_energy == 0:
            return 1.0

        proximity = 1.0 / (1.0 + current_energy)
        return proximity

    def _frustration_cost(self, state: IRPState) -> float:
        """
        Penalize being stuck.

        Returns: [0, 1] where 1 = very frustrated (stuck)
        """
        if not hasattr(state.data, 'energy_history'):
            return 0.0

        energy_history = state.data['energy_history']
        if len(energy_history) < 5:
            return 0.0

        # Check if stuck (energy not decreasing)
        recent_energies = energy_history[-5:]
        energy_variance = torch.var(torch.tensor(recent_energies)).item()

        # Low variance + high energy = stuck
        if energy_variance < 0.01 and recent_energies[-1] > 1.0:
            return 1.0

        # Gradual frustration buildup
        frustration = 1.0 - min(1.0, energy_variance / 0.1)
        return frustration

    def _get_state_representation(self, state: IRPState) -> torch.Tensor:
        """Get latent representation of state for similarity comparison."""
        # Plugin-specific implementation
        raise NotImplementedError

    def _compute_similarity(self, repr1: torch.Tensor,
                          repr2: torch.Tensor) -> float:
        """Compute similarity between two representations."""
        dist = torch.dist(repr1, repr2).item()
        similarity = 1.0 / (1.0 + dist)
        return similarity
```

### Success Criteria

- [ ] Plugins with emotional energy explore more novel situations
- [ ] Mastery drive leads to skill improvement over time
- [ ] Completion drive increases focus as goals approach
- [ ] Frustration detection triggers alternative strategies
- [ ] Intrinsic motivation visible in behavior logs

---

(Continued in next part due to length...)
