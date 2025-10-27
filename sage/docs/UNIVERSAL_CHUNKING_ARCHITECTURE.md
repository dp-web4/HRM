# Universal Chunking Architecture - Phase 4 Design

**Date**: October 26, 2025
**Status**: 🎨 Design Phase
**Scope**: Cross-modal chunking for all IRP modalities + Trust + SNARC integration

---

## Executive Summary

This document extends breath-based prosodic chunking to create a **universal chunking framework** that applies biological and cognitive constraints across all SAGE modalities - sensors, effectors, and cognitive processes.

### Core Insight: Universal Chunking Principle

**All intelligent processing operates on chunks bounded by biological and cognitive constraints:**

- **Duration**: 2-4 seconds (attention window)
- **Size**: 5-9 discrete items (working memory capacity)
- **Boundaries**: Natural transitions in the modality's "prosody"
- **Purpose**: Reduce cognitive load, enable compositional reasoning

**Key Discovery**: Every modality has "prosody" - rhythmic structure that defines natural boundaries:
- **Speech prosody**: Intonation, pauses, breath groups
- **Visual prosody**: Saccades, fixations, attention shifts
- **Motor prosody**: Velocity zero-crossings, sub-movements, kinematic phrases
- **Memory prosody**: Episode boundaries, temporal transitions
- **Semantic prosody**: Clause boundaries, topic shifts

---

## Table of Contents

1. [Universal Chunking Framework](#framework)
2. [Modality-Specific Implementations](#modalities)
   - [Vision (Sensor)](#vision)
   - [Audio (Sensor)](#audio)
   - [Memory (Cognitive)](#memory)
   - [Language (Cognitive)](#language)
   - [Control (Cognitive)](#control)
   - [Motion (Effector)](#motion) **← NEW: Motion Prosody**
3. [Trust Integration](#trust)
4. [SNARC Integration](#snarc)
5. [Implementation Architecture](#architecture)
6. [Biological Parallels](#biology)
7. [Research Foundations](#research)
8. [Implementation Roadmap](#roadmap)

---

<a name="framework"></a>
## 1. Universal Chunking Framework

### Base Chunking Interface

All modalities implement a common chunking interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple

@dataclass
class UniversalChunk:
    """
    Universal chunk representation across all modalities.

    Every chunk carries:
    - Content (modality-specific data)
    - Boundaries (prosodic markers)
    - Trust (quality metrics)
    - Salience (SNARC scores)
    """
    # Core attributes
    content: Any  # Modality-specific data
    modality: str  # "vision", "audio", "memory", "language", "control", "motion"
    timestamp: float
    duration: float

    # Boundary metadata (prosodic structure)
    boundary_type: str  # "major", "minor", "micro", "forced"
    chunk_size: int  # Number of discrete items
    continuation: bool  # Whether more chunks expected

    # Trust metrics (quality + confidence)
    trust_score: float  # 0.0-1.0, from ChunkTrustMetrics
    trust_breakdown: Optional['ChunkTrustMetrics'] = None

    # SNARC salience (5D + prosodic)
    salience_score: float  # 0.0-1.0, combined salience
    salience_breakdown: Optional['ChunkSalienceMetrics'] = None

    # Modality-specific metadata
    prosody: Optional[Any] = None  # e.g., ProsodicMetadata, VisualProsody, MotorProsody
    metadata: dict = None

@dataclass
class ChunkTrustMetrics:
    """
    Trust metrics for chunk quality assessment.

    Trust measures: How reliable is this chunk?
    - Confidence: Detection/generation confidence
    - Consistency: Internal coherence
    - Completeness: Boundary detection quality
    - Fidelity: Compression/reconstruction error
    """
    confidence: float  # 0.0-1.0, detection/generation confidence
    consistency: float  # 0.0-1.0, internal coherence
    completeness: float  # 0.0-1.0, boundary quality (natural vs forced)
    fidelity: float  # 0.0-1.0, compression/reconstruction quality

    def compute_overall_trust(self) -> float:
        """Weighted combination of trust dimensions"""
        weights = {
            'confidence': 0.35,
            'consistency': 0.25,
            'completeness': 0.25,
            'fidelity': 0.15
        }

        return (
            self.confidence * weights['confidence'] +
            self.consistency * weights['consistency'] +
            self.completeness * weights['completeness'] +
            self.fidelity * weights['fidelity']
        )

@dataclass
class ChunkSalienceMetrics:
    """
    SNARC-based salience metrics for chunk importance.

    Extended SNARC 5D:
    - Surprise: Unexpected content (semantic, visual, motor)
    - Novelty: New vs. familiar patterns
    - Arousal: Intensity/energy (pitch, motion, color)
    - Reward: Goal-relevance, value
    - Conflict: Ambiguity, contradiction

    Plus prosodic salience (boundary importance)
    """
    surprise: float  # 0.0-1.0, prediction error
    novelty: float  # 0.0-1.0, familiarity inverse
    arousal: float  # 0.0-1.0, intensity/energy
    reward: float  # 0.0-1.0, goal-relevance
    conflict: float  # 0.0-1.0, ambiguity
    prosodic: float  # 0.0-1.0, boundary importance

    def compute_overall_salience(self) -> float:
        """Weighted combination of salience dimensions"""
        # Base SNARC (equal weights)
        snarc_base = (
            self.surprise + self.novelty + self.arousal +
            self.reward + self.conflict
        ) / 5.0

        # Modulated by prosodic salience (boundary importance)
        # Prosodic salience amplifies base salience (multiplicative)
        return snarc_base * (0.7 + self.prosodic * 0.3)


class UniversalChunker(ABC):
    """
    Abstract base class for modality-specific chunkers.

    All chunkers must implement:
    - Boundary detection (prosodic structure)
    - Trust assessment (quality metrics)
    - Salience computation (SNARC + prosodic)
    """

    def __init__(
        self,
        modality: str,
        min_chunk_size: int,
        target_chunk_size: int,
        max_chunk_size: int,
        chunk_duration: Tuple[float, float]  # (min, max) seconds
    ):
        self.modality = modality
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_duration, self.max_duration = chunk_duration

    @abstractmethod
    def detect_boundary(self, buffer: Any, new_item: Any) -> Tuple[bool, str]:
        """
        Detect prosodic boundary in modality-specific stream.

        Returns: (is_boundary, boundary_type)
        """
        pass

    @abstractmethod
    def compute_trust(self, chunk_content: Any) -> ChunkTrustMetrics:
        """Compute trust metrics for chunk"""
        pass

    @abstractmethod
    def compute_salience(self, chunk_content: Any) -> ChunkSalienceMetrics:
        """Compute SNARC salience for chunk"""
        pass

    @abstractmethod
    def extract_prosody(self, chunk_content: Any) -> Any:
        """Extract modality-specific prosodic features"""
        pass

    def create_chunk(
        self,
        content: Any,
        boundary_type: str,
        chunk_size: int,
        duration: float,
        continuation: bool = True
    ) -> UniversalChunk:
        """
        Create universal chunk with full metadata.

        Computes trust, salience, and prosody automatically.
        """
        # Extract prosodic features
        prosody = self.extract_prosody(content)

        # Compute trust metrics
        trust_metrics = self.compute_trust(content)
        trust_score = trust_metrics.compute_overall_trust()

        # Compute salience metrics
        salience_metrics = self.compute_salience(content)
        salience_score = salience_metrics.compute_overall_salience()

        return UniversalChunk(
            content=content,
            modality=self.modality,
            timestamp=time.time(),
            duration=duration,
            boundary_type=boundary_type,
            chunk_size=chunk_size,
            continuation=continuation,
            trust_score=trust_score,
            trust_breakdown=trust_metrics,
            salience_score=salience_score,
            salience_breakdown=salience_metrics,
            prosody=prosody,
            metadata={}
        )
```

---

<a name="modalities"></a>
## 2. Modality-Specific Implementations

<a name="vision"></a>
### Vision Prosody: Attention-Based Chunking

**Biological Foundation**: Vision operates through saccades (rapid eye movements) and fixations (stable gaze). A "visual breath group" is 3-5 fixations forming a coherent attention episode.

**Visual Prosody Markers**:
- **Major boundaries**: Scene changes, object transitions
- **Minor boundaries**: Saccade endpoints, attention shifts
- **Micro boundaries**: Individual fixations
- **Forced boundaries**: Attention capacity overflow (>7 fixations)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class VisualProsody:
    """Prosodic features for visual attention"""
    # Boundary markers
    boundary_type: str  # "scene", "object", "fixation", "forced"
    fixation_count: int  # Number of fixations in chunk
    saccade_pattern: str  # "exploratory", "targeted", "return"

    # Salience features
    visual_surprise: float  # Unexpected visual patterns (edge density, color novelty)
    visual_arousal: float  # Motion energy, color saturation
    roi_salience: float  # Region of interest importance

    # Temporal dynamics
    gaze_duration: float  # Total fixation time
    saccade_velocity: float  # Average saccade speed
    return_rate: float  # Proportion of return saccades (re-visiting)

class VisionChunker(UniversalChunker):
    """
    Chunk visual attention into fixation-aligned episodes.

    Chunks: 3-5 fixations (target), 1-7 fixations (range)
    Duration: 1.0-2.5 seconds (avg fixation = 200-400ms)
    """

    def __init__(self):
        super().__init__(
            modality="vision",
            min_chunk_size=1,  # 1 fixation minimum
            target_chunk_size=4,  # 4 fixations target
            max_chunk_size=7,  # 7 fixations max (working memory)
            chunk_duration=(0.5, 3.0)  # 0.5-3.0 seconds
        )

        # Visual attention state
        self.current_roi = None
        self.fixation_buffer = []
        self.saccade_history = []

    def detect_boundary(
        self,
        fixation_buffer: List['Fixation'],
        new_fixation: 'Fixation'
    ) -> Tuple[bool, str]:
        """
        Detect visual attention boundaries.

        Boundaries occur at:
        1. Scene changes (global change detection)
        2. Object transitions (ROI changes)
        3. Attention capacity (>7 fixations)
        4. Temporal gaps (>1s between fixations)
        """
        if not fixation_buffer:
            return (False, None)

        fixation_count = len(fixation_buffer)

        # Priority 1: Scene change (major boundary)
        if self._is_scene_change(fixation_buffer, new_fixation):
            return (True, "scene")

        # Priority 2: Object transition (minor boundary)
        if fixation_count >= self.min_chunk_size:
            if self._is_object_transition(fixation_buffer, new_fixation):
                return (True, "object")

        # Priority 3: Attention capacity overflow (forced boundary)
        if fixation_count >= self.max_chunk_size:
            return (True, "forced")

        # Priority 4: Target chunk size with stable gaze
        if fixation_count >= self.target_chunk_size:
            if self._is_stable_gaze(fixation_buffer):
                return (True, "fixation")

        return (False, None)

    def _is_scene_change(self, buffer: List, new_fixation) -> bool:
        """Detect global scene change (optical flow, histogram difference)"""
        # Use frame difference, optical flow magnitude
        if not buffer:
            return False

        last_frame = buffer[-1].frame
        new_frame = new_fixation.frame

        # Simple pixel difference (real implementation uses optical flow)
        diff = np.abs(new_frame - last_frame).mean()
        return diff > 50.0  # Threshold for scene change

    def _is_object_transition(self, buffer: List, new_fixation) -> bool:
        """Detect ROI change (object-based attention shift)"""
        # Check if new fixation targets different object
        if not buffer:
            return False

        current_roi = buffer[0].roi  # ROI of first fixation
        new_roi = new_fixation.roi

        return current_roi != new_roi

    def _is_stable_gaze(self, buffer: List) -> bool:
        """Check if gaze has stabilized on target (low saccade variance)"""
        if len(buffer) < 2:
            return False

        # Calculate saccade distances
        saccades = []
        for i in range(1, len(buffer)):
            dist = np.linalg.norm(buffer[i].position - buffer[i-1].position)
            saccades.append(dist)

        # Stable if variance is low (cluster of fixations)
        return np.var(saccades) < 100.0  # pixels²

    def compute_trust(self, fixations: List) -> ChunkTrustMetrics:
        """
        Compute trust for visual chunk.

        Trust factors:
        - Confidence: Detection confidence (yolo/detectron scores)
        - Consistency: Optical flow smoothness
        - Completeness: Natural vs forced boundary
        - Fidelity: Feature reconstruction error
        """
        # Confidence: Average detection scores
        detection_scores = [f.detection_confidence for f in fixations if f.detection_confidence]
        confidence = np.mean(detection_scores) if detection_scores else 0.5

        # Consistency: Optical flow coherence (low variance = high consistency)
        flow_coherence = self._compute_flow_coherence(fixations)
        consistency = 1.0 - min(1.0, flow_coherence / 10.0)  # Normalize to 0-1

        # Completeness: Natural boundary quality
        # High if boundary is natural (scene/object), low if forced
        last_boundary = fixations[-1].boundary_type if fixations else "unknown"
        completeness = {
            'scene': 1.0,
            'object': 0.8,
            'fixation': 0.6,
            'forced': 0.3
        }.get(last_boundary, 0.5)

        # Fidelity: Feature extraction quality (VAE reconstruction error)
        # Placeholder - real implementation uses actual VAE
        fidelity = 0.8

        return ChunkTrustMetrics(
            confidence=confidence,
            consistency=consistency,
            completeness=completeness,
            fidelity=fidelity
        )

    def compute_salience(self, fixations: List) -> ChunkSalienceMetrics:
        """
        Compute SNARC salience for visual chunk.

        Visual SNARC:
        - Surprise: Edge density, color novelty (vs. scene prior)
        - Novelty: Object novelty (vs. memory)
        - Arousal: Motion energy, color saturation
        - Reward: Task-relevance (goal-directed attention)
        - Conflict: Ambiguity (detection uncertainty)
        - Prosodic: Boundary importance (scene > object > fixation)
        """
        # Surprise: Visual prediction error
        surprise = self._compute_visual_surprise(fixations)

        # Novelty: New objects vs. familiar
        novelty = self._compute_visual_novelty(fixations)

        # Arousal: Motion + color intensity
        motion_energy = np.mean([f.motion_magnitude for f in fixations if f.motion_magnitude])
        color_saturation = np.mean([f.color_saturation for f in fixations if f.color_saturation])
        arousal = (motion_energy + color_saturation) / 2.0

        # Reward: Goal relevance (if task-directed)
        reward = np.mean([f.task_relevance for f in fixations if f.task_relevance]) if any(f.task_relevance for f in fixations) else 0.5

        # Conflict: Detection ambiguity
        detection_variance = np.var([f.detection_confidence for f in fixations if f.detection_confidence])
        conflict = min(1.0, detection_variance * 10.0)  # High variance = high conflict

        # Prosodic: Boundary type importance
        boundary_type = fixations[-1].boundary_type if fixations else "unknown"
        prosodic = {
            'scene': 0.9,
            'object': 0.7,
            'fixation': 0.5,
            'forced': 0.3
        }.get(boundary_type, 0.5)

        return ChunkSalienceMetrics(
            surprise=surprise,
            novelty=novelty,
            arousal=arousal,
            reward=reward,
            conflict=conflict,
            prosodic=prosodic
        )

    def extract_prosody(self, fixations: List) -> VisualProsody:
        """Extract visual prosodic features"""
        boundary_type = fixations[-1].boundary_type if fixations else "unknown"
        fixation_count = len(fixations)

        # Saccade pattern analysis
        saccade_pattern = self._analyze_saccade_pattern(fixations)

        # Salience aggregation
        visual_surprise = self.compute_salience(fixations).surprise
        visual_arousal = self.compute_salience(fixations).arousal
        roi_salience = np.mean([f.roi_importance for f in fixations if hasattr(f, 'roi_importance')])

        # Temporal dynamics
        gaze_duration = sum(f.duration for f in fixations)
        saccade_velocity = np.mean([f.saccade_velocity for f in fixations if hasattr(f, 'saccade_velocity')])

        # Return rate (how many saccades return to previous ROI)
        return_saccades = sum(1 for i in range(1, len(fixations)) if fixations[i].roi == fixations[i-2].roi if i >= 2)
        return_rate = return_saccades / max(1, len(fixations) - 2)

        return VisualProsody(
            boundary_type=boundary_type,
            fixation_count=fixation_count,
            saccade_pattern=saccade_pattern,
            visual_surprise=visual_surprise,
            visual_arousal=visual_arousal,
            roi_salience=roi_salience,
            gaze_duration=gaze_duration,
            saccade_velocity=saccade_velocity,
            return_rate=return_rate
        )

    def _compute_flow_coherence(self, fixations: List) -> float:
        """Compute optical flow coherence (low = smooth, high = chaotic)"""
        # Placeholder - real implementation uses Lucas-Kanade or similar
        return 2.0

    def _compute_visual_surprise(self, fixations: List) -> float:
        """Compute visual prediction error"""
        # Use edge density, color histogram difference vs scene prior
        return 0.6  # Placeholder

    def _compute_visual_novelty(self, fixations: List) -> float:
        """Compute object novelty vs. memory"""
        # Check against object memory (seen before?)
        return 0.5  # Placeholder

    def _analyze_saccade_pattern(self, fixations: List) -> str:
        """Classify saccade pattern: exploratory, targeted, return"""
        if len(fixations) < 2:
            return "single"

        # Analyze saccade distances and directions
        saccade_lengths = []
        for i in range(1, len(fixations)):
            dist = np.linalg.norm(fixations[i].position - fixations[i-1].position)
            saccade_lengths.append(dist)

        avg_length = np.mean(saccade_lengths)

        # Heuristic classification
        if avg_length > 200:  # pixels
            return "exploratory"  # Long saccades = scanning
        elif avg_length < 50:
            return "targeted"  # Short saccades = focused attention
        else:
            return "mixed"
```

**Key Concepts**:
- **Visual "breath"**: 3-5 fixations forming coherent attention episode
- **Saccades as pauses**: Rapid movements between fixations (like breath pauses)
- **Prosodic hierarchy**: Scene > Object > Fixation (like IP > ip > word)
- **Trust from optical flow**: Smooth flow = high consistency trust
- **Salience from novelty**: New objects/scenes = high SNARC salience

---

<a name="motion"></a>
### Motion Prosody: Kinematic Chunking (NEW!)

**Biological Foundation**: Human movement is segmented into discrete motor primitives (reach, grasp, transport). "Kinematic breath groups" are movement phrases bounded by velocity/acceleration zero-crossings.

**Motion Prosody Markers**:
- **Major boundaries**: Goal completions (grasp object, place object)
- **Minor boundaries**: Sub-movements (reach → grasp → retract)
- **Micro boundaries**: Velocity zero-crossings (direction changes)
- **Forced boundaries**: Movement timeout or collision

```python
@dataclass
class MotorProsody:
    """Prosodic features for motor control"""
    # Boundary markers
    boundary_type: str  # "goal", "subgoal", "primitive", "forced"
    primitive_count: int  # Number of motor primitives
    movement_phrase: str  # "reach-grasp", "transport", "place", "retract"

    # Kinematic features
    peak_velocity: float  # Maximum velocity in chunk
    smoothness: float  # Movement smoothness (jerk metric)
    directness: float  # Path efficiency (straight vs curved)

    # Salience features
    motor_surprise: float  # Unexpected trajectory deviations
    motor_arousal: float  # Movement speed/force intensity
    goal_salience: float  # Goal importance

    # Temporal dynamics
    movement_duration: float
    acceleration_pattern: str  # "bell-curve", "multi-peak", "asymmetric"

class MotionChunker(UniversalChunker):
    """
    Chunk motor commands into kinematic phrases.

    Biological parallel: Motor cortex organizes movement into primitives
    bounded by velocity zero-crossings (Sosnik et al. 2004).

    Chunks: 1-3 primitives (target), 1-5 primitives (range)
    Duration: 0.5-2.0 seconds per primitive
    """

    def __init__(self):
        super().__init__(
            modality="motion",
            min_chunk_size=1,  # 1 primitive minimum
            target_chunk_size=2,  # 2 primitives target (reach-grasp)
            max_chunk_size=5,  # 5 primitives max
            chunk_duration=(0.5, 4.0)  # 0.5-4.0 seconds
        )

        # Motor state
        self.current_goal = None
        self.primitive_buffer = []
        self.velocity_history = []

    def detect_boundary(
        self,
        primitive_buffer: List['MotorPrimitive'],
        new_primitive: 'MotorPrimitive'
    ) -> Tuple[bool, str]:
        """
        Detect kinematic boundaries.

        Boundaries occur at:
        1. Goal completion (major boundary)
        2. Sub-goal transitions (minor boundary)
        3. Velocity zero-crossings (micro boundary)
        4. Primitive capacity overflow (forced boundary)
        """
        if not primitive_buffer:
            return (False, None)

        primitive_count = len(primitive_buffer)

        # Priority 1: Goal completion (major boundary)
        if self._is_goal_complete(primitive_buffer, new_primitive):
            return (True, "goal")

        # Priority 2: Sub-goal transition (minor boundary)
        if primitive_count >= self.min_chunk_size:
            if self._is_subgoal_transition(primitive_buffer, new_primitive):
                return (True, "subgoal")

        # Priority 3: Primitive capacity overflow (forced boundary)
        if primitive_count >= self.max_chunk_size:
            return (True, "forced")

        # Priority 4: Target chunk size with smooth completion
        if primitive_count >= self.target_chunk_size:
            if self._is_smooth_completion(primitive_buffer):
                return (True, "primitive")

        return (False, None)

    def _is_goal_complete(self, buffer: List, new_primitive) -> bool:
        """Detect goal completion (e.g., object grasped, target reached)"""
        # Check if new primitive starts new goal
        if not buffer:
            return False

        current_goal = buffer[0].goal_id
        new_goal = new_primitive.goal_id

        return current_goal != new_goal

    def _is_subgoal_transition(self, buffer: List, new_primitive) -> bool:
        """Detect sub-goal transition (reach → grasp → transport)"""
        # Check if new primitive is different type
        if not buffer:
            return False

        current_type = buffer[-1].primitive_type  # "reach", "grasp", "transport"
        new_type = new_primitive.primitive_type

        return current_type != new_type

    def _is_smooth_completion(self, buffer: List) -> bool:
        """Check if movement has smoothly completed (velocity → 0)"""
        if not buffer:
            return False

        # Check if last primitive ended with near-zero velocity
        last_velocity = buffer[-1].end_velocity
        return np.linalg.norm(last_velocity) < 0.05  # m/s

    def compute_trust(self, primitives: List) -> ChunkTrustMetrics:
        """
        Compute trust for motor chunk.

        Trust factors:
        - Confidence: Motion planning confidence
        - Consistency: Movement smoothness (jerk metric)
        - Completeness: Goal vs forced boundary
        - Fidelity: Target accuracy (endpoint error)
        """
        # Confidence: Planning success probability
        confidence = np.mean([p.planning_confidence for p in primitives])

        # Consistency: Movement smoothness (inverse of jerk)
        jerk_values = [p.jerk for p in primitives]
        avg_jerk = np.mean(jerk_values)
        consistency = 1.0 - min(1.0, avg_jerk / 100.0)  # Normalize to 0-1

        # Completeness: Boundary quality
        last_boundary = primitives[-1].boundary_type if primitives else "unknown"
        completeness = {
            'goal': 1.0,
            'subgoal': 0.8,
            'primitive': 0.6,
            'forced': 0.3
        }.get(last_boundary, 0.5)

        # Fidelity: Target accuracy (distance to goal)
        if primitives and hasattr(primitives[-1], 'endpoint_error'):
            endpoint_error = primitives[-1].endpoint_error  # meters
            fidelity = 1.0 - min(1.0, endpoint_error / 0.1)  # <10cm = high fidelity
        else:
            fidelity = 0.7

        return ChunkTrustMetrics(
            confidence=confidence,
            consistency=consistency,
            completeness=completeness,
            fidelity=fidelity
        )

    def compute_salience(self, primitives: List) -> ChunkSalienceMetrics:
        """
        Compute SNARC salience for motor chunk.

        Motor SNARC:
        - Surprise: Trajectory deviations (vs. minimum jerk)
        - Novelty: Novel motor patterns (vs. motor memory)
        - Arousal: Movement speed/force intensity
        - Reward: Goal value/importance
        - Conflict: Planning ambiguity (multiple solutions)
        - Prosodic: Boundary importance (goal > subgoal > primitive)
        """
        # Surprise: Trajectory prediction error
        surprise = self._compute_motor_surprise(primitives)

        # Novelty: Motor pattern novelty
        novelty = self._compute_motor_novelty(primitives)

        # Arousal: Movement intensity (velocity + force)
        velocities = [p.peak_velocity for p in primitives]
        forces = [p.peak_force for p in primitives if hasattr(p, 'peak_force')]
        arousal = (np.mean(velocities) / 2.0 + np.mean(forces) / 50.0) / 2.0 if forces else np.mean(velocities) / 2.0

        # Reward: Goal value
        reward = np.mean([p.goal_value for p in primitives if hasattr(p, 'goal_value')]) if any(hasattr(p, 'goal_value') for p in primitives) else 0.5

        # Conflict: Planning ambiguity
        conflict = np.mean([p.planning_entropy for p in primitives if hasattr(p, 'planning_entropy')]) if any(hasattr(p, 'planning_entropy') for p in primitives) else 0.3

        # Prosodic: Boundary type importance
        boundary_type = primitives[-1].boundary_type if primitives else "unknown"
        prosodic = {
            'goal': 0.9,
            'subgoal': 0.7,
            'primitive': 0.5,
            'forced': 0.3
        }.get(boundary_type, 0.5)

        return ChunkSalienceMetrics(
            surprise=surprise,
            novelty=novelty,
            arousal=arousal,
            reward=reward,
            conflict=conflict,
            prosodic=prosodic
        )

    def extract_prosody(self, primitives: List) -> MotorProsody:
        """Extract motor prosodic features"""
        boundary_type = primitives[-1].boundary_type if primitives else "unknown"
        primitive_count = len(primitives)

        # Movement phrase classification
        movement_phrase = self._classify_movement_phrase(primitives)

        # Kinematic features
        peak_velocity = max(p.peak_velocity for p in primitives)
        smoothness = 1.0 - np.mean([p.jerk for p in primitives]) / 100.0
        directness = self._compute_path_efficiency(primitives)

        # Salience
        motor_surprise = self.compute_salience(primitives).surprise
        motor_arousal = self.compute_salience(primitives).arousal
        goal_salience = self.compute_salience(primitives).reward

        # Temporal dynamics
        movement_duration = sum(p.duration for p in primitives)
        acceleration_pattern = self._classify_acceleration_pattern(primitives)

        return MotorProsody(
            boundary_type=boundary_type,
            primitive_count=primitive_count,
            movement_phrase=movement_phrase,
            peak_velocity=peak_velocity,
            smoothness=smoothness,
            directness=directness,
            motor_surprise=motor_surprise,
            motor_arousal=motor_arousal,
            goal_salience=goal_salience,
            movement_duration=movement_duration,
            acceleration_pattern=acceleration_pattern
        )

    def _compute_motor_surprise(self, primitives: List) -> float:
        """Compute trajectory prediction error (vs. minimum jerk)"""
        # Compare actual trajectory to minimum jerk prediction
        return 0.4  # Placeholder

    def _compute_motor_novelty(self, primitives: List) -> float:
        """Compute motor pattern novelty (vs. motor repertoire)"""
        # Check against learned motor primitives
        return 0.5  # Placeholder

    def _compute_path_efficiency(self, primitives: List) -> float:
        """Compute path directness (straight line / actual path)"""
        # Ratio of straight-line distance to actual path length
        return 0.85  # Placeholder

    def _classify_movement_phrase(self, primitives: List) -> str:
        """Classify movement phrase type"""
        types = [p.primitive_type for p in primitives]

        if 'reach' in types and 'grasp' in types:
            return "reach-grasp"
        elif 'transport' in types:
            return "transport"
        elif 'place' in types:
            return "place"
        elif 'retract' in types:
            return "retract"
        else:
            return "unknown"

    def _classify_acceleration_pattern(self, primitives: List) -> str:
        """Classify acceleration profile"""
        # Analyze acceleration time-series
        # Bell-curve = smooth, multi-peak = corrective, asymmetric = ballistic
        return "bell-curve"  # Placeholder
```

**Key Concepts - Motion Prosody**:
- **Kinematic "breath"**: Movement phrase (reach-grasp-transport) as unit
- **Velocity zero-crossings**: Like pause between breaths
- **Bell-curve acceleration**: Natural movement "intonation"
- **Smoothness as trust**: Low jerk = high consistency
- **Goal value as salience**: Important movements get high salience

**Biological Parallel**:
- Motor cortex segments movement at velocity zero-crossings (Sosnik 2004)
- Movement primitives compose into phrases (Mussa-Ivaldi 1999)
- Minimum jerk trajectory = "natural prosody" of movement
- **This is exactly analogous to speech prosody!**

---

<a name="audio"></a>
### Audio Prosody: Speech Chunking (Implemented)

Already implemented in Phase 1-3. Extended here with trust + SNARC:

```python
class AudioChunker(UniversalChunker):
    """Audio chunking with prosodic boundaries (already implemented)"""

    def __init__(self):
        super().__init__(
            modality="audio",
            min_chunk_size=5,  # 5 words minimum
            target_chunk_size=12,  # 12 words target (breath group)
            max_chunk_size=18,  # 18 words max
            chunk_duration=(1.5, 4.5)  # 1.5-4.5 seconds
        )

        # Use existing prosody chunker
        from cognitive.prosody_chunker import ProsodyAwareChunker
        self.prosody_chunker = ProsodyAwareChunker()

    def compute_trust(self, text: str, audio: np.ndarray) -> ChunkTrustMetrics:
        """
        Compute trust for audio chunk.

        Trust factors:
        - Confidence: Whisper transcription confidence
        - Consistency: Prosodic coherence (F0 continuity)
        - Completeness: Prosodic boundary quality (IP > ip > forced)
        - Fidelity: Audio quality (SNR, no clipping)
        """
        # Confidence: Transcription confidence (from Whisper)
        confidence = 0.85  # Placeholder - get from Whisper logprobs

        # Consistency: Prosodic coherence
        # High if F0 contour is smooth, low if erratic
        consistency = self._compute_prosodic_coherence(audio)

        # Completeness: Boundary quality
        boundary_type = "IP"  # From prosody chunker
        completeness = {
            'IP': 1.0,
            'ip': 0.8,
            'NATURAL': 0.6,
            'BREATH': 0.4
        }.get(boundary_type, 0.5)

        # Fidelity: Audio SNR
        snr = self._compute_snr(audio)
        fidelity = min(1.0, snr / 20.0)  # 20dB = perfect

        return ChunkTrustMetrics(
            confidence=confidence,
            consistency=consistency,
            completeness=completeness,
            fidelity=fidelity
        )

    def compute_salience(self, text: str, prosody: 'ProsodicMetadata') -> ChunkSalienceMetrics:
        """
        Compute SNARC salience for audio chunk.

        Audio SNARC:
        - Surprise: Unexpected words (semantic surprise)
        - Novelty: New topic vs. familiar
        - Arousal: Prosodic arousal (pitch + energy variance)
        - Reward: Relevance to current goals
        - Conflict: Semantic ambiguity
        - Prosodic: Boundary importance
        """
        # Surprise: Semantic surprise (from language model perplexity)
        surprise = 0.5  # Placeholder - compute from LM

        # Novelty: Topic novelty (vs. conversation history)
        novelty = 0.5  # Placeholder - compute from memory

        # Arousal: Prosodic arousal (already computed in prosody)
        arousal = prosody.prosodic_arousal if prosody else 0.5

        # Reward: Goal relevance
        reward = 0.5  # Placeholder - needs task context

        # Conflict: Semantic ambiguity
        conflict = 0.3  # Placeholder - multiple interpretations

        # Prosodic: Boundary importance
        prosodic = prosody.compute_prosodic_salience() if prosody else 0.5

        return ChunkSalienceMetrics(
            surprise=surprise,
            novelty=novelty,
            arousal=arousal,
            reward=reward,
            conflict=conflict,
            prosodic=prosodic
        )
```

---

<a name="memory"></a>
### Memory Prosody: Episode Chunking

**Biological Foundation**: Memory operates through episodes bounded by temporal transitions and context shifts. Working memory capacity (7±2 items) defines chunk size.

```python
@dataclass
class MemoryProsody:
    """Prosodic features for memory episodes"""
    # Boundary markers
    boundary_type: str  # "context_shift", "temporal_gap", "capacity", "forced"
    item_count: int  # Number of memory items
    episode_type: str  # "conversation", "task", "event", "learned_pattern"

    # Temporal features
    episode_duration: float  # Total duration
    temporal_coherence: float  # Timestamp clustering
    decay_rate: float  # Memory decay slope

    # Salience features
    memory_surprise: float  # Unexpected recall
    memory_novelty: float  # New vs. familiar
    consolidation_strength: float  # Long-term potential

class MemoryChunker(UniversalChunker):
    """
    Chunk memory into working-memory-aligned episodes.

    Chunks: 5-7 items (target), 3-9 items (range)
    Duration: Variable (depends on event spacing)
    """

    def __init__(self):
        super().__init__(
            modality="memory",
            min_chunk_size=3,  # 3 items minimum
            target_chunk_size=6,  # 6 items target (working memory)
            max_chunk_size=9,  # 9 items max (7±2 rule)
            chunk_duration=(5.0, 300.0)  # 5s - 5min episodes
        )

    def detect_boundary(
        self,
        memory_buffer: List['MemoryItem'],
        new_memory: 'MemoryItem'
    ) -> Tuple[bool, str]:
        """
        Detect episodic boundaries.

        Boundaries occur at:
        1. Context shifts (topic/task change)
        2. Temporal gaps (>30s silence)
        3. Working memory capacity (>9 items)
        4. Semantic discontinuity
        """
        if not memory_buffer:
            return (False, None)

        item_count = len(memory_buffer)

        # Priority 1: Context shift (major boundary)
        if self._is_context_shift(memory_buffer, new_memory):
            return (True, "context_shift")

        # Priority 2: Temporal gap (minor boundary)
        if item_count >= self.min_chunk_size:
            if self._is_temporal_gap(memory_buffer, new_memory):
                return (True, "temporal_gap")

        # Priority 3: Working memory overflow (forced boundary)
        if item_count >= self.max_chunk_size:
            return (True, "capacity")

        # Priority 4: Target size with semantic coherence
        if item_count >= self.target_chunk_size:
            if self._is_semantically_coherent(memory_buffer):
                return (True, "natural")

        return (False, None)

    def _is_context_shift(self, buffer: List, new_memory) -> bool:
        """Detect context/topic shift"""
        # Compare semantic embedding of new memory to buffer
        # Placeholder - real implementation uses embeddings
        return False

    def _is_temporal_gap(self, buffer: List, new_memory) -> bool:
        """Detect temporal gap (>30s)"""
        last_timestamp = buffer[-1].timestamp
        new_timestamp = new_memory.timestamp
        return (new_timestamp - last_timestamp) > 30.0

    def _is_semantically_coherent(self, buffer: List) -> bool:
        """Check if buffer forms coherent episode"""
        # Placeholder - check embedding clustering
        return True
```

---

<a name="language"></a>
### Language Prosody: Clause Chunking

**For language generation** (already uses prosodic chunker for input speech):

```python
class LanguageChunker(UniversalChunker):
    """
    Chunk language generation into clause-aligned units.

    Reuses prosodic chunker but applies to generated text.
    """

    def __init__(self):
        super().__init__(
            modality="language",
            min_chunk_size=5,  # 5 words minimum
            target_chunk_size=12,  # 12 words target (clause)
            max_chunk_size=18,  # 18 words max
            chunk_duration=(1.5, 4.5)  # Speaking duration if synthesized
        )

        from cognitive.prosody_chunker import ProsodyAwareChunker
        self.prosody_chunker = ProsodyAwareChunker()

    def compute_trust(self, generated_text: str) -> ChunkTrustMetrics:
        """
        Trust for generated language.

        - Confidence: Generation probability (softmax max)
        - Consistency: Perplexity (low = coherent)
        - Completeness: Prosodic boundary quality
        - Fidelity: Semantic preservation (if translating/paraphrasing)
        """
        # Placeholder implementation
        return ChunkTrustMetrics(
            confidence=0.8,
            consistency=0.85,
            completeness=0.9,
            fidelity=0.8
        )
```

---

<a name="control"></a>
### Control Prosody: Goal Hierarchy Chunking

```python
class ControlChunker(UniversalChunker):
    """
    Chunk planning/control into goal-aligned action sequences.

    Chunks: 2-4 actions (target), 1-6 actions (range)
    Duration: Variable (depends on action execution time)
    """

    def __init__(self):
        super().__init__(
            modality="control",
            min_chunk_size=1,  # 1 action minimum
            target_chunk_size=3,  # 3 actions target
            max_chunk_size=6,  # 6 actions max
            chunk_duration=(1.0, 30.0)  # 1-30 seconds
        )

    # Similar structure to MotionChunker but for abstract planning
```

---

<a name="trust"></a>
## 3. Trust Integration

### Trust as Chunking Quality Metric

Trust measures **how reliable** each chunk is across four dimensions:

**1. Confidence**: Detection/generation confidence
- Vision: Object detection scores (YOLO, Detectron)
- Audio: Transcription confidence (Whisper logprobs)
- Motion: Planning success probability
- Memory: Retrieval confidence
- Language: Generation probability

**2. Consistency**: Internal coherence
- Vision: Optical flow smoothness
- Audio: Prosodic coherence (F0 continuity)
- Motion: Movement smoothness (jerk metric)
- Memory: Semantic clustering
- Language: Perplexity

**3. Completeness**: Boundary quality
- Natural boundaries (scene change, sentence end, goal completion) = HIGH
- Forced boundaries (overflow, timeout) = LOW
- Measures whether chunk ended naturally or was cut off

**4. Fidelity**: Compression/reconstruction quality
- Vision: VAE reconstruction error
- Audio: SNR, audio quality
- Motion: Target accuracy (endpoint error)
- Memory: Recall accuracy
- Language: Semantic preservation

### Trust-Weighted Attention Allocation

SAGE uses trust scores to allocate ATP (attention budget):

```python
def allocate_attention(chunks: List[UniversalChunk], total_atp: float) -> dict:
    """
    Allocate ATP budget based on trust and salience.

    High-trust, high-salience chunks get more attention.
    Low-trust chunks trigger verification/re-processing.
    """
    allocations = {}

    # Compute priority scores (trust × salience)
    priorities = {}
    for chunk in chunks:
        priority = chunk.trust_score * chunk.salience_score
        priorities[chunk] = priority

    # Normalize to ATP budget
    total_priority = sum(priorities.values())
    for chunk, priority in priorities.items():
        atp_allocation = (priority / total_priority) * total_atp
        allocations[chunk] = atp_allocation

        # Flag low-trust chunks for verification
        if chunk.trust_score < 0.5:
            allocations[chunk]['needs_verification'] = True

    return allocations
```

### Trust-Based Eviction Policy

Low-trust chunks are evicted first from buffers:

```python
def evict_lowest_trust_chunk(buffer: List[UniversalChunk]) -> UniversalChunk:
    """Evict chunk with lowest combined trust × salience score"""
    return min(buffer, key=lambda c: c.trust_score * c.salience_score)
```

---

<a name="snarc"></a>
## 4. SNARC Integration

### Extended SNARC for Cross-Modal Salience

Original SNARC (5D):
- **S**urprise: Prediction error
- **N**ovelty: New vs. familiar
- **A**rousal: Intensity/energy
- **R**eward: Goal-relevance
- **C**onflict: Ambiguity

**Extended with Prosodic Salience** (6D):
- **Prosodic**: Boundary importance (modality-specific)

### Modality-Specific SNARC

Each modality computes SNARC using domain-specific features:

| Modality | Surprise | Novelty | Arousal | Reward | Conflict | Prosodic |
|----------|----------|---------|---------|--------|----------|----------|
| **Vision** | Edge density, color novelty | New objects | Motion, saturation | Task relevance | Detection uncertainty | Scene > Object > Fixation |
| **Audio** | Semantic surprise (LM) | Topic novelty | Pitch + energy variance | Goal relevance | Semantic ambiguity | IP > ip > word |
| **Motion** | Trajectory deviation | Motor pattern novelty | Velocity + force | Goal value | Planning entropy | Goal > Subgoal > Primitive |
| **Memory** | Recall surprise | Episodic novelty | Emotional intensity | Context relevance | Retrieval conflict | Context shift > Temporal gap |
| **Language** | Perplexity | Topic shift | Semantic intensity | Goal relevance | Ambiguity | Sentence > Clause > Phrase |
| **Control** | Plan deviation | Novel strategy | Urgency | Goal value | Plan conflict | Goal > Subgoal > Action |

### Cross-Modal Salience Fusion

Combine salience across modalities for holistic attention:

```python
def fuse_cross_modal_salience(chunks: List[UniversalChunk]) -> float:
    """
    Fuse salience across modalities.

    When multiple modalities provide evidence for same event:
    - Vision sees novel object (novelty=0.9)
    - Audio hears exclamation (arousal=0.8)
    - Language generates surprised response (surprise=0.85)

    → Fused salience is amplified (multiplicative fusion)
    """
    # Group chunks by timestamp (co-occurring events)
    temporal_window = 0.5  # seconds
    groups = group_by_time(chunks, temporal_window)

    fused_salience = {}
    for timestamp, modal_chunks in groups.items():
        # Modality-weighted fusion
        modality_weights = {
            'vision': 0.25,
            'audio': 0.25,
            'language': 0.20,
            'motion': 0.15,
            'memory': 0.10,
            'control': 0.05
        }

        # Weighted average of saliences
        total_salience = 0.0
        for chunk in modal_chunks:
            weight = modality_weights.get(chunk.modality, 0.1)
            total_salience += chunk.salience_score * weight

        # Amplify if multiple modalities agree (cross-modal coherence bonus)
        if len(modal_chunks) > 1:
            coherence_bonus = min(0.3, len(modal_chunks) * 0.1)
            total_salience *= (1.0 + coherence_bonus)

        fused_salience[timestamp] = min(1.0, total_salience)

    return fused_salience
```

---

<a name="architecture"></a>
## 5. Implementation Architecture

### Unified Chunking Pipeline

```python
class UnifiedChunkingPipeline:
    """
    Central chunking coordinator for all SAGE modalities.

    Maintains per-modality chunkers and orchestrates cross-modal fusion.
    """

    def __init__(self):
        # Initialize modality-specific chunkers
        self.chunkers = {
            'vision': VisionChunker(),
            'audio': AudioChunker(),
            'motion': MotionChunker(),
            'memory': MemoryChunker(),
            'language': LanguageChunker(),
            'control': ControlChunker()
        }

        # Cross-modal fusion
        self.chunk_buffer = []  # Mixed-modality chunk buffer
        self.fusion_window = 0.5  # seconds

    def process_stream(
        self,
        modality: str,
        stream_item: Any
    ) -> Optional[UniversalChunk]:
        """
        Process incoming stream item through modality-specific chunker.

        Returns chunk if boundary detected, None otherwise.
        """
        chunker = self.chunkers.get(modality)
        if not chunker:
            raise ValueError(f"Unknown modality: {modality}")

        # Delegate to modality-specific chunker
        chunk = chunker.process_item(stream_item)

        if chunk:
            # Add to cross-modal buffer
            self.chunk_buffer.append(chunk)

            # Trigger cross-modal fusion
            self._fuse_recent_chunks()

        return chunk

    def _fuse_recent_chunks(self):
        """Fuse chunks from different modalities within temporal window"""
        now = time.time()
        recent_chunks = [
            c for c in self.chunk_buffer
            if (now - c.timestamp) <= self.fusion_window
        ]

        if len(recent_chunks) > 1:
            # Cross-modal salience fusion
            fused_salience = fuse_cross_modal_salience(recent_chunks)

            # Update chunk saliences
            for chunk in recent_chunks:
                chunk.salience_score = max(
                    chunk.salience_score,
                    fused_salience.get(chunk.timestamp, chunk.salience_score)
                )
```

### Integration with SAGE Core

```python
# In SAGE cycle
pipeline = UnifiedChunkingPipeline()

def sage_cycle():
    # Process vision stream
    vision_frame = vision_sensor.poll()
    if vision_frame:
        vision_chunk = pipeline.process_stream('vision', vision_frame)
        if vision_chunk:
            # High salience? Allocate more attention
            if vision_chunk.salience_score > 0.7:
                allocate_attention(vision_chunk, atp_budget * 0.3)

    # Process audio stream
    audio_segment = audio_sensor.poll()
    if audio_segment:
        audio_chunk = pipeline.process_stream('audio', audio_segment)
        if audio_chunk:
            # Low trust? Verify with re-transcription
            if audio_chunk.trust_score < 0.5:
                verify_transcription(audio_chunk)

    # Process motion commands
    motion_primitive = motion_planner.poll()
    if motion_primitive:
        motion_chunk = pipeline.process_stream('motion', motion_primitive)
        if motion_chunk:
            # Execute motor command with trust-weighted confidence
            execute_motion(motion_chunk, confidence=motion_chunk.trust_score)
```

---

<a name="biology"></a>
## 6. Biological Parallels

### Universal Chunking in Biology

Every modality in biological systems exhibits chunking:

| Modality | Biological Chunking | Neural Correlate | Temporal Scale |
|----------|---------------------|------------------|----------------|
| **Vision** | Saccade-fixation cycles | Superior colliculus, FEF | 200-500ms fixations |
| **Speech** | Breath groups | Respiratory control, prosody network | 2-4s exhalations |
| **Motion** | Motor primitives | Motor cortex, cerebellum | 0.5-2s movements |
| **Memory** | Episodes | Hippocampus, temporal context | Variable (sec-min) |
| **Language** | Clauses | Broca's area, syntax network | 8-15 words |
| **Planning** | Sub-goals | Prefrontal cortex, goal hierarchy | Variable (sec-hours) |

### Hierarchical Structure

All modalities share hierarchical organization:

```
Modality      | Micro        | Minor         | Major           |
--------------|--------------|---------------|-----------------|
Vision        | Fixation     | Object        | Scene           |
Audio         | Syllable     | Clause (ip)   | Sentence (IP)   |
Motion        | Velocity-0   | Primitive     | Goal            |
Memory        | Item         | Episode       | Context         |
Language      | Word         | Clause        | Sentence        |
Control       | Action       | Sub-goal      | Goal            |
```

**Universal Pattern**: Micro → Minor → Major boundaries

### Convergence of Constraints

All modalities converge on similar chunk parameters:

- **Duration**: 2-4 seconds (attention window, breath group, movement phrase)
- **Size**: 5-9 items (working memory, fixations, motor primitives)
- **Boundaries**: Natural transitions (prosodic, kinematic, episodic)

**This is not coincidence** - it's the **optimal operating point** for biological intelligence given:
- Attention capacity (2-4s window)
- Working memory (7±2 chunks)
- Metabolic efficiency (chunk processing costs)
- Sensorimotor delays (reaction time ~200ms)

---

<a name="research"></a>
## 7. Research Foundations

### Key Papers by Modality

**Vision**:
- Yarbus (1967) - "Eye Movements and Vision" - Fixation-saccade structure
- Land & Hayhoe (2001) - "In what ways do eye movements contribute to everyday activities?" - Task-driven attention chunks
- Henderson (2003) - "Human gaze control during real-world scene perception" - Scene segmentation

**Speech**:
- Lieberman (1966) - "Intonation, Perception, and Language" - Breath groups
- PMC2945274 - "Breath Group Analysis" - 10-15 word chunks, 2-4s duration
- Nespor & Vogel (1986) - "Prosodic Phonology" - Prosodic hierarchy

**Motion**:
- Flash & Hogan (1985) - "The coordination of arm movements: an experimentally confirmed mathematical model" - Minimum jerk trajectory
- Sosnik et al. (2004) - "The segmentation of movement: a proposed functional role for velocity zero-crossings" - **Motion prosody**
- Mussa-Ivaldi & Solla (2004) - "Motor primitives" - Compositional movement

**Memory**:
- Miller (1956) - "The magical number seven, plus or minus two" - Working memory capacity
- Tulving (1972) - "Episodic and semantic memory" - Episode boundaries
- Ezzyat & Davachi (2011) - "What constitutes an episode in episodic memory?" - Temporal chunking

**Language**:
- Hawkins (2014) - "Cross-Linguistic Variation and Efficiency" - Clause as processing unit
- Gibson (1998) - "Linguistic complexity" - Memory costs of long-distance dependencies

**Cross-Modal**:
- Doupe & Kuhl (1999) - "Birdsong and human speech" - Hierarchical chunking across species
- Graziano (2006) - "The organization of behavioral repertoire in motor cortex" - Goal hierarchies

### Universal Principles Discovered

1. **Hierarchical Organization** (Micro → Minor → Major)
2. **Temporal Constraints** (2-4s attention window)
3. **Capacity Limits** (7±2 working memory chunks)
4. **Prosodic Structure** (Natural vs. forced boundaries)
5. **Compositional Assembly** (Chunks combine into larger structures)

---

<a name="roadmap"></a>
## 8. Implementation Roadmap

### Phase 4.1: Vision Chunking (3-5 days)

**Goal**: Implement attention-based visual chunking

**Tasks**:
1. Implement `VisionChunker` with fixation detection
2. Add optical flow analysis for boundary detection
3. Integrate with Vision IRP plugin
4. Compute visual trust (detection confidence, flow consistency)
5. Compute visual SNARC (surprise, novelty, arousal from motion/color)

**Expected Benefits**:
- Natural visual attention boundaries
- Trust-weighted object detection
- Salience-driven attention allocation

### Phase 4.2: Motion Chunking (3-5 days)

**Goal**: Implement kinematic chunking for motion control

**Tasks**:
1. Implement `MotionChunker` with velocity zero-crossing detection
2. Add movement primitive classification
3. Integrate with GR00T/Isaac integration
4. Compute motor trust (smoothness, accuracy)
5. Compute motor SNARC (trajectory surprise, goal value)

**Expected Benefits**:
- Natural movement segmentation
- Trust-weighted motion execution
- Smooth kinematic phrases

### Phase 4.3: Memory/Language/Control Chunking (2-3 days each)

Implement remaining chunkers following same pattern.

### Phase 4.4: Cross-Modal Fusion (2-3 days)

**Goal**: Implement unified chunking pipeline with cross-modal fusion

**Tasks**:
1. Create `UnifiedChunkingPipeline`
2. Implement temporal alignment (co-occurring chunks)
3. Implement cross-modal salience fusion
4. Test with multi-modal scenarios

### Phase 4.5: Trust + SNARC Integration (2-3 days)

**Goal**: Integrate trust and salience into SAGE core

**Tasks**:
1. ATP allocation based on trust × salience
2. Low-trust chunk verification
3. High-salience event logging
4. Cross-modal coherence metrics

---

## Conclusion

This universal chunking architecture extends biological and cognitive principles across all SAGE modalities. Every sensor, effector, and cognitive process chunks its data stream at natural prosodic boundaries, computes trust quality metrics, and contributes to cross-modal SNARC salience.

**Key Innovations**:

1. **Motion Prosody** - First formalization of kinematic chunking analogous to speech prosody
2. **Universal Trust Framework** - 4D trust metrics (confidence, consistency, completeness, fidelity) across all modalities
3. **Extended SNARC** - 6D salience (5D SNARC + prosodic) with cross-modal fusion
4. **Hierarchical Boundary Detection** - Micro/Minor/Major boundaries in all modalities
5. **Biological Validation** - Grounded in neuroscience, motor control, vision research

**Impact**:
- **Reduced cognitive load** - Natural chunking aligns with biological constraints
- **Improved trust** - Quality metrics enable adaptive processing
- **Enhanced salience** - Cross-modal fusion amplifies important events
- **Compositional reasoning** - Chunks as building blocks for higher-level cognition

This provides the **foundation for true cross-modal intelligence** - not just processing multiple modalities independently, but understanding how they chunk, trust, and attend to the world in unified, biologically-grounded ways.

---

**Next**: Implement Phase 4.1 (Vision Chunking) when ready to proceed.
