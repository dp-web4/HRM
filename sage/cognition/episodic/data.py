"""
Data structures for episodic memory.

Episode: a bound moment (what+where+when+outcome)
EpisodicCue: partial input for pattern-completion recall
RecallResult: a matched episode with similarity metadata
ConsolidationReport: summary of dream-state compression
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# Embedding dimensions for each component
DIM_STATE = 64
DIM_SNARC = 5
DIM_ACTION = 32
DIM_OUTCOME = 16
DIM_CONTEXT = 64
EMBEDDING_DIM = DIM_STATE + DIM_SNARC + DIM_ACTION + DIM_OUTCOME + DIM_CONTEXT  # 181


@dataclass
class Episode:
    """A bound episodic memory — one moment in the consciousness loop."""

    # Identity
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    session_id: str = ''
    cycle_id: int = 0

    # The binding (what+where+when)
    state_signature: dict = field(default_factory=dict)
    sensory_summary: dict = field(default_factory=dict)
    snarc_scores: dict = field(default_factory=dict)
    cognitive_stance: str = ''

    # What happened
    action_taken: Optional[str] = None
    action_args: dict = field(default_factory=dict)
    outcome: Optional[str] = None
    reward: float = 0.0
    success: Optional[bool] = None

    # Retrieval support
    embedding: Optional[np.ndarray] = None  # Set by EpisodicIndex.bind()
    tags: list = field(default_factory=list)
    access_count: int = 0
    last_access: float = 0.0

    def compute_embedding(self) -> np.ndarray:
        """Compute dense embedding from episode fields."""
        parts = []

        # State signature → deterministic hash projection
        parts.append(_hash_to_vector(self.state_signature, DIM_STATE))

        # SNARC scores → direct 5-dim vector
        snarc_keys = ['surprise', 'novelty', 'arousal', 'reward', 'conflict']
        snarc_vec = np.array([self.snarc_scores.get(k, 0.0) for k in snarc_keys], dtype=np.float32)
        parts.append(snarc_vec)

        # Action → hash of action name + args signature
        action_data = {'action': self.action_taken or '', 'args_keys': sorted(self.action_args.keys())}
        parts.append(_hash_to_vector(action_data, DIM_ACTION))

        # Outcome → success flag + reward + outcome hash
        outcome_vec = np.zeros(DIM_OUTCOME, dtype=np.float32)
        if self.success is not None:
            outcome_vec[0] = 1.0 if self.success else -1.0
        outcome_vec[1] = np.clip(self.reward, -1.0, 1.0)
        if self.outcome:
            outcome_hash = _hash_to_vector({'outcome': self.outcome}, DIM_OUTCOME - 2)
            outcome_vec[2:] = outcome_hash
        parts.append(outcome_vec)

        # Context tags → sparse feature hashing
        parts.append(_tags_to_vector(self.tags, DIM_CONTEXT))

        self.embedding = np.concatenate(parts).astype(np.float32)
        assert self.embedding.shape == (EMBEDDING_DIM,), f"Expected {EMBEDDING_DIM}, got {self.embedding.shape}"
        return self.embedding

    def to_dict(self) -> dict:
        """Serialize for storage."""
        d = {
            'episode_id': self.episode_id,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'cycle_id': self.cycle_id,
            'state_signature': self.state_signature,
            'sensory_summary': self.sensory_summary,
            'snarc_scores': self.snarc_scores,
            'cognitive_stance': self.cognitive_stance,
            'action_taken': self.action_taken,
            'action_args': self.action_args,
            'outcome': self.outcome,
            'reward': self.reward,
            'success': self.success,
            'tags': self.tags,
            'access_count': self.access_count,
            'last_access': self.last_access,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Episode':
        """Deserialize from storage."""
        ep = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        ep.compute_embedding()
        return ep


@dataclass
class EpisodicCue:
    """Partial cue for pattern-completion retrieval."""

    state_signature: Optional[dict] = None
    sensory_summary: Optional[dict] = None
    snarc_scores: Optional[dict] = None
    action_taken: Optional[str] = None
    tags: Optional[list] = None

    # Search parameters
    time_window: Optional[tuple] = None  # (start_ts, end_ts)
    min_salience: float = 0.0

    def to_embedding_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert cue to embedding + mask (1 where cue provides data, 0 otherwise)."""
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        mask = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        offset = 0

        # State signature
        if self.state_signature is not None:
            emb[offset:offset + DIM_STATE] = _hash_to_vector(self.state_signature, DIM_STATE)
            mask[offset:offset + DIM_STATE] = 1.0
        offset += DIM_STATE

        # SNARC scores
        if self.snarc_scores is not None:
            snarc_keys = ['surprise', 'novelty', 'arousal', 'reward', 'conflict']
            for i, k in enumerate(snarc_keys):
                if k in self.snarc_scores:
                    emb[offset + i] = self.snarc_scores[k]
                    mask[offset + i] = 1.0
        offset += DIM_SNARC

        # Action
        if self.action_taken is not None:
            action_data = {'action': self.action_taken, 'args_keys': []}
            emb[offset:offset + DIM_ACTION] = _hash_to_vector(action_data, DIM_ACTION)
            mask[offset:offset + DIM_ACTION] = 1.0
        offset += DIM_ACTION

        # Outcome (not in cue — we don't know the outcome yet)
        offset += DIM_OUTCOME

        # Tags
        if self.tags is not None:
            emb[offset:offset + DIM_CONTEXT] = _tags_to_vector(self.tags, DIM_CONTEXT)
            mask[offset:offset + DIM_CONTEXT] = 1.0
        offset += DIM_CONTEXT

        return emb, mask


@dataclass
class RecallResult:
    """A recalled episode with similarity metadata."""

    episode: Episode
    similarity: float
    match_components: dict = field(default_factory=dict)
    recency_weight: float = 1.0
    access_boost: float = 1.0

    @property
    def score(self) -> float:
        """Combined retrieval score."""
        return self.similarity * self.recency_weight * self.access_boost


@dataclass
class ConsolidationReport:
    """Summary of dream-state consolidation."""

    episodes_before: int = 0
    episodes_after: int = 0
    prototypes_created: int = 0
    episodes_merged: int = 0
    episodes_forgotten: int = 0
    duration_seconds: float = 0.0


# ─── Embedding helpers ─────────────────────────────────────────────

def _hash_to_vector(data: Any, dim: int) -> np.ndarray:
    """Deterministic projection of arbitrary data to fixed-dim vector.

    Uses SHA-256 hash expanded to fill the target dimension,
    then maps bytes to [-1, 1] range. Deterministic: same input
    always produces the same vector.
    """
    raw = json.dumps(data, sort_keys=True, default=str).encode()
    h = hashlib.sha256(raw).digest()
    # Expand hash to fill dimension (each byte → one float)
    expanded = bytearray()
    i = 0
    while len(expanded) < dim:
        expanded.extend(hashlib.sha256(h + i.to_bytes(2, 'big')).digest())
        i += 1
    arr = np.array([b / 127.5 - 1.0 for b in expanded[:dim]], dtype=np.float32)
    return arr


def _tags_to_vector(tags: list, dim: int) -> np.ndarray:
    """Sparse feature hashing for tags."""
    vec = np.zeros(dim, dtype=np.float32)
    for tag in tags:
        h = int(hashlib.md5(tag.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
