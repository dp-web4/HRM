"""Tests for mechanics_encoder.py — model forward shapes, embedding
similarity lookup, text-anchor helpers, dataset wrapper.

No network, no arcengine, no training. Heavy paths (`build_mechanics_dataset`,
`train`) depend on real arcengine traces and are exercised end-to-end when
the encoder is trained on the fleet; they are not unit-tested here.

Covers the parts of mechanics_encoder that are pure logic and benefit from
regression protection: the shared-dynamics module's forward contracts and
the cosine-similarity neighbor lookup that the llm_dispatch prompt
consumes at invoke time.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.thalamic_router.frame_router import (
    FRAME_H, FRAME_W, N_ACTIONS, N_COLORS,
)
from sage.cognition.thalamic_router.mechanics_encoder import (
    DEFAULT_EMB_DIM,
    DEFAULT_POOL_DIM,
    NOMIC_EMB_DIM,
    MechanicsDataset,
    MechanicsEncoder,
    load_world_model_text,
    nearest_games,
    nomic_embed_text,
)


# ───────────────────────────────────────────────────────────────────
# MechanicsEncoder — forward shapes
# ───────────────────────────────────────────────────────────────────


def test_mechanics_encoder_default_dims_match_module_constants():
    model = MechanicsEncoder(n_games=4)
    assert model.n_games == 4
    assert model.emb_dim == DEFAULT_EMB_DIM
    assert model.pool_dim == DEFAULT_POOL_DIM
    # Per-game embedding table sized (n_games, emb_dim)
    assert model.game_embedding.weight.shape == (4, DEFAULT_EMB_DIM)


def test_encode_frames_returns_pooled_visual_feature():
    model = MechanicsEncoder(n_games=3).eval()
    b = 2
    prev = torch.zeros(b, N_COLORS, FRAME_H, FRAME_W)
    curr = torch.zeros(b, N_COLORS, FRAME_H, FRAME_W)
    with torch.no_grad():
        pool = model.encode_frames(prev, curr)
    assert pool.shape == (b, DEFAULT_POOL_DIM)


def test_predict_next_pool_shape_matches_pool_dim():
    model = MechanicsEncoder(n_games=3).eval()
    b = 4
    pool_t = torch.randn(b, DEFAULT_POOL_DIM)
    action_oh = torch.zeros(b, N_ACTIONS)
    action_oh[:, 2] = 1.0
    game_emb = torch.randn(b, DEFAULT_EMB_DIM)
    with torch.no_grad():
        pred = model.predict_next_pool(pool_t, action_oh, game_emb)
    assert pred.shape == (b, DEFAULT_POOL_DIM)


def test_game_text_projection_outputs_768d():
    """The text-anchor head projects the 32d game embedding into the same
    space as nomic-embed-text (768d) for MSE regularization."""
    model = MechanicsEncoder(n_games=5).eval()
    # Feed all game embeddings — shape (n_games, emb_dim) → (n_games, 768)
    with torch.no_grad():
        proj = model.game_text_projection(model.game_embedding.weight)
    assert proj.shape == (5, NOMIC_EMB_DIM)


def test_game_embedding_lookup_returns_correct_row():
    model = MechanicsEncoder(n_games=6).eval()
    for idx in [0, 3, 5]:
        emb = model.game_embedding(torch.tensor([idx]))
        assert emb.shape == (1, DEFAULT_EMB_DIM)
        # Matches the raw weight table row
        assert torch.allclose(emb[0], model.game_embedding.weight[idx])


# ───────────────────────────────────────────────────────────────────
# nearest_games — cosine similarity neighbor lookup
# ───────────────────────────────────────────────────────────────────


def test_nearest_games_excludes_self_and_orders_by_similarity():
    # Craft four vectors with known pairwise angles:
    #   g0 = x̂,  g1 ≈ x̂ (nearest),  g2 = ŷ (orthogonal),  g3 = -x̂ (farthest)
    embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.14],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    slugs = ["a", "b", "c", "d"]
    neighbors = nearest_games(query_idx=0, embeddings=embeddings, game_slugs=slugs, k=3)
    names = [n for n, _ in neighbors]
    # Self ("a") excluded; b closest, then c, then d
    assert "a" not in names
    assert names == ["b", "c", "d"]
    # Similarities strictly decreasing
    sims = [s for _, s in neighbors]
    assert sims[0] > sims[1] > sims[2]


def test_nearest_games_k_truncates():
    embeddings = np.random.RandomState(0).randn(8, 16).astype(np.float32)
    slugs = [f"g{i}" for i in range(8)]
    top1 = nearest_games(0, embeddings, slugs, k=1)
    top5 = nearest_games(0, embeddings, slugs, k=5)
    assert len(top1) == 1
    assert len(top5) == 5
    # top1 must be the first entry of top5
    assert top1[0][0] == top5[0][0]


def test_nearest_games_identical_vector_gives_similarity_one():
    # Two games with identical embeddings → cosine similarity = 1.0
    embeddings = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    slugs = ["twin_a", "twin_b", "other"]
    neighbors = nearest_games(0, embeddings, slugs, k=2)
    assert neighbors[0][0] == "twin_b"
    assert neighbors[0][1] == pytest.approx(1.0, abs=1e-5)


def test_nearest_games_returns_slugs_not_indices():
    """Guard against an off-by-one that would return indices as names."""
    embeddings = np.eye(4, dtype=np.float32)
    slugs = ["alpha", "beta", "gamma", "delta"]
    neighbors = nearest_games(2, embeddings, slugs, k=1)
    # Any of alpha/beta/delta is an acceptable "nearest" (all orthogonal), but
    # the return must be one of the string slugs, not the query slug.
    name, _ = neighbors[0]
    assert name in {"alpha", "beta", "delta"}
    assert name != "gamma"


# ───────────────────────────────────────────────────────────────────
# load_world_model_text — path resolution + truncation
# ───────────────────────────────────────────────────────────────────


def test_load_world_model_text_missing_returns_empty(monkeypatch, tmp_path):
    # Point SHARED_CONTEXT_DIR at an empty tmp_path and mask the fallback
    # home/repos paths so nothing can match.
    monkeypatch.setenv("SHARED_CONTEXT_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "nonexistent_home")
    assert load_world_model_text("definitely_not_a_game") == ""


def test_load_world_model_text_via_shared_context_env(monkeypatch, tmp_path):
    # Seed a fake world-model markdown in the env-var-resolved location.
    wm_dir = tmp_path / "arc-agi-3" / "world-models"
    wm_dir.mkdir(parents=True)
    (wm_dir / "faketris.md").write_text("Mechanics of faketris: rotate and drop.")
    monkeypatch.setenv("SHARED_CONTEXT_DIR", str(tmp_path))

    text = load_world_model_text("faketris")
    assert "rotate and drop" in text


def test_load_world_model_text_truncates_at_max_chars(monkeypatch, tmp_path):
    wm_dir = tmp_path / "arc-agi-3" / "world-models"
    wm_dir.mkdir(parents=True)
    # Write more than default max_chars=3000.
    (wm_dir / "bigtris.md").write_text("x" * 10_000)
    monkeypatch.setenv("SHARED_CONTEXT_DIR", str(tmp_path))

    text = load_world_model_text("bigtris", max_chars=500)
    assert len(text) == 500


# ───────────────────────────────────────────────────────────────────
# nomic_embed_text — Ollama embedding endpoint with graceful failure
# ───────────────────────────────────────────────────────────────────


def test_nomic_embed_text_returns_none_on_network_error(monkeypatch):
    """Ollama down / wrong URL must not crash training — returns None."""
    import urllib.request
    def _boom(*args, **kwargs):
        raise OSError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert nomic_embed_text("any text", base_url="http://localhost:1") is None


def test_nomic_embed_text_parses_embedding_response(monkeypatch):
    """Valid Ollama response → float32 numpy vector of correct length."""
    import urllib.request

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    target = [0.1, -0.2, 0.3, 0.4]
    payload = json.dumps({"embedding": target}).encode("utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(payload))

    emb = nomic_embed_text("hello")
    assert emb is not None
    assert emb.dtype == np.float32
    assert emb.shape == (len(target),)
    np.testing.assert_allclose(emb, np.array(target, dtype=np.float32))


# ───────────────────────────────────────────────────────────────────
# MechanicsDataset — torch Dataset adapter
# ───────────────────────────────────────────────────────────────────


def _dummy_tuple(action: int = 0, game_idx: int = 0):
    zero = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    return {
        "prev_oh": zero, "curr_oh": zero, "next_oh": zero,
        "action": action, "game_idx": game_idx,
    }


def test_mechanics_dataset_len_matches_input():
    tuples = [_dummy_tuple() for _ in range(7)]
    ds = MechanicsDataset(tuples)
    assert len(ds) == 7


def test_mechanics_dataset_getitem_returns_torch_tensors_with_expected_shapes():
    ds = MechanicsDataset([_dummy_tuple(action=3, game_idx=2)])
    item = ds[0]
    assert isinstance(item["prev_oh"], torch.Tensor)
    assert item["prev_oh"].shape == (N_COLORS, FRAME_H, FRAME_W)
    assert item["curr_oh"].shape == (N_COLORS, FRAME_H, FRAME_W)
    assert item["next_oh"].shape == (N_COLORS, FRAME_H, FRAME_W)
    # Scalar tensors, right dtypes
    assert item["action"].dtype == torch.long
    assert int(item["action"]) == 3
    assert item["game_idx"].dtype == torch.long
    assert int(item["game_idx"]) == 2
