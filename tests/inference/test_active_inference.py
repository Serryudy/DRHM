"""Tests for drhm/inference/active_inference.py — GenerativeModel (CLAUDE.md §8 M5).

Acceptance assertions:
  - Novel stimuli register as high surprise (near 1.0 before any observations).
  - Repeated exposure to the same stimulus decreases surprise monotonically.
  - After many repetitions, surprise drops well below 0.5 for the familiar stimulus.
  - A different stimulus after habituation causes surprise to spike back up.
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm import config
from drhm.inference.active_inference import GenerativeModel

D_SMALL = 500


@pytest.fixture
def model() -> GenerativeModel:
    return GenerativeModel(D=D_SMALL, eta=config.GENERATIVE_MODEL_ETA)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(config.SEED)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:

    def test_n_obs_zero_at_start(self, model: GenerativeModel) -> None:
        assert model.n_obs == 0

    def test_belief_zero_at_start(self, model: GenerativeModel) -> None:
        np.testing.assert_array_equal(model.belief, np.zeros(D_SMALL))

    def test_wrong_D_raises(self) -> None:
        with pytest.raises(ValueError):
            GenerativeModel(D=0)

    def test_wrong_eta_raises(self) -> None:
        with pytest.raises(ValueError):
            GenerativeModel(D=10, eta=0.0)
        with pytest.raises(ValueError):
            GenerativeModel(D=10, eta=1.1)

    def test_belief_is_copy(self, model: GenerativeModel) -> None:
        b = model.belief
        b[0] = 999.0
        assert model.belief[0] != 999.0


# ── surprise() ────────────────────────────────────────────────────────────────

class TestSurprise:

    # ★ Acceptance: novel stimulus → high surprise before any observations
    def test_acceptance_novel_surprise_is_one_before_any_obs(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        """Before any observations, every stimulus is maximally novel (surprise = 1.0)."""
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        assert model.surprise(x) == pytest.approx(1.0)

    def test_surprise_after_zero_obs_is_one(self, model: GenerativeModel) -> None:
        assert model.surprise(np.ones(D_SMALL, dtype=np.int8)) == pytest.approx(1.0)

    def test_surprise_range(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        model.observe(x)
        s = model.surprise(x)
        assert 0.0 <= s <= 2.0

    def test_surprise_after_self_observe_is_zero(self) -> None:
        """Observing x then computing surprise(x) with eta=1 yields 0."""
        m = GenerativeModel(D=D_SMALL, eta=1.0)
        x = np.ones(D_SMALL, dtype=np.int8)
        m.observe(x)
        assert m.surprise(x) == pytest.approx(0.0, abs=1e-6)

    def test_surprise_does_not_modify_model(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        model.observe(x)
        belief_before = model.belief.copy()
        model.surprise(rng.choice([-1, 1], size=D_SMALL).astype(np.int8))
        np.testing.assert_array_equal(model.belief, belief_before)

    def test_opposite_vector_has_high_surprise(self) -> None:
        """After observing +1-vector, its antipode has surprise ≈ 2."""
        m = GenerativeModel(D=D_SMALL, eta=1.0)
        x = np.ones(D_SMALL, dtype=np.int8)
        m.observe(x)
        assert m.surprise(-x) == pytest.approx(2.0, abs=1e-6)


# ── observe() ─────────────────────────────────────────────────────────────────

class TestObserve:

    def test_increments_n_obs(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        for i in range(1, 6):
            model.observe(x)
            assert model.n_obs == i

    def test_returns_pre_update_surprise(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        pre_surprise = model.surprise(x)
        returned = model.observe(x)
        assert returned == pytest.approx(pre_surprise)

    def test_updates_belief(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        belief_before = model.belief.copy()
        model.observe(x)
        assert not np.array_equal(model.belief, belief_before)

    # ★ Acceptance: repeated exposure decreases surprise
    def test_acceptance_repeated_observation_decreases_surprise(self, rng: np.random.Generator) -> None:
        """Surprise(X_t) must decrease monotonically as the same X_t is observed repeatedly."""
        m = GenerativeModel(D=D_SMALL, eta=0.3)
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        surprises = []
        for _ in range(20):
            m.observe(x)
            surprises.append(m.surprise(x))
        # Must be monotonically non-increasing (within small noise tolerance)
        for i in range(1, len(surprises)):
            assert surprises[i] <= surprises[i - 1] + 1e-9, (
                f"Surprise increased at step {i}: {surprises[i - 1]:.4f} → {surprises[i]:.4f}"
            )

    # ★ Acceptance: surprise drops well below 0.5 after many repetitions
    def test_acceptance_familiar_stimulus_low_surprise(self, rng: np.random.Generator) -> None:
        """After 50 observations of the same stimulus, surprise must be < 0.1."""
        m = GenerativeModel(D=D_SMALL, eta=0.3)
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        for _ in range(50):
            m.observe(x)
        assert m.surprise(x) < 0.1

    # ★ Acceptance: different stimulus after habituation spikes surprise
    def test_acceptance_novel_after_familiar_spikes_surprise(self, rng: np.random.Generator) -> None:
        """A novel stimulus after repeated exposure to a different one causes surprise > 0.5."""
        m = GenerativeModel(D=D_SMALL, eta=0.3)
        x_a = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        x_b = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        for _ in range(30):
            m.observe(x_a)
        # x_b is orthogonal to x_a (random, high-D) → surprise should be near 1
        assert m.surprise(x_b) > 0.5


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_zeroes_belief(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        model.observe(x)
        model.reset()
        np.testing.assert_array_equal(model.belief, np.zeros(D_SMALL))

    def test_reset_zeroes_n_obs(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        model.observe(x)
        model.reset()
        assert model.n_obs == 0

    def test_reset_restores_max_surprise(self, model: GenerativeModel, rng: np.random.Generator) -> None:
        x = rng.choice([-1, 1], size=D_SMALL).astype(np.int8)
        for _ in range(20):
            model.observe(x)
        model.reset()
        assert model.surprise(x) == pytest.approx(1.0)
