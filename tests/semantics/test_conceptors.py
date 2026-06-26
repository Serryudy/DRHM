"""★ Conceptor Logic Test — M4 acceptance gate (CLAUDE.md §8 M4).

Acceptance criteria:
  1. VSA bind produces a vector ORTHOGONAL to both operands.
  2. VSA bundle stays SIMILAR to its constituents.
  3. Permute makes "A then B" ≠ "B then A".
  4. Conceptor singular values ∈ [0, 1] — for C, NOT C, AND, OR.
  5. C_tone AND NOT(C_tone) ≈ zero matrix (law of contradiction).
  6. C_tone OR NOT(C_tone) ≈ identity matrix (law of excluded middle).
  7. C_tone recognises tone stimuli better than shape stimuli (and vice-versa).

Tests prefixed ``test_acceptance_`` are the canonical gate assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm import config
from drhm.semantics.conceptors import Conceptor
from drhm.semantics.vsa import VSA

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_STATE = 50    # conceptor state dimension (fast for CI)
N_SAMPLES = 200  # data snapshots per stimulus class
ALPHA = 8.0      # aperture for test conceptors
SMALL_D = 1_000  # VSA dimension for the combined VSA assertions in this file

RNG_SEED = config.SEED


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="module")
def vsa(rng: np.random.Generator) -> VSA:
    return VSA(D=SMALL_D, rng=rng)


@pytest.fixture(scope="module")
def tone_conceptor(rng: np.random.Generator) -> Conceptor:
    """Conceptor learned from 'tone' stimulus: activity biased to first half of neurons."""
    X = rng.standard_normal((N_SAMPLES, N_STATE))
    X[:, N_STATE // 2 :] *= 0.1  # sparse second half
    return Conceptor.from_data(X, alpha=ALPHA)


@pytest.fixture(scope="module")
def shape_conceptor(rng: np.random.Generator) -> Conceptor:
    """Conceptor learned from 'shape' stimulus: activity biased to second half of neurons."""
    X = rng.standard_normal((N_SAMPLES, N_STATE))
    X[:, : N_STATE // 2] *= 0.1  # sparse first half
    return Conceptor.from_data(X, alpha=ALPHA)


# ── ★ Acceptance: singular values in [0, 1] ───────────────────────────────────

class TestAcceptanceSingularValues:
    """All conceptors and their Boolean combinations must have SVs in [0, 1]."""

    def test_acceptance_tone_svs_in_unit_interval(self, tone_conceptor: Conceptor) -> None:
        svs = tone_conceptor.singular_values()
        assert svs.min() >= -1e-9, f"min SV {svs.min():.6f} < 0"
        assert svs.max() <=  1 + 1e-9, f"max SV {svs.max():.6f} > 1"

    def test_acceptance_shape_svs_in_unit_interval(self, shape_conceptor: Conceptor) -> None:
        svs = shape_conceptor.singular_values()
        assert svs.min() >= -1e-9
        assert svs.max() <=  1 + 1e-9

    def test_acceptance_not_tone_svs_in_unit_interval(self, tone_conceptor: Conceptor) -> None:
        not_c = ~tone_conceptor
        svs = not_c.singular_values()
        assert svs.min() >= -1e-9
        assert svs.max() <=  1 + 1e-9

    def test_acceptance_and_svs_in_unit_interval(
        self, tone_conceptor: Conceptor, shape_conceptor: Conceptor
    ) -> None:
        and_c = tone_conceptor & shape_conceptor
        svs = and_c.singular_values()
        assert svs.min() >= -1e-9
        assert svs.max() <=  1 + 1e-9

    def test_acceptance_or_svs_in_unit_interval(
        self, tone_conceptor: Conceptor, shape_conceptor: Conceptor
    ) -> None:
        or_c = tone_conceptor | shape_conceptor
        svs = or_c.singular_values()
        assert svs.min() >= -1e-9
        assert svs.max() <=  1 + 1e-9


# ── ★ Acceptance: Boolean logic laws ─────────────────────────────────────────

class TestAcceptanceBooleanLaws:

    def test_acceptance_contradiction_near_zero(self, tone_conceptor: Conceptor) -> None:
        """C AND NOT(C) is near-zero (law of contradiction).

        The Jaeger AND formula gives per-eigenvalue result λ(1-λ)/(1-λ+λ²), which
        peaks at exactly 1/3 when λ=0.5.  The result is NOT exactly 0 for finite α;
        it approaches 0 only as α→0.  Bound: max SV ≤ 1/3 ≈ 0.334.
        """
        and_c = tone_conceptor & (~tone_conceptor)
        svs = and_c.singular_values()
        assert svs.max() < 0.36, f"max SV of C∧¬C = {svs.max():.4f}, expected ≤ 1/3"

    def test_acceptance_excluded_middle_near_identity(self, tone_conceptor: Conceptor) -> None:
        """C OR NOT(C) is near-identity (law of excluded middle).

        By de Morgan, min SV of C∨¬C = 1 − max SV of C∧¬C ≥ 1 − 1/3 = 2/3 ≈ 0.667.
        """
        or_c = tone_conceptor | (~tone_conceptor)
        svs = or_c.singular_values()
        assert svs.min() > 0.64, f"min SV of C∨¬C = {svs.min():.4f}, expected ≥ 2/3"

    def test_acceptance_not_not_recovers_original(self, tone_conceptor: Conceptor) -> None:
        """NOT(NOT(C)) ≈ C."""
        recovered = ~(~tone_conceptor)
        diff = np.abs(recovered.matrix - tone_conceptor.matrix).max()
        assert diff < 1e-9, f"‖¬¬C − C‖_∞ = {diff:.2e}"

    def test_and_commutative(
        self, tone_conceptor: Conceptor, shape_conceptor: Conceptor
    ) -> None:
        """C1 ∧ C2 ≈ C2 ∧ C1 (symmetric formula → symmetric result)."""
        and_12 = (tone_conceptor & shape_conceptor).matrix
        and_21 = (shape_conceptor & tone_conceptor).matrix
        np.testing.assert_allclose(and_12, and_21, atol=1e-6)

    def test_or_commutative(
        self, tone_conceptor: Conceptor, shape_conceptor: Conceptor
    ) -> None:
        and_12 = (tone_conceptor | shape_conceptor).matrix
        and_21 = (shape_conceptor | tone_conceptor).matrix
        np.testing.assert_allclose(and_12, and_21, atol=1e-6)


# ── ★ Acceptance: discrimination ─────────────────────────────────────────────

class TestAcceptanceDiscrimination:

    def test_acceptance_tone_conceptor_recognises_tone(
        self, rng: np.random.Generator, tone_conceptor: Conceptor
    ) -> None:
        """C_tone projects tone stimuli to a higher-norm output than shape stimuli."""
        X_tone = rng.standard_normal((50, N_STATE))
        X_tone[:, N_STATE // 2 :] *= 0.1
        X_shape = rng.standard_normal((50, N_STATE))
        X_shape[:, : N_STATE // 2] *= 0.1

        tone_norms  = np.linalg.norm(X_tone  @ tone_conceptor.matrix.T, axis=1).mean()
        shape_norms = np.linalg.norm(X_shape @ tone_conceptor.matrix.T, axis=1).mean()
        assert tone_norms > shape_norms, (
            f"C_tone failed to discriminate: tone={tone_norms:.3f} shape={shape_norms:.3f}"
        )

    def test_acceptance_shape_conceptor_recognises_shape(
        self, rng: np.random.Generator, shape_conceptor: Conceptor
    ) -> None:
        """C_shape projects shape stimuli higher than tone stimuli."""
        X_tone = rng.standard_normal((50, N_STATE))
        X_tone[:, N_STATE // 2 :] *= 0.1
        X_shape = rng.standard_normal((50, N_STATE))
        X_shape[:, : N_STATE // 2] *= 0.1

        tone_norms  = np.linalg.norm(X_tone  @ shape_conceptor.matrix.T, axis=1).mean()
        shape_norms = np.linalg.norm(X_shape @ shape_conceptor.matrix.T, axis=1).mean()
        assert shape_norms > tone_norms


# ── ★ Acceptance: VSA properties (CLAUDE.md §8 M4) ───────────────────────────
# Repeated here so this acceptance-gate file is self-contained.

class TestAcceptanceVSA:

    def test_acceptance_bind_orthogonal_to_operands(self, vsa: VSA) -> None:
        a, b = vsa.random(), vsa.random()
        bound = vsa.bind(a, b)
        assert abs(vsa.cosine(bound, a)) < 0.1
        assert abs(vsa.cosine(bound, b)) < 0.1

    def test_acceptance_bundle_similar_to_constituents(self, vsa: VSA) -> None:
        a, b = vsa.random(), vsa.random()
        bundled = vsa.bundle([a, b])
        assert vsa.cosine(bundled, a) > 0.3
        assert vsa.cosine(bundled, b) > 0.3

    def test_acceptance_permute_order_sensitive(self, vsa: VSA) -> None:
        """bind(A, permute(B)) must be orthogonal to bind(B, permute(A))."""
        a, b = vsa.random(), vsa.random()
        ab_seq = vsa.bind(a, vsa.permute(b))
        ba_seq = vsa.bind(b, vsa.permute(a))
        assert abs(vsa.cosine(ab_seq, ba_seq)) < 0.15


# ── from_data mechanics ───────────────────────────────────────────────────────

class TestFromData:

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            Conceptor.from_data(np.ones((10,)), alpha=1.0)

    def test_n_property(self, tone_conceptor: Conceptor) -> None:
        assert tone_conceptor.n == N_STATE

    def test_matrix_shape(self, tone_conceptor: Conceptor) -> None:
        assert tone_conceptor.matrix.shape == (N_STATE, N_STATE)

    def test_matrix_is_copy(self, tone_conceptor: Conceptor) -> None:
        m = tone_conceptor.matrix
        m[0, 0] = 999.0
        assert tone_conceptor.matrix[0, 0] != 999.0

    def test_from_matrix_clips_svs(self) -> None:
        """from_matrix must clip eigenvalues even if input violates the invariant."""
        rng = np.random.default_rng(3)
        raw = rng.standard_normal((10, 10))
        raw = raw @ raw.T  # PSD but eigenvalues >> 1
        c = Conceptor.from_matrix(raw, alpha=0.0)
        svs = c.singular_values()
        assert svs.max() <= 1 + 1e-9


# ── apply() ───────────────────────────────────────────────────────────────────

class TestApply:

    def test_shape(self, tone_conceptor: Conceptor) -> None:
        x = np.random.default_rng(0).standard_normal(N_STATE)
        assert tone_conceptor.apply(x).shape == (N_STATE,)

    def test_zero_input(self, tone_conceptor: Conceptor) -> None:
        np.testing.assert_allclose(
            tone_conceptor.apply(np.zeros(N_STATE)), np.zeros(N_STATE)
        )


# ── NOT / AND / OR ────────────────────────────────────────────────────────────

class TestBooleanOps:

    def test_not_shape(self, tone_conceptor: Conceptor) -> None:
        assert (~tone_conceptor).matrix.shape == (N_STATE, N_STATE)

    def test_and_shape(self, tone_conceptor: Conceptor, shape_conceptor: Conceptor) -> None:
        assert (tone_conceptor & shape_conceptor).matrix.shape == (N_STATE, N_STATE)

    def test_or_shape(self, tone_conceptor: Conceptor, shape_conceptor: Conceptor) -> None:
        assert (tone_conceptor | shape_conceptor).matrix.shape == (N_STATE, N_STATE)

    def test_not_is_complement_of_trace(self, tone_conceptor: Conceptor) -> None:
        """trace(C) + trace(NOT C) ≈ n."""
        n = tone_conceptor.n
        tr_c    = np.trace(tone_conceptor.matrix)
        tr_notc = np.trace((~tone_conceptor).matrix)
        assert abs(tr_c + tr_notc - n) < 1e-6

    def test_or_dominates_and(
        self, tone_conceptor: Conceptor, shape_conceptor: Conceptor
    ) -> None:
        """trace(C1 OR C2) ≥ trace(C1 AND C2) always holds."""
        and_c = tone_conceptor & shape_conceptor
        or_c  = tone_conceptor | shape_conceptor
        assert np.trace(or_c.matrix) >= np.trace(and_c.matrix) - 1e-6

    def test_and_weakens(
        self, tone_conceptor: Conceptor, shape_conceptor: Conceptor
    ) -> None:
        """AND must not strengthen either operand: trace(AND) ≤ min(trace(C1), trace(C2))."""
        and_c = tone_conceptor & shape_conceptor
        assert np.trace(and_c.matrix) <= np.trace(tone_conceptor.matrix) + 1e-6
        assert np.trace(and_c.matrix) <= np.trace(shape_conceptor.matrix) + 1e-6
