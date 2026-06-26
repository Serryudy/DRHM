"""Tests for drhm/semantics/vsa.py — VSA hypervector algebra (CLAUDE.md §8 M4).

The ★ Conceptor Logic Test (acceptance gate) lives in test_conceptors.py.
This file covers the VSA half of that gate plus supporting unit tests.

Acceptance assertions from CLAUDE.md §8 M4:
  - bind produces a vector ORTHOGONAL (≈0 cosine) to both operands
  - bundle stays SIMILAR to its constituents
  - permute makes "A then B" ≠ "B then A"
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm.semantics.vsa import VSA

# ── Fixtures ──────────────────────────────────────────────────────────────────

SMALL_D = 1_000  # fast tests; cosine std ≈ 0.032 at this D


@pytest.fixture
def vsa() -> VSA:
    return VSA(D=SMALL_D, rng=np.random.default_rng(42))


@pytest.fixture
def ab(vsa: VSA) -> tuple[np.ndarray, np.ndarray]:
    return vsa.random(), vsa.random()


# ── random() ─────────────────────────────────────────────────────────────────

class TestRandom:

    def test_shape(self, vsa: VSA) -> None:
        assert vsa.random().shape == (SMALL_D,)

    def test_bipolar(self, vsa: VSA) -> None:
        v = vsa.random()
        assert set(v.tolist()).issubset({-1, 1})

    def test_dtype(self, vsa: VSA) -> None:
        assert vsa.random().dtype == np.int8

    def test_approximately_balanced(self, vsa: VSA) -> None:
        """Mean of a long sample should be near 0."""
        rng = np.random.default_rng(7)
        vsa2 = VSA(D=10_000, rng=rng)
        v = vsa2.random()
        assert abs(v.mean()) < 0.05

    def test_two_randoms_are_orthogonal(self, vsa: VSA) -> None:
        """Two independent random HVs should have cosine ≈ 0 (std ≈ 1/√D)."""
        a, b = vsa.random(), vsa.random()
        assert abs(vsa.cosine(a, b)) < 0.1


# ── bind() ────────────────────────────────────────────────────────────────────

class TestBind:

    def test_shape_preserved(self, vsa: VSA, ab: tuple) -> None:
        a, b = ab
        assert vsa.bind(a, b).shape == (SMALL_D,)

    def test_bipolar_result(self, vsa: VSA, ab: tuple) -> None:
        a, b = ab
        result = vsa.bind(a, b)
        assert set(result.tolist()).issubset({-1, 1})

    def test_commutative(self, vsa: VSA, ab: tuple) -> None:
        a, b = ab
        np.testing.assert_array_equal(vsa.bind(a, b), vsa.bind(b, a))

    def test_self_inverse(self, vsa: VSA, ab: tuple) -> None:
        """bind(bind(a, b), b) == a  (⊗ is its own inverse)."""
        a, b = ab
        np.testing.assert_array_equal(vsa.bind(vsa.bind(a, b), b), a)

    # ★ Acceptance: bind result orthogonal to both operands
    def test_orthogonal_to_operand_a(self, vsa: VSA, ab: tuple) -> None:
        a, b = ab
        bound = vsa.bind(a, b)
        assert abs(vsa.cosine(bound, a)) < 0.1

    def test_orthogonal_to_operand_b(self, vsa: VSA, ab: tuple) -> None:
        a, b = ab
        bound = vsa.bind(a, b)
        assert abs(vsa.cosine(bound, b)) < 0.1

    def test_bind_identity(self, vsa: VSA) -> None:
        """Binding with the all-ones vector is an identity."""
        a = vsa.random()
        ones = np.ones(SMALL_D, dtype=np.int8)
        np.testing.assert_array_equal(vsa.bind(a, ones), a)


# ── bundle() ──────────────────────────────────────────────────────────────────

class TestBundle:

    def test_shape_preserved(self, vsa: VSA) -> None:
        a, b = vsa.random(), vsa.random()
        assert vsa.bundle([a, b]).shape == (SMALL_D,)

    def test_bipolar_result(self, vsa: VSA) -> None:
        result = vsa.bundle([vsa.random(), vsa.random()])
        assert set(result.tolist()).issubset({-1, 1})

    def test_single_vector_roundtrip(self, vsa: VSA) -> None:
        """bundle of a single vector must equal that vector."""
        a = vsa.random()
        np.testing.assert_array_equal(vsa.bundle([a]), a)

    def test_empty_raises(self, vsa: VSA) -> None:
        with pytest.raises(ValueError):
            vsa.bundle([])

    # ★ Acceptance: bundle stays similar to its constituents
    def test_similar_to_first_constituent(self, vsa: VSA) -> None:
        a, b = vsa.random(), vsa.random()
        bundled = vsa.bundle([a, b])
        assert vsa.cosine(bundled, a) > 0.3

    def test_similar_to_second_constituent(self, vsa: VSA) -> None:
        a, b = vsa.random(), vsa.random()
        bundled = vsa.bundle([a, b])
        assert vsa.cosine(bundled, b) > 0.3

    def test_three_vector_bundle_similar_to_each(self, vsa: VSA) -> None:
        a, b, c = vsa.random(), vsa.random(), vsa.random()
        bundled = vsa.bundle([a, b, c])
        assert vsa.cosine(bundled, a) > 0.2
        assert vsa.cosine(bundled, b) > 0.2
        assert vsa.cosine(bundled, c) > 0.2


# ── permute() ─────────────────────────────────────────────────────────────────

class TestPermute:

    def test_shape_preserved(self, vsa: VSA) -> None:
        a = vsa.random()
        assert vsa.permute(a).shape == (SMALL_D,)

    def test_bipolar_preserved(self, vsa: VSA) -> None:
        a = vsa.random()
        assert set(vsa.permute(a).tolist()).issubset({-1, 1})

    def test_permute_changes_vector(self, vsa: VSA) -> None:
        a = vsa.random()
        # Probability that permute(a) == a is astronomically small for large D
        assert not np.array_equal(vsa.permute(a), a)

    def test_inv_permute_roundtrip(self, vsa: VSA) -> None:
        a = vsa.random()
        np.testing.assert_array_equal(vsa.inv_permute(vsa.permute(a)), a)

    def test_permute_preserves_cosine_with_self(self, vsa: VSA) -> None:
        """cosine(permute(a), permute(b)) ≈ cosine(a, b)  (isometry)."""
        a, b = vsa.random(), vsa.random()
        assert abs(vsa.cosine(vsa.permute(a), vsa.permute(b)) - vsa.cosine(a, b)) < 0.05

    # ★ Acceptance: permute makes "A then B" ≠ "B then A"
    def test_order_sensitive_encoding(self, vsa: VSA) -> None:
        """bind(A, permute(B)) is orthogonal to bind(B, permute(A))."""
        a, b = vsa.random(), vsa.random()
        ab_seq = vsa.bind(a, vsa.permute(b))  # "A then B"
        ba_seq = vsa.bind(b, vsa.permute(a))  # "B then A"
        # Different orderings produce orthogonal representations
        assert abs(vsa.cosine(ab_seq, ba_seq)) < 0.15


# ── cosine() ──────────────────────────────────────────────────────────────────

class TestCosine:

    def test_self_similarity_is_one(self, vsa: VSA) -> None:
        a = vsa.random()
        assert abs(vsa.cosine(a, a) - 1.0) < 1e-6

    def test_opposite_similarity_is_minus_one(self, vsa: VSA) -> None:
        a = vsa.random()
        assert abs(vsa.cosine(a, -a) + 1.0) < 1e-6

    def test_range(self, vsa: VSA) -> None:
        a, b = vsa.random(), vsa.random()
        c = vsa.cosine(a, b)
        assert -1.0 <= c <= 1.0


# ── encode_x() — the unit "X" formula ────────────────────────────────────────

class TestEncodeX:

    def test_shape(self, vsa: VSA) -> None:
        ctx = vsa.random()
        struct = vsa.random()
        feats = [vsa.random(), vsa.random()]
        x = vsa.encode_x(ctx, struct, feats)
        assert x.shape == (SMALL_D,)

    def test_bipolar(self, vsa: VSA) -> None:
        ctx, struct = vsa.random(), vsa.random()
        x = vsa.encode_x(ctx, struct, [vsa.random()])
        assert set(x.tolist()).issubset({-1, 1})

    def test_no_features(self, vsa: VSA) -> None:
        """With no features, X = bind(context, structure)."""
        ctx, struct = vsa.random(), vsa.random()
        x = vsa.encode_x(ctx, struct, [])
        expected = vsa.bind(ctx, struct)
        np.testing.assert_array_equal(x, expected)

    def test_different_contexts_yield_different_x(self, vsa: VSA) -> None:
        """Same feature in two different contexts → orthogonal X_t vectors."""
        feature = vsa.random()
        struct = vsa.random()
        ctx1, ctx2 = vsa.random(), vsa.random()
        x1 = vsa.encode_x(ctx1, struct, [feature])
        x2 = vsa.encode_x(ctx2, struct, [feature])
        # Different contexts stamp the feature differently
        assert abs(vsa.cosine(x1, x2)) < 0.3

    def test_different_features_yield_different_x(self, vsa: VSA) -> None:
        """Same context/structure but different features → different X_t."""
        ctx, struct = vsa.random(), vsa.random()
        f1, f2 = vsa.random(), vsa.random()
        x1 = vsa.encode_x(ctx, struct, [f1])
        x2 = vsa.encode_x(ctx, struct, [f2])
        assert vsa.cosine(x1, x2) < 0.9  # must differ meaningfully
