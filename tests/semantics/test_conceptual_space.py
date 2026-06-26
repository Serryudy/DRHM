"""Tests for drhm/semantics/conceptual_space.py (CLAUDE.md §8 M4 supporting tests).

Supporting acceptance assertions:
  - convexity: midpoint of two same-category instances is in-category
  - prototype recall: encode → decode round-trips to the correct name
  - hypervector noise tolerance: random bit flips don't change nearest concept
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm.semantics.conceptual_space import ConceptualSpace, Prototype
from drhm.semantics.vsa import VSA

SMALL_D = 1_000


@pytest.fixture
def vsa() -> VSA:
    return VSA(D=SMALL_D, rng=np.random.default_rng(0))


@pytest.fixture
def pitch_space(vsa: VSA) -> ConceptualSpace:
    """1-D pitch space with three prototypes: low, mid, high."""
    cs = ConceptualSpace(n_dims=1, vsa=vsa)
    cs.add_prototype("low",  np.array([0.0]))
    cs.add_prototype("mid",  np.array([0.5]))
    cs.add_prototype("high", np.array([1.0]))
    return cs


@pytest.fixture
def color_space(vsa: VSA) -> ConceptualSpace:
    """2-D (hue, brightness) space with four corner prototypes."""
    cs = ConceptualSpace(n_dims=2, vsa=vsa)
    cs.add_prototype("dark_cool",  np.array([0.1, 0.1]))
    cs.add_prototype("dark_warm",  np.array([0.9, 0.1]))
    cs.add_prototype("light_cool", np.array([0.1, 0.9]))
    cs.add_prototype("light_warm", np.array([0.9, 0.9]))
    return cs


# ── Prototype management ──────────────────────────────────────────────────────

class TestPrototypeManagement:

    def test_add_prototype_returns_hypervector(self, vsa: VSA) -> None:
        cs = ConceptualSpace(n_dims=1, vsa=vsa)
        hv = cs.add_prototype("x", np.array([0.0]))
        assert hv.shape == (SMALL_D,)
        assert set(hv.tolist()).issubset({-1, 1})

    def test_prototypes_list_length(self, pitch_space: ConceptualSpace) -> None:
        assert len(pitch_space.prototypes) == 3

    def test_empty_space_raises_on_nearest(self, vsa: VSA) -> None:
        cs = ConceptualSpace(n_dims=1, vsa=vsa)
        with pytest.raises(ValueError):
            cs.nearest(np.array([0.5]))

    def test_empty_space_raises_on_decode(self, vsa: VSA) -> None:
        cs = ConceptualSpace(n_dims=1, vsa=vsa)
        with pytest.raises(ValueError):
            cs.decode(vsa.random())

    def test_two_prototypes_have_distinct_hypervectors(self, vsa: VSA) -> None:
        cs = ConceptualSpace(n_dims=1, vsa=vsa)
        h1 = cs.add_prototype("a", np.array([0.0]))
        h2 = cs.add_prototype("b", np.array([1.0]))
        assert not np.array_equal(h1, h2)


# ── Voronoi lookup ────────────────────────────────────────────────────────────

class TestVoronoiLookup:

    def test_exact_prototype_coordinates(self, pitch_space: ConceptualSpace) -> None:
        assert pitch_space.nearest(np.array([0.0])).name == "low"
        assert pitch_space.nearest(np.array([0.5])).name == "mid"
        assert pitch_space.nearest(np.array([1.0])).name == "high"

    def test_nearby_point_maps_to_nearest(self, pitch_space: ConceptualSpace) -> None:
        assert pitch_space.nearest(np.array([0.1])).name == "low"
        assert pitch_space.nearest(np.array([0.45])).name == "mid"
        assert pitch_space.nearest(np.array([0.9])).name == "high"

    def test_is_in_region_true(self, pitch_space: ConceptualSpace) -> None:
        assert pitch_space.is_in_region(np.array([0.05]), "low")

    def test_is_in_region_false(self, pitch_space: ConceptualSpace) -> None:
        assert not pitch_space.is_in_region(np.array([0.9]), "low")

    def test_2d_voronoi(self, color_space: ConceptualSpace) -> None:
        assert color_space.nearest(np.array([0.15, 0.15])).name == "dark_cool"
        assert color_space.nearest(np.array([0.85, 0.85])).name == "light_warm"


# ── Encode / decode ───────────────────────────────────────────────────────────

class TestEncodeDecode:

    # ★ Prototype recall
    def test_decode_recovers_prototype_name(self, pitch_space: ConceptualSpace) -> None:
        """Encoding a prototype coordinate then decoding must return its name."""
        for name, coord in [("low", np.array([0.0])), ("mid", np.array([0.5])), ("high", np.array([1.0]))]:
            hv = pitch_space.encode(coord)
            assert pitch_space.decode(hv) == name

    def test_encode_returns_correct_hypervector(self, pitch_space: ConceptualSpace) -> None:
        hv = pitch_space.encode(np.array([0.0]))
        p = pitch_space.nearest(np.array([0.0]))
        np.testing.assert_array_equal(hv, p.hypervector)

    def test_same_voronoi_cell_same_hypervector(self, pitch_space: ConceptualSpace) -> None:
        """Two points in the same Voronoi cell encode to the identical hypervector."""
        hv1 = pitch_space.encode(np.array([0.0]))
        hv2 = pitch_space.encode(np.array([0.05]))
        np.testing.assert_array_equal(hv1, hv2)

    def test_different_cells_different_hypervectors(self, pitch_space: ConceptualSpace) -> None:
        hv_low  = pitch_space.encode(np.array([0.0]))
        hv_high = pitch_space.encode(np.array([1.0]))
        assert not np.array_equal(hv_low, hv_high)


# ── ★ Convexity (Gärdenfors' axiom) ──────────────────────────────────────────

class TestConvexity:

    def test_midpoint_of_same_region_is_in_region(self, pitch_space: ConceptualSpace) -> None:
        """Midpoint of two 'low' instances must remain in the 'low' region."""
        a = np.array([0.0])
        b = np.array([0.12])  # still in 'low' cell (boundary at 0.25)
        mid = ConceptualSpace.midpoint(a, b)
        assert pitch_space.is_in_region(mid, "low")

    def test_midpoint_api(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        mid = ConceptualSpace.midpoint(a, b)
        np.testing.assert_allclose(mid, np.array([0.5, 0.5]))

    def test_alpha_zero_returns_a(self) -> None:
        a = np.array([1.0])
        b = np.array([3.0])
        np.testing.assert_allclose(ConceptualSpace.midpoint(a, b, alpha=0.0), a)

    def test_alpha_one_returns_b(self) -> None:
        a = np.array([1.0])
        b = np.array([3.0])
        np.testing.assert_allclose(ConceptualSpace.midpoint(a, b, alpha=1.0), b)


# ── ★ Noise tolerance ─────────────────────────────────────────────────────────

class TestNoiseTolerance:

    def test_random_bit_flips_do_not_change_decode(self, vsa: VSA, pitch_space: ConceptualSpace) -> None:
        """Flipping 5% of bits in a prototype hypervector must not change its decoded name.

        Noise tolerance holds because decode uses cosine similarity; at D=1 000 a 5%
        flip rate leaves cosine ≈ 0.9, easily higher than any cross-prototype similarity.
        """
        rng = np.random.default_rng(99)
        for name, coord in [("low", np.array([0.0])), ("mid", np.array([0.5])), ("high", np.array([1.0]))]:
            hv = pitch_space.encode(coord).copy()
            # Flip 5% of bits
            n_flip = int(0.05 * SMALL_D)
            flip_idx = rng.choice(SMALL_D, size=n_flip, replace=False)
            hv[flip_idx] *= -1
            assert pitch_space.decode(hv) == name, f"Decode failed for '{name}' after noise"
