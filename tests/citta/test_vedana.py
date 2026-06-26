"""Tests for vedanā categorisation (M2.5 / M4.5 acceptance)."""

import pytest

from drhm.citta.vedana import VedanāType, categorise, PLEASANT_VEDANA, PAINFUL_VEDANA, NEUTRAL_VEDANA


def test_high_positive_is_somanassa() -> None:
    assert categorise(0.9) == VedanāType.SOMANASSA


def test_low_negative_is_domanassa() -> None:
    assert categorise(-0.9) == VedanāType.DOMANASSA


def test_near_zero_is_upekkha() -> None:
    assert categorise(0.0) == VedanāType.UPEKKHA


def test_at_pleasant_threshold_is_somanassa() -> None:
    from drhm import config
    assert categorise(config.VEDANA_PLEASANT_THRESHOLD) == VedanāType.SOMANASSA


def test_just_below_pleasant_threshold_is_upekkha() -> None:
    from drhm import config
    assert categorise(config.VEDANA_PLEASANT_THRESHOLD - 0.01) == VedanāType.UPEKKHA


def test_at_painful_threshold_is_domanassa() -> None:
    from drhm import config
    assert categorise(config.VEDANA_PAINFUL_THRESHOLD) == VedanāType.DOMANASSA


def test_just_above_painful_threshold_is_upekkha() -> None:
    from drhm import config
    assert categorise(config.VEDANA_PAINFUL_THRESHOLD + 0.01) == VedanāType.UPEKKHA


def test_bodily_positive_is_sukha() -> None:
    assert categorise(0.9, bodily=True) == VedanāType.SUKHA


def test_bodily_negative_is_dukkha() -> None:
    assert categorise(-0.9, bodily=True) == VedanāType.DUKKHA


def test_bodily_neutral_is_upekkha() -> None:
    assert categorise(0.0, bodily=True) == VedanāType.UPEKKHA


def test_clamp_above_one() -> None:
    assert categorise(99.0) == VedanāType.SOMANASSA


def test_clamp_below_minus_one() -> None:
    assert categorise(-99.0) == VedanāType.DOMANASSA


def test_pleasant_vedana_set_contents() -> None:
    assert VedanāType.SOMANASSA in PLEASANT_VEDANA
    assert VedanāType.SUKHA in PLEASANT_VEDANA


def test_painful_vedana_set_contents() -> None:
    assert VedanāType.DOMANASSA in PAINFUL_VEDANA
    assert VedanāType.DUKKHA in PAINFUL_VEDANA


def test_neutral_vedana_set_contents() -> None:
    assert VedanāType.UPEKKHA in NEUTRAL_VEDANA


def test_vedana_sets_are_disjoint() -> None:
    assert PLEASANT_VEDANA.isdisjoint(PAINFUL_VEDANA)
    assert PLEASANT_VEDANA.isdisjoint(NEUTRAL_VEDANA)
    assert PAINFUL_VEDANA.isdisjoint(NEUTRAL_VEDANA)


def test_five_vedana_types_total() -> None:
    assert len(VedanāType) == 5
