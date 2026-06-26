"""Tests for cetasika coexistence rules and universal sets (M2.5 acceptance)."""

import pytest

from drhm.citta.cetasikas import (
    Cetasika,
    UNIVERSALS,
    UNIVERSAL_UNWHOLESOME,
    UNIVERSAL_BEAUTIFUL,
    OCCASIONALS,
    validate_profile,
)


def _valid_base() -> frozenset[Cetasika]:
    return UNIVERSALS


def test_universals_count() -> None:
    assert len(UNIVERSALS) == 7


def test_universal_unwholesome_count() -> None:
    assert len(UNIVERSAL_UNWHOLESOME) == 4


def test_universal_beautiful_count() -> None:
    assert len(UNIVERSAL_BEAUTIFUL) == 19


def test_occasionals_count() -> None:
    assert len(OCCASIONALS) == 6


def test_total_cetasika_count() -> None:
    assert len(Cetasika) == 52


def test_validate_accepts_universals_only() -> None:
    validate_profile(_valid_base())  # must not raise


def test_validate_rejects_missing_universal() -> None:
    incomplete = UNIVERSALS - {Cetasika.PHASSA}
    with pytest.raises(ValueError, match="universal"):
        validate_profile(incomplete)


def test_validate_rejects_lobha_and_dosa_together() -> None:
    profile = _valid_base() | UNIVERSAL_UNWHOLESOME | {Cetasika.LOBHA, Cetasika.DOSA}
    with pytest.raises(ValueError, match="coexistence"):
        validate_profile(profile)


def test_validate_rejects_ditthi_and_mana_together() -> None:
    profile = _valid_base() | UNIVERSAL_UNWHOLESOME | {Cetasika.DITTHI, Cetasika.MANA}
    with pytest.raises(ValueError, match="coexistence"):
        validate_profile(profile)


def test_validate_rejects_karuna_and_mudita_together() -> None:
    profile = _valid_base() | UNIVERSAL_BEAUTIFUL | {Cetasika.KARUNA, Cetasika.MUDITA}
    with pytest.raises(ValueError, match="coexistence"):
        validate_profile(profile)


def test_validate_rejects_beautiful_and_unwholesome_together() -> None:
    # Any beautiful universal paired with any universal unwholesome is illegal.
    profile = _valid_base() | UNIVERSAL_BEAUTIFUL | UNIVERSAL_UNWHOLESOME
    with pytest.raises(ValueError, match="coexistence"):
        validate_profile(profile)


def test_validate_accepts_lobha_without_dosa() -> None:
    profile = _valid_base() | UNIVERSAL_UNWHOLESOME | {Cetasika.LOBHA}
    validate_profile(profile)  # must not raise


def test_validate_accepts_dosa_without_lobha() -> None:
    profile = _valid_base() | UNIVERSAL_UNWHOLESOME | {Cetasika.DOSA}
    validate_profile(profile)  # must not raise


def test_validate_accepts_beautiful_without_unwholesome() -> None:
    profile = _valid_base() | UNIVERSAL_BEAUTIFUL
    validate_profile(profile)  # must not raise


def test_validate_accepts_karuna_without_mudita() -> None:
    profile = _valid_base() | UNIVERSAL_BEAUTIFUL | {Cetasika.KARUNA}
    validate_profile(profile)


def test_validate_accepts_mudita_without_karuna() -> None:
    profile = _valid_base() | UNIVERSAL_BEAUTIFUL | {Cetasika.MUDITA}
    validate_profile(profile)


def test_universals_are_disjoint_from_unwholesome() -> None:
    assert UNIVERSALS.isdisjoint(UNIVERSAL_UNWHOLESOME)


def test_universals_are_disjoint_from_beautiful() -> None:
    assert UNIVERSALS.isdisjoint(UNIVERSAL_BEAUTIFUL)


def test_unwholesome_and_beautiful_are_disjoint() -> None:
    assert UNIVERSAL_UNWHOLESOME.isdisjoint(UNIVERSAL_BEAUTIFUL)
