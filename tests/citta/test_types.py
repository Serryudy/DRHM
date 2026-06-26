"""Tests for the 89/121 citta type registry (M2.5 acceptance)."""

import pytest

from drhm.citta.cetasikas import Cetasika, UNIVERSALS, UNIVERSAL_UNWHOLESOME, UNIVERSAL_BEAUTIFUL
from drhm.citta.types import (
    CITTA_TYPES,
    CittaFunction,
    MoralRoot,
    get,
    javana_types,
    akusala_javana_types,
    sobhana_javana_types,
)


def test_registry_non_empty() -> None:
    assert len(CITTA_TYPES) > 0


def test_every_type_has_all_seven_universals() -> None:
    for cid, ct in CITTA_TYPES.items():
        missing = UNIVERSALS - ct.cetasika_profile
        assert not missing, f"{cid} missing universals: {missing}"


def test_no_type_has_lobha_and_dosa() -> None:
    for cid, ct in CITTA_TYPES.items():
        has_both = Cetasika.LOBHA in ct.cetasika_profile and Cetasika.DOSA in ct.cetasika_profile
        assert not has_both, f"{cid} has both lobha and dosa"


def test_no_type_has_ditthi_and_mana() -> None:
    for cid, ct in CITTA_TYPES.items():
        has_both = Cetasika.DITTHI in ct.cetasika_profile and Cetasika.MANA in ct.cetasika_profile
        assert not has_both, f"{cid} has both ditthi and mana"


def test_no_type_mixes_beautiful_and_unwholesome_universals() -> None:
    for cid, ct in CITTA_TYPES.items():
        has_beautiful = bool(UNIVERSAL_BEAUTIFUL & ct.cetasika_profile)
        has_unwholesome = bool(UNIVERSAL_UNWHOLESOME & ct.cetasika_profile)
        assert not (has_beautiful and has_unwholesome), (
            f"{cid} mixes beautiful and unwholesome universals"
        )


def test_javana_types_all_have_javana_function() -> None:
    for ct in javana_types():
        assert ct.function == CittaFunction.JAVANA, f"{ct.id} has wrong function"


def test_akusala_javana_count() -> None:
    # 8 lobha + 2 dosa + 2 moha = 12
    assert len(akusala_javana_types()) == 12


def test_sobhana_javana_count() -> None:
    # 8 maha-kusala + 8 maha-kiriya = 16
    assert len(sobhana_javana_types()) == 16


def test_lobha_types_have_lobha_cetasika() -> None:
    for ct in akusala_javana_types():
        if ct.moral_root == MoralRoot.LOBHA:
            assert Cetasika.LOBHA in ct.cetasika_profile, f"{ct.id} missing lobha"


def test_dosa_types_have_dosa_cetasika() -> None:
    for ct in akusala_javana_types():
        if ct.moral_root == MoralRoot.DOSA:
            assert Cetasika.DOSA in ct.cetasika_profile, f"{ct.id} missing dosa"


def test_sobhana_types_have_beautiful_universals() -> None:
    for ct in sobhana_javana_types():
        assert UNIVERSAL_BEAUTIFUL.issubset(ct.cetasika_profile), (
            f"{ct.id} missing beautiful universals"
        )


def test_get_known_type() -> None:
    ct = get("bhavanga")
    assert ct.function == CittaFunction.BHAVANGA


def test_get_unknown_type_raises() -> None:
    with pytest.raises(KeyError):
        get("not_a_real_citta")


def test_bhavanga_has_no_moral_root() -> None:
    assert get("bhavanga").moral_root is None


def test_lobha_javana_has_moral_root_lobha() -> None:
    ct = get("lobha_somanassa_ditthi_asankharika")
    assert ct.moral_root == MoralRoot.LOBHA


def test_maha_kusala_with_panna_has_panna() -> None:
    ct = get("maha_kusala_somanassa_panna_asankharika")
    assert Cetasika.PANNA in ct.cetasika_profile


def test_maha_kusala_without_panna_lacks_panna() -> None:
    ct = get("maha_kusala_somanassa_nopanna_asankharika")
    assert Cetasika.PANNA not in ct.cetasika_profile


def test_javana_count_from_config() -> None:
    from drhm import config
    assert config.JAVANA_COUNT == 7
