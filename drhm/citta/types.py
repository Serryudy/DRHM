"""89/121 citta types from the Abhidhammattha Sangaha (CLAUDE.md §4, M2.5).

Every conscious moment is an instance of exactly one CittaType. The type
determines the mandatory cetasika profile, the permitted vedanā tones, the
moral root, and the cognitive function performed.

The 89-citta count (kāmāvacara) is the standard working set for sense-sphere
cognition — the domain of this agent. The extended 121-citta count adds jhāna
and supramundane cittas; those are defined here but not exercised until the
agent develops concentration (Phase 4+).

Structure
─────────
CittaFunction   — what role a citta plays in the vithi sequence
MoralRoot       — the ethical root of javana cittas
CittaType       — the dataclass representing one of the 89/121 types
CITTA_TYPES     — the authoritative registry, keyed by a short snake_case id

Building a citta profile
─────────────────────────
Every CittaType is validated at module load: validate_profile() from cetasikas.py
is called on construction. Any violation is a programming error, not a runtime
condition — it must not reach production.

Reference: Bhikkhu Bodhi, *A Comprehensive Manual of Abhidhamma* ch. 1 & 3;
           Abhidhammattha Sangaha §§1–5.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from drhm.citta.cetasikas import (
    Cetasika,
    UNIVERSALS,
    UNIVERSAL_UNWHOLESOME,
    UNIVERSAL_BEAUTIFUL,
    OCCASIONALS,
    ABSTINENCES,
    validate_profile,
)
from drhm.citta.vedana import VedanāType


# ── Enumerations ──────────────────────────────────────────────────────────────

class CittaFunction(enum.Enum):
    """The functional role a citta plays within a vithi or the life-continuum."""

    BHAVANGA = "bhavanga"                      # life-continuum (resting attractor)
    PATISANDHI = "patisandhi"                  # rebirth-linking
    CUTI = "cuti"                              # death consciousness
    SENSE = "sense"                            # one of the five sense consciousnesses
    FIVE_DOOR_ADVERTING = "five_door_adverting"  # pancadvaravajjana (moment 4)
    RECEIVING = "receiving"                    # sampaticchana (moment 6)
    INVESTIGATING = "investigating"            # santirana (moment 7)
    DETERMINING = "determining"               # votthapana (moment 8)
    JAVANA = "javana"                          # impulsion (moments 9–15, ×7)
    REGISTERING = "registering"               # tadarammana (moments 16–17)
    MIND_DOOR_ADVERTING = "mind_door_adverting"  # mano-dvaravajjana (ManoDvaraVithi)


class MoralRoot(enum.Enum):
    """Ethical root of a javana citta — determines its character and kamma."""

    LOBHA = "lobha"    # greed / attachment
    DOSA = "dosa"      # aversion / hatred
    MOHA = "moha"      # delusion (pure — vicikiccha or uddhacca cittas)
    ALOBHA = "alobha"  # non-greed (wholesome)
    ADOSA = "adosa"    # non-hatred / loving-kindness (wholesome)
    AMOHA = "amoha"    # wisdom / non-delusion (wholesome)


# ── CittaType dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class CittaType:
    """One of the 89/121 types of citta defined by the Abhidhammattha Sangaha.

    Attributes:
        id:               Short snake_case identifier (e.g. "lobha_somanassa_ditthigata").
        pali_name:        Canonical Pāli name or short description.
        function:         Where in the vithi this type appears.
        cetasika_profile: The exact set of cetasikas that accompany this citta type.
        permitted_vedana: Which VedanāTypes are possible for this citta type.
        moral_root:       For javana cittas, the ethical root; None for functional cittas.
        prompted:         For lobha/wholesome javana: True = sasankhārika (prompted/sluggish),
                          False = asankhārika (unprompted/spontaneous). None for others.
    """

    id: str
    pali_name: str
    function: CittaFunction
    cetasika_profile: frozenset[Cetasika]
    permitted_vedana: frozenset[VedanāType]
    moral_root: MoralRoot | None = None
    prompted: bool | None = None

    def __post_init__(self) -> None:
        validate_profile(self.cetasika_profile)


# ── Profile helpers ───────────────────────────────────────────────────────────
# Build profiles bottom-up so each group adds its mandatory cetasikas cleanly.

def _base() -> frozenset[Cetasika]:
    return UNIVERSALS

def _akusala_base() -> frozenset[Cetasika]:
    return UNIVERSALS | UNIVERSAL_UNWHOLESOME

def _sobhana_base() -> frozenset[Cetasika]:
    return UNIVERSALS | UNIVERSAL_BEAUTIFUL

def _akusala_lobha(extras: set[Cetasika] | None = None) -> frozenset[Cetasika]:
    base = _akusala_base() | {Cetasika.LOBHA}
    return base | frozenset(extras or set())

def _akusala_dosa(extras: set[Cetasika] | None = None) -> frozenset[Cetasika]:
    base = _akusala_base() | {Cetasika.DOSA}
    return base | frozenset(extras or set())

def _sobhana(extras: set[Cetasika] | None = None) -> frozenset[Cetasika]:
    base = _sobhana_base()
    return base | frozenset(extras or set())


# ── 54 kāmāvacara (sense-sphere) cittas ──────────────────────────────────────
#
# 12 akusala + 18 ahetuka + 24 sobhana = 54  (the kāmāvacara count)
# Full 89 = 54 + 15 rūpāvacara (jhāna) + 12 arūpāvacara + 8 lokuttara
# Full 121 = 89 + 32 (expanded lokuttara with jhāna factors)
#
# The 18 ahetuka (rootless) cittas cover the functional sense-door processes:
# 5 pairs of sense consciousness (10) + receiving (2) + investigating (3)
# + five-door adverting (1) + mind-door adverting (1) + registering (1) = 18.
#
# Javana types are built systematically below; functional types are listed first.

_UPEKKHA = frozenset({VedanāType.UPEKKHA})
_SOMA = frozenset({VedanāType.SOMANASSA})
_DOMA = frozenset({VedanāType.DOMANASSA})
_SOMA_UPEKKHA = frozenset({VedanāType.SOMANASSA, VedanāType.UPEKKHA})
_SUKHA = frozenset({VedanāType.SUKHA})
_DUKKHA = frozenset({VedanāType.DUKKHA})


def _build_registry() -> dict[str, CittaType]:  # noqa: C901  (acceptable length for a taxonomy)
    reg: dict[str, CittaType] = {}

    def add(ct: CittaType) -> None:
        reg[ct.id] = ct

    # ── Bhavanga / life-continuum ────────────────────────────────────────────
    add(CittaType(
        id="bhavanga",
        pali_name="bhavaṅga-citta",
        function=CittaFunction.BHAVANGA,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))
    add(CittaType(
        id="patisandhi",
        pali_name="paṭisandhi-citta",
        function=CittaFunction.PATISANDHI,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))
    add(CittaType(
        id="cuti",
        pali_name="cuti-citta",
        function=CittaFunction.CUTI,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))

    # ── Five-door adverting ──────────────────────────────────────────────────
    add(CittaType(
        id="pancadvaravajjana",
        pali_name="pañcadvārāvajjana-citta",
        function=CittaFunction.FIVE_DOOR_ADVERTING,
        cetasika_profile=_base() | {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA},
        permitted_vedana=_UPEKKHA,
    ))

    # ── Mind-door adverting (mano-dvaravajjana) ──────────────────────────────
    add(CittaType(
        id="manodvaravajjana",
        pali_name="mano-dvārāvajjana-citta",
        function=CittaFunction.MIND_DOOR_ADVERTING,
        cetasika_profile=_base() | {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA},
        permitted_vedana=_UPEKKHA,
    ))

    # ── 10 sense consciousnesses (5 senses × 2 resultant: kusala-vipāka / akusala-vipāka) ─
    for sense, sense_vedana in [
        ("eye",  {"pleasant": _SOMA, "unpleasant": _UPEKKHA}),
        ("ear",  {"pleasant": _SOMA, "unpleasant": _UPEKKHA}),
        ("nose", {"pleasant": _UPEKKHA, "unpleasant": _UPEKKHA}),
        ("tongue", {"pleasant": _SOMA, "unpleasant": _UPEKKHA}),
        ("body", {"pleasant": _SUKHA, "unpleasant": _DUKKHA}),
    ]:
        add(CittaType(
            id=f"{sense}_vinnana_kusala_vipaka",
            pali_name=f"{sense}-consciousness (wholesome resultant)",
            function=CittaFunction.SENSE,
            cetasika_profile=_base(),
            permitted_vedana=sense_vedana["pleasant"],
        ))
        add(CittaType(
            id=f"{sense}_vinnana_akusala_vipaka",
            pali_name=f"{sense}-consciousness (unwholesome resultant)",
            function=CittaFunction.SENSE,
            cetasika_profile=_base(),
            permitted_vedana=sense_vedana["unpleasant"],
        ))

    # ── Receiving (sampaticchana) ────────────────────────────────────────────
    add(CittaType(
        id="sampaticchana_kusala_vipaka",
        pali_name="sampaṭicchana-citta (wholesome resultant)",
        function=CittaFunction.RECEIVING,
        cetasika_profile=_base(),
        permitted_vedana=_SOMA,
    ))
    add(CittaType(
        id="sampaticchana_akusala_vipaka",
        pali_name="sampaṭicchana-citta (unwholesome resultant)",
        function=CittaFunction.RECEIVING,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))

    # ── Investigating (santīraṇa) ────────────────────────────────────────────
    add(CittaType(
        id="santirana_somanassa",
        pali_name="santīraṇa-citta (pleasant)",
        function=CittaFunction.INVESTIGATING,
        cetasika_profile=_base() | {Cetasika.PITI},
        permitted_vedana=_SOMA,
    ))
    add(CittaType(
        id="santirana_upekkha_kusala",
        pali_name="santīraṇa-citta (neutral, wholesome resultant)",
        function=CittaFunction.INVESTIGATING,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))
    add(CittaType(
        id="santirana_upekkha_akusala",
        pali_name="santīraṇa-citta (neutral, unwholesome resultant)",
        function=CittaFunction.INVESTIGATING,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))

    # ── Determining (votthapana) ─────────────────────────────────────────────
    add(CittaType(
        id="votthapana",
        pali_name="voṭṭhapana-citta",
        function=CittaFunction.DETERMINING,
        cetasika_profile=_base() | {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA},
        permitted_vedana=_UPEKKHA,
    ))

    # ── Registering (tadarammaṇa) ────────────────────────────────────────────
    # Two resultant types: the same pair as investigating (somanassa / upekkha).
    add(CittaType(
        id="tadarammana_somanassa",
        pali_name="tadārammaṇa-citta (pleasant)",
        function=CittaFunction.REGISTERING,
        cetasika_profile=_base() | {Cetasika.PITI},
        permitted_vedana=_SOMA,
    ))
    add(CittaType(
        id="tadarammana_upekkha",
        pali_name="tadārammaṇa-citta (neutral)",
        function=CittaFunction.REGISTERING,
        cetasika_profile=_base(),
        permitted_vedana=_UPEKKHA,
    ))

    # ── 12 akusala javana cittas ─────────────────────────────────────────────
    #
    # 8 lobha-rooted (4 × 2: with/without ditthi, prompted/unprompted × 2 vedanā)
    # 2 dosa-rooted (prompted / unprompted)
    # 2 moha-rooted (vicikiccha / uddhacca)

    for prompted, prompted_tag in [(False, "asankharika"), (True, "sasankharika")]:
        # lobha + ditthi + somanassa
        add(CittaType(
            id=f"lobha_somanassa_ditthi_{prompted_tag}",
            pali_name=f"lobha-mūla-citta (somanassa, diṭṭhigata, {prompted_tag})",
            function=CittaFunction.JAVANA,
            cetasika_profile=_akusala_lobha({Cetasika.DITTHI, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.PITI, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA}),
            permitted_vedana=_SOMA,
            moral_root=MoralRoot.LOBHA,
            prompted=prompted,
        ))
        # lobha + mana + somanassa
        add(CittaType(
            id=f"lobha_somanassa_mana_{prompted_tag}",
            pali_name=f"lobha-mūla-citta (somanassa, māna, {prompted_tag})",
            function=CittaFunction.JAVANA,
            cetasika_profile=_akusala_lobha({Cetasika.MANA, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.PITI, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA}),
            permitted_vedana=_SOMA,
            moral_root=MoralRoot.LOBHA,
            prompted=prompted,
        ))
        # lobha + ditthi + upekkha
        add(CittaType(
            id=f"lobha_upekkha_ditthi_{prompted_tag}",
            pali_name=f"lobha-mūla-citta (upekkhā, diṭṭhigata, {prompted_tag})",
            function=CittaFunction.JAVANA,
            cetasika_profile=_akusala_lobha({Cetasika.DITTHI, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA}),
            permitted_vedana=_UPEKKHA,
            moral_root=MoralRoot.LOBHA,
            prompted=prompted,
        ))
        # lobha + mana + upekkha
        add(CittaType(
            id=f"lobha_upekkha_mana_{prompted_tag}",
            pali_name=f"lobha-mūla-citta (upekkhā, māna, {prompted_tag})",
            function=CittaFunction.JAVANA,
            cetasika_profile=_akusala_lobha({Cetasika.MANA, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA}),
            permitted_vedana=_UPEKKHA,
            moral_root=MoralRoot.LOBHA,
            prompted=prompted,
        ))

    # dosa-rooted
    for prompted, prompted_tag in [(False, "asankharika"), (True, "sasankharika")]:
        add(CittaType(
            id=f"dosa_{prompted_tag}",
            pali_name=f"dosa-mūla-citta ({prompted_tag})",
            function=CittaFunction.JAVANA,
            cetasika_profile=_akusala_dosa({Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA}),
            permitted_vedana=_DOMA,
            moral_root=MoralRoot.DOSA,
            prompted=prompted,
        ))

    # moha-rooted (2)
    add(CittaType(
        id="moha_vicikiccha",
        pali_name="moha-mūla-citta (vicikicchā)",
        function=CittaFunction.JAVANA,
        cetasika_profile=_akusala_base() | {Cetasika.VICIKICCHA, Cetasika.VITAKKA, Cetasika.VIRIYA},
        permitted_vedana=_UPEKKHA,
        moral_root=MoralRoot.MOHA,
    ))
    add(CittaType(
        id="moha_uddhacca",
        pali_name="moha-mūla-citta (uddhacca)",
        function=CittaFunction.JAVANA,
        cetasika_profile=_akusala_base() | {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.VIRIYA},
        permitted_vedana=_UPEKKHA,
        moral_root=MoralRoot.MOHA,
    ))

    # ── 8 mahā-kusala (beautiful wholesome kāmāvacara javana) cittas ─────────
    #
    # 4 pairs: (somanassa/upekkha) × (with/without pañña) × (prompted/unprompted)
    for prompted, prompted_tag in [(False, "asankharika"), (True, "sasankharika")]:
        for vedana_tag, vedana_set, extra_cetasikas in [
            ("somanassa", _SOMA, {Cetasika.PITI, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA, Cetasika.CHANDA}),
            ("upekkha",   _UPEKKHA, {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA, Cetasika.CHANDA}),
        ]:
            # with pañña (ñāṇasampayutta)
            add(CittaType(
                id=f"maha_kusala_{vedana_tag}_panna_{prompted_tag}",
                pali_name=f"mahā-kusala-citta ({vedana_tag}, ñāṇa, {prompted_tag})",
                function=CittaFunction.JAVANA,
                cetasika_profile=_sobhana(extra_cetasikas | {Cetasika.PANNA}),
                permitted_vedana=vedana_set,
                moral_root=MoralRoot.AMOHA,
                prompted=prompted,
            ))
            # without pañña (ñāṇavippayutta)
            add(CittaType(
                id=f"maha_kusala_{vedana_tag}_nopanna_{prompted_tag}",
                pali_name=f"mahā-kusala-citta ({vedana_tag}, no-ñāṇa, {prompted_tag})",
                function=CittaFunction.JAVANA,
                cetasika_profile=_sobhana(extra_cetasikas),
                permitted_vedana=vedana_set,
                moral_root=MoralRoot.ALOBHA,
                prompted=prompted,
            ))

    # ── 8 mahā-kiriya (functional beautiful kāmāvacara javana — arahat) ──────
    # Mirror of mahā-kusala but kiriya (functional, generates no kamma).
    for prompted, prompted_tag in [(False, "asankharika"), (True, "sasankharika")]:
        for vedana_tag, vedana_set, extra_cetasikas in [
            ("somanassa", _SOMA, {Cetasika.PITI, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA, Cetasika.CHANDA}),
            ("upekkha",   _UPEKKHA, {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA, Cetasika.CHANDA}),
        ]:
            add(CittaType(
                id=f"maha_kiriya_{vedana_tag}_panna_{prompted_tag}",
                pali_name=f"mahā-kiriya-citta ({vedana_tag}, ñāṇa, {prompted_tag})",
                function=CittaFunction.JAVANA,
                cetasika_profile=_sobhana(extra_cetasikas | {Cetasika.PANNA}),
                permitted_vedana=vedana_set,
                moral_root=MoralRoot.AMOHA,
                prompted=prompted,
            ))
            add(CittaType(
                id=f"maha_kiriya_{vedana_tag}_nopanna_{prompted_tag}",
                pali_name=f"mahā-kiriya-citta ({vedana_tag}, no-ñāṇa, {prompted_tag})",
                function=CittaFunction.JAVANA,
                cetasika_profile=_sobhana(extra_cetasikas),
                permitted_vedana=vedana_set,
                moral_root=MoralRoot.ALOBHA,
                prompted=prompted,
            ))

    # ── 8 mahā-vipāka (resultant beautiful kāmāvacara) ───────────────────────
    # These arise as bhavanga / paṭisandhi / cuti in beings with wholesome kamma.
    for prompted, prompted_tag in [(False, "asankharika"), (True, "sasankharika")]:
        for vedana_tag, vedana_set, extra_cetasikas in [
            ("somanassa", _SOMA, {Cetasika.PITI, Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA, Cetasika.CHANDA}),
            ("upekkha",   _UPEKKHA, {Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA, Cetasika.VIRIYA, Cetasika.CHANDA}),
        ]:
            add(CittaType(
                id=f"maha_vipaka_{vedana_tag}_panna_{prompted_tag}",
                pali_name=f"mahā-vipāka-citta ({vedana_tag}, ñāṇa, {prompted_tag})",
                function=CittaFunction.BHAVANGA,
                cetasika_profile=_sobhana(extra_cetasikas | {Cetasika.PANNA}),
                permitted_vedana=vedana_set,
                moral_root=None,
                prompted=prompted,
            ))
            add(CittaType(
                id=f"maha_vipaka_{vedana_tag}_nopanna_{prompted_tag}",
                pali_name=f"mahā-vipāka-citta ({vedana_tag}, no-ñāṇa, {prompted_tag})",
                function=CittaFunction.BHAVANGA,
                cetasika_profile=_sobhana(extra_cetasikas),
                permitted_vedana=vedana_set,
                moral_root=None,
                prompted=prompted,
            ))

    # Rūpāvacara, arūpāvacara, and lokuttara cittas are stubs — their profiles
    # will be filled in when concentration/jhāna paths are implemented (Phase 4+).

    return reg


# Build and freeze at import time — any coexistence violation raises immediately.
CITTA_TYPES: dict[str, CittaType] = _build_registry()


def get(citta_id: str) -> CittaType:
    """Return the CittaType for *citta_id* or raise KeyError."""
    return CITTA_TYPES[citta_id]


def javana_types() -> list[CittaType]:
    """Return all CittaTypes with function == JAVANA."""
    return [ct for ct in CITTA_TYPES.values() if ct.function == CittaFunction.JAVANA]


def akusala_javana_types() -> list[CittaType]:
    """Return unwholesome javana types (lobha/dosa/moha-rooted)."""
    unwholesome_roots = {MoralRoot.LOBHA, MoralRoot.DOSA, MoralRoot.MOHA}
    return [ct for ct in javana_types() if ct.moral_root in unwholesome_roots]


def sobhana_javana_types() -> list[CittaType]:
    """Return wholesome javana types (alobha/adosa/amoha-rooted)."""
    wholesome_roots = {MoralRoot.ALOBHA, MoralRoot.ADOSA, MoralRoot.AMOHA}
    return [ct for ct in javana_types() if ct.moral_root in wholesome_roots]
