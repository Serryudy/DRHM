"""The 52 cetasikas (mental factors) and their coexistence rules (CLAUDE.md §4, M2.5).

Every citta must carry all 7 universal cetasikas (UNIVERSALS). On top of those,
a citta carries a lawful subset of the remaining 45 subject to hard mutual-exclusion
constraints (MUTUAL_EXCLUSIONS). These rules are not soft preferences — they are the
logical structure of mind as defined in the Abhidhammattha Sangaha §2.

Groupings follow the standard Abhidhamma taxonomy:
  7  sabbacittasādhāraṇa  — universal to every citta
  6  pakiṇṇaka            — ethically variable occasionals
  4  universal unwholesome — present in every akusala citta
 10  occasional unwholesome
 25  sobhana               — 19 universal beautiful + 3 abstinences + 2 immeasurables + 1 wisdom

Reference: Nyanatiloka, *Buddhist Dictionary* s.v. cetasika;
           Bhikkhu Bodhi, *A Comprehensive Manual of Abhidhamma* ch. 2.
"""

from __future__ import annotations

import enum


class Cetasika(enum.Enum):
    """All 52 cetasikas as a closed enum.

    Members are grouped in the order established by the Abhidhammattha Sangaha.
    The string value is the Pāli name; use it in logs/traces for readability.
    """

    # ── 7 universal (sabbacittasādhāraṇa) ────────────────────────────────────
    PHASSA = "phassa"              # contact        → SpikeEvent binding
    VEDANA = "vedana"              # feeling        → VedanāType scalar
    SANNA = "sanna"                # perception     → VSA F_i feature encoding
    CETANA = "cetana"              # volition       → Active Inference policy
    EKAGGATA = "ekaggata"          # one-pointedness → FocusRouter
    JIVITINDRIYA = "jivitindriya"  # life faculty   → Bhavanga continuity
    MANASIKARA = "manasikara"      # attention      → MOMENT_ADVERTING

    # ── 6 ethically variable occasionals (pakiṇṇaka) ─────────────────────────
    VITAKKA = "vitakka"        # initial application — lifts mind onto object
    VICARA = "vicara"          # sustained application — keeps examining
    ADHIMOKKHA = "adhimokkha"  # decision / resolve
    VIRIYA = "viriya"          # energy / effort
    PITI = "piti"              # joy / rapture
    CHANDA = "chanda"          # desire-to-do / zeal

    # ── 4 universal unwholesome (present in every akusala citta) ─────────────
    MOHA = "moha"          # delusion / ignorance
    AHIRIKA = "ahirika"    # shamelessness
    ANOTTAPPA = "anottappa"  # moral recklessness
    UDDHACCA = "uddhacca"  # restlessness

    # ── 10 occasional unwholesome ─────────────────────────────────────────────
    LOBHA = "lobha"            # greed / attachment
    DITTHI = "ditthi"          # wrong view
    MANA = "mana"              # conceit
    DOSA = "dosa"              # aversion / hatred
    THINA = "thina"            # sloth
    MIDDHA = "middha"          # torpor
    VICIKICCHA = "vicikiccha"  # doubt
    ISSA = "issa"              # envy
    MACCHARIYA = "macchariya"  # stinginess
    KUKKUCCA = "kukkucca"      # remorse / worry

    # ── 19 universal beautiful (present in every sobhana citta) ──────────────
    SADDHA = "saddha"                    # confidence / faith
    SATI = "sati"                        # mindfulness
    HIRI = "hiri"                        # moral shame
    OTTAPPA = "ottappa"                  # moral dread
    ALOBHA = "alobha"                    # non-greed / generosity
    ADOSA = "adosa"                      # non-hatred / loving-kindness
    TATRAMAJJHATTATA = "tatramajjhattata"  # mental balance / equanimity
    KAYA_PASSADDHI = "kaya_passaddhi"    # tranquillity of mental factors
    CITTA_PASSADDHI = "citta_passaddhi"  # tranquillity of consciousness
    KAYA_LAHUTA = "kaya_lahuta"          # lightness of mental factors
    CITTA_LAHUTA = "citta_lahuta"        # lightness of consciousness
    KAYA_MUDUTA = "kaya_muduta"          # pliancy of mental factors
    CITTA_MUDUTA = "citta_muduta"        # pliancy of consciousness
    KAYA_KAMMANNATA = "kaya_kammannata"  # workableness of mental factors
    CITTA_KAMMANNATA = "citta_kammannata"  # workableness of consciousness
    KAYA_PAGUNNATA = "kaya_pagunnata"    # proficiency of mental factors
    CITTA_PAGUNNATA = "citta_pagunnata"  # proficiency of consciousness
    KAYUJJUKATA = "kayujjukata"          # rectitude of mental factors
    CITTUJJUKATA = "cittujjukata"        # rectitude of consciousness

    # ── 3 abstinences (virati) — arise only when refraining from wrong action ─
    SAMMA_VACA = "samma_vaca"          # abstinence from wrong speech
    SAMMA_KAMMANTA = "samma_kammanta"  # abstinence from wrong action
    SAMMA_AJIVA = "samma_ajiva"        # abstinence from wrong livelihood

    # ── 2 immeasurables (appamaññā) ───────────────────────────────────────────
    KARUNA = "karuna"  # compassion — cannot coexist with muditā in one citta
    MUDITA = "mudita"  # sympathetic joy — cannot coexist with karuṇā

    # ── 1 wisdom faculty ──────────────────────────────────────────────────────
    PANNA = "panna"  # wisdom / discernment — not in every beautiful citta


# ── Universal sets ────────────────────────────────────────────────────────────

UNIVERSALS: frozenset[Cetasika] = frozenset({
    Cetasika.PHASSA, Cetasika.VEDANA, Cetasika.SANNA, Cetasika.CETANA,
    Cetasika.EKAGGATA, Cetasika.JIVITINDRIYA, Cetasika.MANASIKARA,
})

UNIVERSAL_UNWHOLESOME: frozenset[Cetasika] = frozenset({
    Cetasika.MOHA, Cetasika.AHIRIKA, Cetasika.ANOTTAPPA, Cetasika.UDDHACCA,
})

UNIVERSAL_BEAUTIFUL: frozenset[Cetasika] = frozenset({
    Cetasika.SADDHA, Cetasika.SATI, Cetasika.HIRI, Cetasika.OTTAPPA,
    Cetasika.ALOBHA, Cetasika.ADOSA, Cetasika.TATRAMAJJHATTATA,
    Cetasika.KAYA_PASSADDHI, Cetasika.CITTA_PASSADDHI,
    Cetasika.KAYA_LAHUTA,    Cetasika.CITTA_LAHUTA,
    Cetasika.KAYA_MUDUTA,    Cetasika.CITTA_MUDUTA,
    Cetasika.KAYA_KAMMANNATA, Cetasika.CITTA_KAMMANNATA,
    Cetasika.KAYA_PAGUNNATA,  Cetasika.CITTA_PAGUNNATA,
    Cetasika.KAYUJJUKATA,     Cetasika.CITTUJJUKATA,
})

OCCASIONALS: frozenset[Cetasika] = frozenset({
    Cetasika.VITAKKA, Cetasika.VICARA, Cetasika.ADHIMOKKHA,
    Cetasika.VIRIYA, Cetasika.PITI, Cetasika.CHANDA,
})

OCCASIONAL_UNWHOLESOME: frozenset[Cetasika] = frozenset({
    Cetasika.LOBHA, Cetasika.DITTHI, Cetasika.MANA, Cetasika.DOSA,
    Cetasika.THINA, Cetasika.MIDDHA, Cetasika.VICIKICCHA,
    Cetasika.ISSA, Cetasika.MACCHARIYA, Cetasika.KUKKUCCA,
})

ABSTINENCES: frozenset[Cetasika] = frozenset({
    Cetasika.SAMMA_VACA, Cetasika.SAMMA_KAMMANTA, Cetasika.SAMMA_AJIVA,
})

IMMEASURABLES: frozenset[Cetasika] = frozenset({
    Cetasika.KARUNA, Cetasika.MUDITA,
})

# ── Mutual-exclusion rules ────────────────────────────────────────────────────
# Each entry is a pair (a, b): a and b can NEVER appear in the same citta profile.
# validate_profile() checks all pairs; raise ValueError on violation.

MUTUAL_EXCLUSIONS: list[tuple[Cetasika, Cetasika]] = [
    (Cetasika.LOBHA,  Cetasika.DOSA),      # greed and hatred never coexist
    (Cetasika.DITTHI, Cetasika.MANA),       # wrong view and conceit never coexist
    (Cetasika.KARUNA, Cetasika.MUDITA),     # compassion and sympathetic joy never coexist
]

# Beautiful universals and universal-unwholesome factors never mix.
# Represented as cross-product pairs so validate_profile() is O(n) per citta.
_BEAUTIFUL_X_UNWHOLESOME: list[tuple[Cetasika, Cetasika]] = [
    (b, u) for b in UNIVERSAL_BEAUTIFUL for u in UNIVERSAL_UNWHOLESOME
]

ALL_EXCLUSIONS: list[tuple[Cetasika, Cetasika]] = MUTUAL_EXCLUSIONS + _BEAUTIFUL_X_UNWHOLESOME


def validate_profile(profile: frozenset[Cetasika]) -> None:
    """Raise ValueError if *profile* violates any coexistence rule.

    Call this whenever constructing a CittaType or a live Citta instance so that
    invalid mental states are caught at the boundary, not silently propagated.
    """
    if not UNIVERSALS.issubset(profile):
        missing = UNIVERSALS - profile
        raise ValueError(f"Cetasika profile missing universal(s): {missing}")
    for a, b in ALL_EXCLUSIONS:
        if a in profile and b in profile:
            raise ValueError(
                f"Cetasika coexistence violation: {a.value!r} and {b.value!r} "
                "cannot appear in the same citta."
            )
