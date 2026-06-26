"""Vedanā — feeling / hedonic valence (CLAUDE.md §4, M2.5 / M4.5).

Every citta carries vedanā — the hedonic tone of the experience. In Abhidhamma
five types are distinguished; for kāmāvacara (sense-sphere) cittas the relevant
ones are somanassa (joy), domanassa (grief), and upekkhā (neutral). Bodily
pleasant (sukha) and painful (dukkha) vedanā arise only with body-door cittas.

This module defines:
  VedanāType   — the five discrete categories
  categorise() — bridge from a continuous Conceptual Space valence scalar to
                 a VedanāType (the bridge the plan calls for at M4.5).
                 Thresholds are tunable in config.py.

The continuous scalar itself is produced by drhm/semantics/conceptual_space.py
(Phase 3). This module only performs the float → enum conversion.

Reference: Bhikkhu Bodhi, *A Comprehensive Manual of Abhidhamma* §2 (vedanā);
           Nyanatiloka, *Buddhist Dictionary* s.v. vedanā.
"""

from __future__ import annotations

import enum

from drhm import config


class VedanāType(enum.Enum):
    """The five types of vedanā (feeling/valence).

    The three mental tones (somanassa, domanassa, upekkhā) are the primary axis
    for sense-sphere cognition. Sukha and dukkha are reserved for body-door cittas.
    """

    SOMANASSA = "somanassa"   # pleasant mental feeling — joy, gladness
    DOMANASSA = "domanassa"   # unpleasant mental feeling — grief, displeasure
    UPEKKHA = "upekkha"       # neutral / equanimous feeling
    SUKHA = "sukha"           # bodily pleasant feeling (body-door only)
    DUKKHA = "dukkha"         # bodily painful feeling (body-door only)


def categorise(valence: float, bodily: bool = False) -> VedanāType:
    """Convert a continuous valence scalar in [-1, +1] to a VedanāType.

    Args:
        valence: A float in [-1, +1] produced by the Conceptual Space module.
                 Values outside this range are clamped silently.
        bodily:  True for body-door (touch) cittas; uses sukha/dukkha instead
                 of somanassa/domanassa.

    Returns:
        The discrete VedanāType for this citta.
    """
    v = max(-1.0, min(1.0, valence))
    if bodily:
        if v >= config.VEDANA_PLEASANT_THRESHOLD:
            return VedanāType.SUKHA
        if v <= config.VEDANA_PAINFUL_THRESHOLD:
            return VedanāType.DUKKHA
        return VedanāType.UPEKKHA
    if v >= config.VEDANA_PLEASANT_THRESHOLD:
        return VedanāType.SOMANASSA
    if v <= config.VEDANA_PAINFUL_THRESHOLD:
        return VedanāType.DOMANASSA
    return VedanāType.UPEKKHA


# Convenience sets for type-checking in CittaType definitions.
PLEASANT_VEDANA: frozenset[VedanāType] = frozenset({VedanāType.SOMANASSA, VedanāType.SUKHA})
PAINFUL_VEDANA: frozenset[VedanāType] = frozenset({VedanāType.DOMANASSA, VedanāType.DUKKHA})
NEUTRAL_VEDANA: frozenset[VedanāType] = frozenset({VedanāType.UPEKKHA})
