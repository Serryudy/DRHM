"""Salience grading by predictive novelty (CLAUDE.md §8.5).

Abhidhamma grades a sensory object by intensity into four ``ārammaṇa`` grades,
and the grade determines how far the cognitive process runs — down to the
``moghavāra`` ("futile course") where a very-slight object only makes the
life-continuum vibrate and "the object is not known at all" (verified against
abhidhamma.com, *Process of Consciousness and Matter*).

We reproduce that with **predictive novelty**. Each region of the sensory field
carries a running expectation (an EMA of its recent activity *pattern*). The
novelty of the current activity is the normalized prediction error against that
expectation:

    novelty = ||x - x̂||₁ / (||x||₁ + ||x̂||₁)        ∈ [0, 1]

A *repetitive* object (a clock that redraws the same digits, a cursor blinking
between two states) is predicted well -> novelty -> 0 -> ATI_PARITTA -> dropped.
A *rich, changing* object (a language-course video) is predicted poorly each
frame -> novelty stays high -> MAHANTA/ATI_MAHANTA -> processed. This is exactly
Bayesian surprise, and the EMA is the crude precursor of the generative world
model that Active Inference (M4) will install behind the moment-8 gate.

``SalienceGate`` is pure numpy and hardware-free (unit-testable with vectors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drhm import config
from drhm.sensory.events import ArammanaGrade


@dataclass(frozen=True, slots=True)
class SalienceVerdict:
    """The grade of one region's current activity, plus the raw novelty score."""

    grade: ArammanaGrade
    novelty: float

    @property
    def is_futile(self) -> bool:
        """True for ``moghavāra``: the object is dropped, not processed."""
        return self.grade == ArammanaGrade.ATI_PARITTA


class SalienceGate:
    """Per-region predictive-novelty grader (Abhidhamma object-salience).

    Call ``assess(region_key, feature)`` once per cognitive cycle per active
    region. A region with no activity (all-zero feature) returns ``None`` — there
    is no object to grade. The region's expectation is updated *after* scoring,
    so the first sight of a pattern is maximally novel.
    """

    def __init__(
        self,
        ema_decay: float | None = None,
        grade_cuts: tuple[float, float, float] | None = None,
    ) -> None:
        self.ema_decay = config.ATTENTION_EMA_DECAY if ema_decay is None else ema_decay
        self.grade_cuts = grade_cuts or config.ATTENTION_GRADE_CUTS
        self._expectation: dict[object, np.ndarray] = {}

    def reset(self) -> None:
        self._expectation.clear()

    def _grade_for(self, novelty: float) -> ArammanaGrade:
        c0, c1, c2 = self.grade_cuts
        if novelty < c0:
            return ArammanaGrade.ATI_PARITTA
        if novelty < c1:
            return ArammanaGrade.PARITTA
        if novelty < c2:
            return ArammanaGrade.MAHANTA
        return ArammanaGrade.ATI_MAHANTA

    def assess(self, region_key: object, feature) -> SalienceVerdict | None:
        """Grade ``feature`` for ``region_key`` and update its expectation."""
        x = np.asarray(feature, dtype=np.float64).ravel()
        if not np.any(x):
            return None  # no impingement -> no object

        x_hat = self._expectation.get(region_key)
        if x_hat is None:
            x_hat = np.zeros_like(x)

        denom = np.abs(x).sum() + np.abs(x_hat).sum()
        novelty = float(np.abs(x - x_hat).sum() / denom) if denom > 0 else 0.0

        # Habituate: move the expectation toward what was just seen.
        d = self.ema_decay
        self._expectation[region_key] = d * x_hat + (1.0 - d) * x

        return SalienceVerdict(self._grade_for(novelty), novelty)
