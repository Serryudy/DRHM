"""SalienceGate — the four ārammaṇa grades by predictive novelty (CLAUDE.md §8.5).

The keystone of the digital-embodiment phase: a *repetitive* object (a ticking
clock) must habituate to ATI_PARITTA (moghavāra, dropped), while a *changing*
object (a course video) must keep passing.
"""

from __future__ import annotations

import numpy as np

from drhm.attention.salience import SalienceGate
from drhm.sensory.events import ArammanaGrade


def test_no_activity_is_no_object():
    gate = SalienceGate()
    assert gate.assess("region", np.zeros(16)) is None


def test_first_sighting_is_maximally_novel():
    gate = SalienceGate()
    verdict = gate.assess("region", np.array([1.0, 0.0, 1.0, 0.0]))
    assert verdict is not None
    assert verdict.novelty == 1.0  # nothing predicted it
    assert verdict.grade == ArammanaGrade.ATI_MAHANTA


def test_repetitive_stimulus_habituates_to_moghavara():
    """A clock redrawing the same digits decays to ATI_PARITTA and is dropped."""
    gate = SalienceGate(ema_decay=0.6)
    clock_pattern = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    grades = [gate.assess("clock", clock_pattern).grade for _ in range(8)]
    assert grades[0] == ArammanaGrade.ATI_MAHANTA  # first tick is salient
    assert grades[-1] == ArammanaGrade.ATI_PARITTA  # habituated -> futile
    assert gate.assess("clock", clock_pattern).is_futile


def test_changing_stimulus_keeps_passing():
    """A video region (different pattern each frame) stays above moghavāra."""
    rng = np.random.default_rng(0)
    gate = SalienceGate(ema_decay=0.6)
    futile = 0
    for _ in range(20):
        frame = rng.random(16)
        verdict = gate.assess("video", frame)
        if verdict.is_futile:
            futile += 1
    # Rich, ever-changing content is rarely dismissed as futile.
    assert futile <= 2


def test_regions_habituate_independently():
    gate = SalienceGate(ema_decay=0.6)
    pat = np.array([1.0, 0.0, 1.0])
    for _ in range(8):
        gate.assess("clock", pat)
    # A fresh region seeing the very same pattern is still novel.
    assert gate.assess("new", pat).grade == ArammanaGrade.ATI_MAHANTA
