"""AttentionFrontend — focus + grade + drop-the-futile (CLAUDE.md §8.5)."""

from __future__ import annotations

from drhm.attention.focus import FocusRouter
from drhm.attention.frontend import AttentionFrontend
from drhm.attention.salience import SalienceGate
from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent


def _vis(channel: int) -> SpikeEvent:
    return SpikeEvent(0.0, Modality.VISION, channel, 1.0)


def _frontend() -> AttentionFrontend:
    return AttentionFrontend(
        resolution=(8, 8),
        region_grid=(2, 2),
        focus=FocusRouter((8, 8)),
        salience=SalienceGate(ema_decay=0.6),
    )


def test_novel_batch_survives_and_is_graded():
    fe = _frontend()
    out = fe.process([_vis(0), _vis(1)])  # region (0,0), first sighting
    assert out, "a novel object must survive the front-end"
    assert all(e.grade == ArammanaGrade.ATI_MAHANTA for e in out)


def test_repeated_region_habituates_and_is_dropped():
    fe = _frontend()
    batch = [_vis(0), _vis(1), _vis(8)]  # same region, same pattern each cycle
    survived = [bool(fe.process(batch)) for _ in range(8)]
    assert survived[0] is True
    assert survived[-1] is False  # moghavāra: the futile object is dropped


def test_out_of_focus_vision_is_dropped():
    fe = _frontend()
    fe.focus.set_focus((0, 0, 4, 4))  # attend top-left quadrant only
    out = fe.process([_vis(0), _vis(63)])  # 63 is bottom-right, outside focus
    assert all(e.channel == 0 for e in out)


def test_empty_batch_returns_empty():
    assert _frontend().process([]) == []
