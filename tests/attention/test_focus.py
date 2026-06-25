"""FocusRouter — single-object spatial adverting (CLAUDE.md §8.5)."""

from __future__ import annotations

from drhm.attention.focus import FocusRouter
from drhm.sensory.events import Modality, SpikeEvent


def _vis(channel: int) -> SpikeEvent:
    return SpikeEvent(0.0, Modality.VISION, channel, 1.0)


def test_no_focus_passes_everything():
    router = FocusRouter(resolution=(8, 8))
    events = [_vis(0), _vis(63)]
    assert router.route(events) == events


def test_focus_drops_out_of_region_vision():
    router = FocusRouter(resolution=(8, 8))
    router.set_focus((0, 0, 4, 4))  # top-left quadrant in grid coords
    inside = _vis(0)  # row 0, col 0
    outside = _vis(63)  # row 7, col 7
    routed = router.route([inside, outside])
    assert routed == [inside]


def test_non_vision_passes_through_focus():
    router = FocusRouter(resolution=(8, 8))
    router.set_focus((0, 0, 1, 1))
    audio = SpikeEvent(0.0, Modality.AUDITION, channel=5, payload=2.0)
    proprio = SpikeEvent(0.0, Modality.PROPRIOCEPTION, channel=9, payload=1.0)
    assert router.route([audio, proprio]) == [audio, proprio]


def test_set_focus_pixels_maps_screen_rect_to_grid():
    router = FocusRouter(resolution=(10, 10))
    # Focus the bottom-right quarter of a 1000x1000 screen.
    router.set_focus_pixels(500, 500, 500, 500, screen_w=1000, screen_h=1000)
    assert router.focus == (5, 5, 10, 10)
