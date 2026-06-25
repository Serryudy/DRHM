"""Single-object spatial adverting — ``pañcadvārāvajjana`` (CLAUDE.md §8.5).

Abhidhamma allows exactly one cognitive process, taking one object, at a time
("the sense consciousnesses are not occurring simultaneously" — abhidhamma.com,
verified). For digital vision that means the agent attends to ONE region of the
screen — by default the currently focused window — and ignores the rest, rather
than encoding the whole desktop in parallel.

``FocusRouter`` works in the encoder's coarse grid coordinates. The screen
source converts the focused-window screen rectangle into grid cells and sets it
here; vision spikes outside the focus rectangle are dropped. Non-spatial
modalities (audition, proprioception) pass through untouched.

Geometry comes from a caller-supplied ``focus_provider`` so the router stays
hardware-free and testable; the live X11 active-window lookup is wired in by the
screen source via python-xlib.
"""

from __future__ import annotations

from drhm import config
from drhm.sensory.events import Modality, SpikeEvent


class FocusRouter:
    """Keep only the vision spikes whose pixel lies in the focused grid region."""

    def __init__(self, resolution: tuple[int, int] | None = None) -> None:
        self.resolution = resolution or config.DVS_RESOLUTION
        # Focus rectangle in grid coords: (row0, col0, row1, col1), half-open.
        # None == attend to the whole field (no spatial gating yet).
        self._focus: tuple[int, int, int, int] | None = None

    @property
    def focus(self) -> tuple[int, int, int, int] | None:
        return self._focus

    def set_focus(self, rect: tuple[int, int, int, int] | None) -> None:
        """Set the attended rectangle in grid coords, or ``None`` for all."""
        self._focus = rect

    def set_focus_pixels(
        self, x: int, y: int, w: int, h: int, screen_w: int, screen_h: int
    ) -> None:
        """Convert a focused-window *screen* rectangle into grid coords."""
        gh, gw = self.resolution
        col0 = max(0, min(gw, x * gw // max(1, screen_w)))
        col1 = max(0, min(gw, (x + w) * gw // max(1, screen_w)))
        row0 = max(0, min(gh, y * gh // max(1, screen_h)))
        row1 = max(0, min(gh, (y + h) * gh // max(1, screen_h)))
        self._focus = (row0, col0, max(row0 + 1, row1), max(col0 + 1, col1))

    def _in_focus(self, channel: int) -> bool:
        if self._focus is None:
            return True
        gh, gw = self.resolution
        row, col = divmod(channel, gw)
        row0, col0, row1, col1 = self._focus
        return row0 <= row < row1 and col0 <= col < col1

    def route(self, events: list[SpikeEvent]) -> list[SpikeEvent]:
        """Drop out-of-focus vision spikes; pass everything else through."""
        if self._focus is None:
            return events
        return [
            e for e in events if e.modality != Modality.VISION or self._in_focus(e.channel)
        ]
