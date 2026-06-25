"""Screen capture as digital vision (CLAUDE.md §8.5).

The agent's "eyes" become the pixels of your monitor. Captured frames flow into
the **existing** ``DvsEncoder`` — a static screen produces no spikes; scrolling
or video produces sparse ON/OFF spikes — and through the ``AttentionFrontend``
so only the focused, salient region is processed.

Capture backend (lazy):
  * Preferred (truly event-driven): the X11 DAMAGE extension pushes "this
    rectangle changed" notifications — the OS-level analogue of ``bhavanga-calana``.
  * Fallback: ``mss`` grabs in a thread executor at a modest cadence. Its idle
    CPU is not literally zero (the grab itself costs), but the encoder output
    stays silent on a static screen and capture is off the event loop.

``feed(frame)`` drives the encoder synthetically for tests / non-mss sources.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from drhm import config
from drhm.attention.frontend import AttentionFrontend
from drhm.sensory.events import EventBus, SpikeEvent
from drhm.sensory.vision import DvsEncoder


class ScreenVisionSource:
    """Desktop screen-capture vision source feeding the DVS encoder + attention."""

    def __init__(
        self,
        bus: EventBus,
        encoder: DvsEncoder | None = None,
        attention: AttentionFrontend | None = None,
    ) -> None:
        self.bus = bus
        self.encoder = encoder or DvsEncoder()
        self.attention = attention  # None == no spatial/salience gating
        self._running = False

    def feed(self, frame: np.ndarray) -> list[SpikeEvent]:
        """Encode one screen frame, apply attention, publish survivors."""
        events = self.encoder.encode(frame)
        if self.attention is not None:
            events = self.attention.process(events)
        for event in events:
            self.bus.publish_nowait(event)
        return events

    def update_focus_from_active_window(self) -> None:  # pragma: no cover - needs X11
        """Point the attention focus at the current X11 active window."""
        if self.attention is None:
            return
        try:
            from drhm.sensory.digital._x11 import active_window_rect, screen_size
        except Exception:
            return
        rect = active_window_rect()
        if rect is None:
            self.attention.focus.set_focus(None)
            return
        sw, sh = screen_size()
        x, y, w, h = rect
        self.attention.focus.set_focus_pixels(x, y, w, h, sw, sh)

    async def run(self) -> None:  # pragma: no cover - needs a display
        """Capture frames via mss in an executor and publish DVS spikes."""
        import mss

        self._running = True
        loop = asyncio.get_running_loop()
        period = 1.0 / config.SCREEN_CAPTURE_FPS
        with mss.mss() as sct:
            monitor = sct.monitors[config.SCREEN_MONITOR]

            def grab() -> np.ndarray:
                shot = sct.grab(monitor)
                return np.asarray(shot)[:, :, :3]  # drop alpha

            while self._running:
                frame = await loop.run_in_executor(None, grab)
                self.update_focus_from_active_window()
                events = self.encoder.encode(frame)
                if self.attention is not None:
                    events = self.attention.process(events)
                for event in events:
                    await self.bus.publish(event)
                await asyncio.sleep(period)

    def stop(self) -> None:
        self._running = False


def _now() -> float:
    return time.monotonic()
