"""Vision — Dynamic Vision Sensor (DVS) emulation (CLAUDE.md §8 M1).

A biological retina / DVS pixel fires only when the *log* light intensity at
that pixel changes by more than a contrast threshold, emitting an ON (+1) or
OFF (-1) polarity spike. We reproduce that here: the output is a sparse spike
stream, not dense frames. A static scene produces no events.

``DvsEncoder`` is pure numpy and hardware-free, so it is fully unit-testable by
feeding it synthetic frames. ``DvsVisionSource`` wraps it around a webcam read
performed in an executor thread (lazy OpenCV import) — there is no busy
frame-diff polling loop in any cognitive hot path (CLAUDE.md §6.1).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import numpy as np

from drhm import config
from drhm.sensory.events import EventBus, Modality, SpikeEvent


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Collapse an HxW or HxWxC frame to a 2-D float intensity map."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    elif arr.ndim != 2:
        raise ValueError(f"expected a 2-D or 3-D frame, got shape {arr.shape}")
    return arr


def _resize_nearest(gray: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to ``shape`` (height, width) using numpy only."""
    h, w = shape
    src_h, src_w = gray.shape
    if (src_h, src_w) == (h, w):
        return gray
    rows = (np.arange(h) * src_h // h).clip(0, src_h - 1)
    cols = (np.arange(w) * src_w // w).clip(0, src_w - 1)
    return gray[np.ix_(rows, cols)]


class DvsEncoder:
    """Stateful log-intensity change detector producing ON/OFF polarity spikes.

    The reference frame is updated only at pixels that crossed threshold (as a
    real DVS resets per-pixel), so slow drift accumulates until it trips a spike.
    """

    def __init__(
        self,
        resolution: tuple[int, int] | None = None,
        contrast_threshold: float | None = None,
    ) -> None:
        self.resolution = resolution or config.DVS_RESOLUTION
        self.contrast_threshold = (
            config.DVS_CONTRAST_THRESHOLD
            if contrast_threshold is None
            else contrast_threshold
        )
        self._log_ref: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        h, w = self.resolution
        return h * w

    def reset(self) -> None:
        self._log_ref = None

    def encode(self, frame: np.ndarray, timestamp: float | None = None) -> list[SpikeEvent]:
        """Return the ON/OFF spikes triggered by ``frame``.

        The first frame ever seen only primes the reference and yields no spikes.
        """
        ts = time.monotonic() if timestamp is None else timestamp
        gray = _resize_nearest(_to_gray(frame), self.resolution)
        # log1p keeps the log defined at zero intensity and compresses range as a
        # photoreceptor does.
        log_i = np.log1p(gray)

        if self._log_ref is None:
            self._log_ref = log_i
            return []

        diff = log_i - self._log_ref
        c = self.contrast_threshold
        on = diff > c
        off = diff < -c

        events: list[SpikeEvent] = []
        flat_on = np.flatnonzero(on.ravel())
        flat_off = np.flatnonzero(off.ravel())
        for ch in flat_on.tolist():
            events.append(SpikeEvent(ts, Modality.VISION, int(ch), 1.0, "dvs"))
        for ch in flat_off.tolist():
            events.append(SpikeEvent(ts, Modality.VISION, int(ch), -1.0, "dvs"))

        crossed = on | off
        self._log_ref = np.where(crossed, log_i, self._log_ref)
        return events


class DvsVisionSource:
    """Adapter: drive a ``DvsEncoder`` from a webcam without polling the loop.

    Frame capture (a blocking OpenCV call) runs in a thread executor; the asyncio
    loop only wakes to publish the sparse spikes the encoder returns. OpenCV is
    imported lazily so the package works headless.
    """

    def __init__(self, bus: EventBus, encoder: DvsEncoder | None = None) -> None:
        self.bus = bus
        self.encoder = encoder or DvsEncoder()
        self._running = False

    def feed(self, frame: np.ndarray) -> list[SpikeEvent]:
        """Encode an externally supplied frame and publish its spikes.

        Lets tests and non-webcam frame sources drive vision synthetically.
        """
        events = self.encoder.encode(frame)
        for event in events:
            self.bus.publish_nowait(event)
        return events

    async def run(self) -> None:
        """Capture frames in an executor and publish DVS spikes until stopped."""
        import cv2  # lazy: only needed for live capture

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._running = True
        loop = asyncio.get_running_loop()
        period = 1.0 / config.CAMERA_FPS
        try:
            while self._running:
                ok, frame = await loop.run_in_executor(None, cap.read)
                if ok:
                    for event in self.encoder.encode(frame):
                        await self.bus.publish(event)
                await asyncio.sleep(period)
        finally:
            cap.release()

    def stop(self) -> None:
        self._running = False


def encode_frames(
    frames: Sequence[np.ndarray], encoder: DvsEncoder | None = None
) -> list[list[SpikeEvent]]:
    """Convenience: encode a sequence of frames with one encoder (test helper)."""
    enc = encoder or DvsEncoder()
    return [enc.encode(f) for f in frames]
