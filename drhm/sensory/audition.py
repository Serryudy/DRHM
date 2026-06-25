"""Audition — silicon-cochlea emulation (CLAUDE.md §8 M1).

A silicon cochlea splits sound into log-spaced frequency channels (like the
basilar membrane) and emits a spike on a channel only when that band's acoustic
energy crosses a threshold. We decompose each audio block with an rFFT, bin the
magnitude spectrum into log-spaced bands, and spike the bands that carry a
significant share of the block's energy. Silence yields no spikes.

``CochleaEncoder`` is pure numpy and hardware-free (unit-testable with synthetic
tones). ``CochleaAuditionSource`` wraps it around a ``sounddevice`` input stream
whose callback fires from the audio thread — genuinely event-driven, no polling
(CLAUDE.md §6.1). sounddevice is imported lazily so the package works without an
audio device.
"""

from __future__ import annotations

import time

import numpy as np

from drhm import config
from drhm.sensory.events import EventBus, Modality, SpikeEvent


class CochleaEncoder:
    """Bandpass filterbank → per-channel energy spikes."""

    def __init__(
        self,
        sample_rate: int | None = None,
        n_channels: int | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        energy_fraction: float | None = None,
        silence_floor: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.n_channels = n_channels or config.COCHLEA_CHANNELS
        self.fmin = fmin or config.COCHLEA_FMIN
        self.fmax = fmax or config.COCHLEA_FMAX
        self.energy_fraction = (
            config.COCHLEA_ENERGY_FRACTION
            if energy_fraction is None
            else energy_fraction
        )
        self.silence_floor = (
            config.COCHLEA_SILENCE_FLOOR if silence_floor is None else silence_floor
        )
        # Log-spaced band edges across [fmin, fmax]: n_channels+1 edges.
        self._edges = np.geomspace(self.fmin, self.fmax, self.n_channels + 1)

    def band_for_frequency(self, freq: float) -> int:
        """Return the channel index whose band contains ``freq`` (clamped)."""
        idx = int(np.searchsorted(self._edges, freq, side="right") - 1)
        return max(0, min(self.n_channels - 1, idx))

    def encode(self, block: np.ndarray, timestamp: float | None = None) -> list[SpikeEvent]:
        """Return the spikes triggered by one block of mono audio samples."""
        ts = time.monotonic() if timestamp is None else timestamp
        samples = np.asarray(block, dtype=np.float64).ravel()
        if samples.size == 0:
            return []

        windowed = samples * np.hanning(samples.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        power = spectrum**2
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)

        total = float(power.sum())
        if total <= self.silence_floor:
            return []

        events: list[SpikeEvent] = []
        for ch in range(self.n_channels):
            lo, hi = self._edges[ch], self._edges[ch + 1]
            mask = (freqs >= lo) & (freqs < hi)
            band_power = float(power[mask].sum())
            if band_power / total >= self.energy_fraction:
                events.append(
                    SpikeEvent(ts, Modality.AUDITION, ch, band_power, "cochlea")
                )
        return events


class CochleaAuditionSource:
    """Adapter: drive a ``CochleaEncoder`` from a sounddevice input callback."""

    def __init__(self, bus: EventBus, encoder: CochleaEncoder | None = None) -> None:
        self.bus = bus
        self.encoder = encoder or CochleaEncoder()
        self._stream = None
        self._loop = None

    def feed(self, block: np.ndarray) -> list[SpikeEvent]:
        """Encode an externally supplied audio block and publish its spikes."""
        events = self.encoder.encode(block)
        for event in events:
            self.bus.publish_nowait(event)
        return events

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - needs hw
        # Runs on the PortAudio thread: hand spikes back to the loop thread-safely.
        events = self.encoder.encode(indata[:, 0])
        if self._loop is not None:
            for event in events:
                self._loop.call_soon_threadsafe(self.bus.publish_nowait, event)

    async def run(self) -> None:  # pragma: no cover - needs hw
        """Open the input stream and let its callback publish spikes."""
        import asyncio

        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.encoder.sample_rate,
            channels=1,
            blocksize=config.COCHLEA_BLOCK,
            callback=self._callback,
        )
        self._stream.start()
        try:
            # Park forever; the callback does the work. Cancellation stops us.
            await asyncio.Event().wait()
        finally:
            self.stop()

    def stop(self) -> None:  # pragma: no cover - needs hw
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
