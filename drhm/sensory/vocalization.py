"""Vocalization — the mouth: a spike-driven articulatory synthesizer (CLAUDE.md §8 M1).

This is the agent's only *efferent* sensory channel. It consumes motor
``SpikeEvent``s (``Modality.MOTOR``) and decodes their spike train into a
continuous audio waveform, letting the agent "babble" based on internal
generative drive. Each motor channel maps to a carrier frequency; each spike
excites a short tone burst on that carrier. More spikes -> more acoustic energy.

``render_motor_spikes`` is a pure function (numpy only, unit-testable).
``VocalizationSink`` plays the rendered audio via ``sounddevice`` (lazy import).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from drhm import config
from drhm.sensory.events import Modality, SpikeEvent


def channel_frequency(channel: int) -> float:
    """Carrier frequency (Hz) for a motor channel index."""
    return config.VOCAL_BASE_FREQ + channel * config.VOCAL_FREQ_STEP


def render_motor_spikes(
    events: Iterable[SpikeEvent],
    sample_rate: int | None = None,
    duration: float | None = None,
) -> np.ndarray:
    """Render motor spikes to a mono waveform in [-1, 1].

    Spikes are placed on a timeline relative to the earliest event; each excites
    a Hann-windowed tone burst at its channel's carrier frequency. Non-motor
    events are ignored. With no motor spikes the result is silence.
    """
    sr = sample_rate or config.VOCAL_SAMPLE_RATE
    motor = [e for e in events if e.modality == Modality.MOTOR]

    burst_len = max(1, int(sr * config.VOCAL_BURST_MS / 1000.0))
    if not motor:
        n = int(sr * duration) if duration else 0
        return np.zeros(n, dtype=np.float64)

    t0 = min(e.timestamp for e in motor)
    span = max(e.timestamp for e in motor) - t0
    total = int(sr * (duration if duration is not None else span)) + burst_len
    out = np.zeros(total, dtype=np.float64)

    window = np.hanning(burst_len)
    t = np.arange(burst_len) / sr
    for event in motor:
        start = int((event.timestamp - t0) * sr)
        freq = channel_frequency(event.channel)
        burst = np.sin(2.0 * np.pi * freq * t) * window
        out[start : start + burst_len] += burst

    peak = np.abs(out).max()
    if peak > 1.0:
        out /= peak
    return out


class VocalizationSink:
    """Consume motor spikes off the bus and vocalize them."""

    def __init__(self, subscription, sample_rate: int | None = None) -> None:
        self.subscription = subscription
        self.sample_rate = sample_rate or config.VOCAL_SAMPLE_RATE

    async def run(self) -> None:  # pragma: no cover - needs hw
        """Drain motor spikes and play short bursts as they arrive."""
        import sounddevice as sd

        async for event in self.subscription:
            if event.modality != Modality.MOTOR:
                continue
            waveform = render_motor_spikes([event], self.sample_rate)
            sd.play(waveform, self.sample_rate)
