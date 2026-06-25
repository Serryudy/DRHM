"""System-audio loopback as digital hearing (CLAUDE.md §8.5).

The agent's "ears" become the operating system's audio output buffer, so it
hears what the computer plays (the course instructor's voice) rather than
ambient room noise. Captured blocks flow into the **existing** ``CochleaEncoder``.

Backend (lazy): on this PipeWire/PulseAudio system the default speaker's
*monitor* source is the loopback channel; ``soundcard`` opens it directly. The
capture callback runs off the event loop. ``feed(block)`` drives the encoder
synthetically for tests.
"""

from __future__ import annotations

import numpy as np

from drhm import config
from drhm.sensory.audition import CochleaEncoder
from drhm.sensory.events import EventBus, SpikeEvent


class LoopbackAuditionSource:
    """Loopback (monitor) audition source feeding the silicon-cochlea encoder."""

    def __init__(self, bus: EventBus, encoder: CochleaEncoder | None = None) -> None:
        self.bus = bus
        self.encoder = encoder or CochleaEncoder()
        self._running = False

    def feed(self, block: np.ndarray) -> list[SpikeEvent]:
        """Encode one block of loopback audio and publish its spikes."""
        events = self.encoder.encode(block)
        for event in events:
            self.bus.publish_nowait(event)
        return events

    async def run(self) -> None:  # pragma: no cover - needs an audio device
        """Open the default speaker's monitor and publish cochlea spikes."""
        import asyncio

        import soundcard as sc

        loop = asyncio.get_running_loop()
        speaker = sc.default_speaker()
        mic = sc.get_microphone(speaker.name, include_loopback=True)
        self._running = True
        block = config.COCHLEA_BLOCK
        with mic.recorder(samplerate=self.encoder.sample_rate, channels=1) as rec:
            while self._running:
                data = await loop.run_in_executor(None, rec.record, block)
                for event in self.encoder.encode(np.asarray(data).ravel()):
                    await self.bus.publish(event)

    def stop(self) -> None:
        self._running = False
