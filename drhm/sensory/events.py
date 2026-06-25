"""The unified spike contract and the asynchronous event bus (CLAUDE.md §6.2).

All sensory output — vision, audition, touch — and all motor commands are a
single ``SpikeEvent`` type. Downstream cognition consumes one homogeneous spike
stream; modality-specific structure stays inside the sensory layer.

The ``EventBus`` is a fan-out over ``asyncio.Queue``s. Consumers ``await`` the
queue, so an idle system blocks (near-zero CPU) instead of spinning — this is
what makes the Perturbation Test pass (CLAUDE.md §8 M1).
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field

from drhm import config


class Modality(enum.IntEnum):
    """Origin of a spike.

    Vision/audition/touch/proprioception are afferent; motor is efferent.
    PROPRIOCEPTION carries the agent's observation of input-device activity
    (mouse/keyboard) when the agent is digitally embodied (CLAUDE.md §8.5).
    """

    VISION = 0
    AUDITION = 1
    TOUCH = 2
    MOTOR = 3
    PROPRIOCEPTION = 4


class ArammanaGrade(enum.IntEnum):
    """Salience grade of a sensory object, after Abhidhamma ``ārammaṇa`` grades.

    The grade an object receives at the attention front-end determines how far
    the (future) 17-moment cognitive process runs for it (CLAUDE.md §8.5):

      * ATI_PARITTA (very slight) -> ``moghavāra``: object not processed at all;
        the attention gate DROPS these (a ticking clock, a blinking cursor).
      * PARITTA (slight) -> process stops at determining (moment 8); no javana,
        no learning.
      * MAHANTA (great) -> full process through javana.
      * ATI_MAHANTA (very great) -> full process including registration/learning.

    Encoder output defaults to MAHANTA ("process it") until an attention gate
    grades it.
    """

    ATI_PARITTA = 0
    PARITTA = 1
    MAHANTA = 2
    ATI_MAHANTA = 3


@dataclass(frozen=True, order=True, slots=True)
class SpikeEvent:
    """One asynchronous spike.

    Ordering is by ``timestamp`` only (a monotonic clock reading), so events
    sort into temporal order regardless of modality/channel.

    Attributes:
        timestamp: ``time.monotonic()`` reading when the spike was generated.
        modality:  which sense/effector produced it.
        channel:   modality-local address (pixel index, cochlea band, tactile
                   cell, or motor unit).
        payload:   modality-local magnitude — ON/OFF polarity (+1/-1) for
                   vision, band energy for audition, pressure for touch, a
                   motor command for MOTOR.
        source:    optional human-readable tag for debugging/tracing.
        grade:     salience grade assigned by the attention front-end. Defaults
                   to MAHANTA (un-graded events are "process it") and is stamped
                   by the ``SalienceGate`` (CLAUDE.md §8.5).
    """

    timestamp: float
    modality: Modality = field(compare=False)
    channel: int = field(compare=False)
    payload: float = field(default=0.0, compare=False)
    source: str = field(default="", compare=False)
    grade: ArammanaGrade = field(default=ArammanaGrade.MAHANTA, compare=False)

    @classmethod
    def now(
        cls,
        modality: Modality,
        channel: int,
        payload: float = 0.0,
        source: str = "",
        grade: ArammanaGrade = ArammanaGrade.MAHANTA,
    ) -> SpikeEvent:
        """Build an event stamped with the current monotonic clock."""
        return cls(time.monotonic(), modality, channel, payload, source, grade)


class Subscription:
    """A consumer's view of the bus: an awaitable, async-iterable spike stream."""

    __slots__ = ("_bus", "_queue")

    def __init__(self, bus: EventBus, queue: asyncio.Queue) -> None:
        self._bus = bus
        self._queue = queue

    async def get(self) -> SpikeEvent:
        """Await the next spike. Suspends (no busy-spin) while the stream is idle."""
        return await self._queue.get()

    def get_nowait(self) -> SpikeEvent:
        """Return a queued spike or raise ``asyncio.QueueEmpty``."""
        return self._queue.get_nowait()

    def qsize(self) -> int:
        return self._queue.qsize()

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> SpikeEvent:
        return await self._queue.get()

    def close(self) -> None:
        self._bus._unsubscribe(self._queue)


class EventBus:
    """Fan-out spike bus. Each subscriber gets its own queue.

    With an unbounded queue (``maxsize == 0``) ``publish`` never blocks and never
    drops. With a bounded queue, overflow is counted in ``dropped`` so tests can
    assert lossless delivery.
    """

    __slots__ = ("_subscribers", "_maxsize", "published", "dropped")

    def __init__(self, maxsize: int | None = None) -> None:
        self._maxsize = config.EVENT_QUEUE_MAXSIZE if maxsize is None else maxsize
        self._subscribers: list[asyncio.Queue] = []
        self.published: int = 0
        self.dropped: int = 0

    def subscribe(self) -> Subscription:
        """Register a new consumer and return its ``Subscription``."""
        queue: asyncio.Queue = asyncio.Queue(self._maxsize)
        self._subscribers.append(queue)
        return Subscription(self, queue)

    def _unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def publish(self, event: SpikeEvent) -> None:
        """Deliver ``event`` to every subscriber.

        Uses non-blocking puts so a slow consumer can never stall a sensor.
        Async to keep a uniform awaitable API for sources.
        """
        self.publish_nowait(event)

    def publish_nowait(self, event: SpikeEvent) -> None:
        """Synchronous publish, callable from hardware callback threads via
        ``loop.call_soon_threadsafe``."""
        self.published += 1
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped += 1
