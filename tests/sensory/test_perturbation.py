"""★ The Perturbation Test — canonical M1 acceptance gate (CLAUDE.md §8, Phase 1).

The doctrine: the system rests in a near-zero-energy state and is driven out of
it by *physical* perturbation, never by polling (docs §2.2, §3.2; CLAUDE.md §6.1).

We verify the two halves:
  1. Quiescence — with no stimulus, the consumer awaits the bus, the event loop
     burns negligible CPU, and zero spikes are produced.
  2. Perturbation — a sudden stimulus produces a spike within a bounded latency
     and nothing is dropped.

This is the fast CI form. The long-form 24h vigil is ``test_perturbation_long``
below, marked ``slow`` and deselected by default.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from drhm.sensory.audition import CochleaAuditionSource, CochleaEncoder
from drhm.sensory.events import EventBus
from drhm.sensory.vision import DvsVisionSource

IDLE_WINDOW_S = 0.5
# CPU budget for an idle window. A busy-spin polling loop would consume CPU
# time close to the wall-clock window; an await-blocked consumer consumes almost
# none. The generous ceiling still catches a spin loop.
IDLE_CPU_CEILING_S = 0.10
PERTURBATION_LATENCY_S = 0.5


async def test_quiescence_burns_no_cpu_and_emits_no_spikes():
    """Half 1: an idle substrate is silent and (near) free."""
    bus = EventBus(maxsize=0)
    sub = bus.subscribe()

    received: list = []

    async def consumer():
        async for event in sub:
            received.append(event)

    task = asyncio.create_task(consumer())
    cpu_start = time.process_time()
    await asyncio.sleep(IDLE_WINDOW_S)  # no source publishes anything
    cpu_used = time.process_time() - cpu_start

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert received == [], "an unperturbed substrate must emit no spikes"
    assert bus.published == 0
    assert cpu_used < IDLE_CPU_CEILING_S, (
        f"idle consumer used {cpu_used:.3f}s CPU over {IDLE_WINDOW_S}s wall — "
        "this looks like a polling/busy-spin loop"
    )


async def test_vision_flash_perturbs_within_latency():
    """Half 2 (vision): a light flash spikes the substrate promptly, no drops."""
    bus = EventBus(maxsize=0)
    sub = bus.subscribe()
    source = DvsVisionSource(bus)

    # Prime on darkness, then idle briefly: no spikes yet.
    source.feed(np.zeros((32, 32), dtype=np.float64))
    await asyncio.sleep(0.05)
    assert bus.published == 0

    # Sudden flash.
    t_flash = time.monotonic()
    source.feed(np.full((32, 32), 255.0, dtype=np.float64))

    event = await asyncio.wait_for(sub.get(), timeout=PERTURBATION_LATENCY_S)
    latency = time.monotonic() - t_flash
    assert latency < PERTURBATION_LATENCY_S
    assert event.payload == 1.0  # ON spike
    assert bus.dropped == 0


async def test_audition_onset_perturbs_within_latency():
    """Half 2 (audition): a sudden tone spikes the substrate promptly."""
    bus = EventBus(maxsize=0)
    sub = bus.subscribe()
    source = CochleaAuditionSource(bus, CochleaEncoder())

    # Silence produces nothing.
    source.feed(np.zeros(512))
    await asyncio.sleep(0.05)
    assert bus.published == 0

    # Sudden 1 kHz onset.
    sr = source.encoder.sample_rate
    t = np.arange(2048) / sr
    source.feed(np.sin(2.0 * np.pi * 1000.0 * t))

    event = await asyncio.wait_for(sub.get(), timeout=PERTURBATION_LATENCY_S)
    assert event.payload > 0.0
    assert bus.dropped == 0


import pytest  # noqa: E402


@pytest.mark.slow
async def test_perturbation_long():
    """Long-form vigil (docs: 24h dark/quiet room).

    Run with ``-m slow``. Here we scale the window to a few seconds for CI
    feasibility; bump ``window`` for a real soak. The assertion is the same:
    idle CPU stays negligible and no spurious spikes appear, then a late
    stimulus still perturbs the substrate.
    """
    window = 3.0
    bus = EventBus(maxsize=0)
    sub = bus.subscribe()
    source = DvsVisionSource(bus)
    source.feed(np.zeros((32, 32)))  # prime

    cpu_start = time.process_time()
    await asyncio.sleep(window)
    cpu_used = time.process_time() - cpu_start
    assert bus.published == 0
    assert cpu_used < IDLE_CPU_CEILING_S * (window / IDLE_WINDOW_S)

    source.feed(np.full((32, 32), 255.0))
    event = await asyncio.wait_for(sub.get(), timeout=PERTURBATION_LATENCY_S)
    assert event.payload == 1.0
