"""Unit tests for the spike contract and the event bus (CLAUDE.md §8 M1 supporting)."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from drhm.sensory.events import EventBus, Modality, SpikeEvent


def test_spikeevent_orders_by_timestamp_only():
    early = SpikeEvent(1.0, Modality.VISION, channel=99, payload=-1.0)
    late = SpikeEvent(2.0, Modality.AUDITION, channel=0, payload=5.0)
    assert early < late
    # Channel/modality/payload do not participate in ordering.
    a = SpikeEvent(1.0, Modality.VISION, channel=1, payload=1.0)
    b = SpikeEvent(1.0, Modality.TOUCH, channel=500, payload=0.0)
    assert not (a < b) and not (b < a)


def test_spikeevent_is_immutable():
    event = SpikeEvent(1.0, Modality.TOUCH, channel=3, payload=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.channel = 4  # frozen dataclass


def test_now_uses_monotonic_clock():
    e1 = SpikeEvent.now(Modality.MOTOR, channel=0)
    e2 = SpikeEvent.now(Modality.MOTOR, channel=0)
    assert e2.timestamp >= e1.timestamp  # monotonic never goes backwards


async def test_bus_fans_out_to_all_subscribers():
    bus = EventBus(maxsize=0)
    sub_a = bus.subscribe()
    sub_b = bus.subscribe()
    assert bus.subscriber_count == 2

    event = SpikeEvent(1.0, Modality.VISION, channel=7, payload=1.0)
    await bus.publish(event)

    got_a = await asyncio.wait_for(sub_a.get(), timeout=1.0)
    got_b = await asyncio.wait_for(sub_b.get(), timeout=1.0)
    assert got_a is event and got_b is event


async def test_unbounded_bus_never_drops():
    bus = EventBus(maxsize=0)
    sub = bus.subscribe()
    for i in range(1000):
        await bus.publish(SpikeEvent(float(i), Modality.AUDITION, channel=i % 16))
    assert bus.published == 1000
    assert bus.dropped == 0
    assert sub.qsize() == 1000


async def test_bus_preserves_temporal_order_per_subscriber():
    bus = EventBus(maxsize=0)
    sub = bus.subscribe()
    for i in range(50):
        await bus.publish(SpikeEvent(float(i), Modality.TOUCH, channel=0))
    received = [(await sub.get()).timestamp for _ in range(50)]
    assert received == sorted(received)


def test_bounded_bus_counts_drops():
    bus = EventBus(maxsize=2)
    bus.subscribe()
    for i in range(5):
        bus.publish_nowait(SpikeEvent(float(i), Modality.VISION, channel=0))
    assert bus.dropped == 3  # only 2 fit


def test_unsubscribe_removes_queue():
    bus = EventBus()
    sub = bus.subscribe()
    assert bus.subscriber_count == 1
    sub.close()
    assert bus.subscriber_count == 0
