"""SessionRecorder — capture today's usage for tomorrow's brain (CLAUDE.md §8.5)."""

from __future__ import annotations

from drhm.recording.session import SessionRecorder, replay
from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent


def _stream() -> list[SpikeEvent]:
    return [
        SpikeEvent(1.0, Modality.VISION, 7, 1.0, "dvs", ArammanaGrade.ATI_MAHANTA),
        SpikeEvent(1.1, Modality.AUDITION, 3, 4.2, "cochlea", ArammanaGrade.MAHANTA),
        SpikeEvent(1.2, Modality.PROPRIOCEPTION, 42, 1.0, "peripheral", ArammanaGrade.PARITTA),
    ]


def test_roundtrip_preserves_the_stream(tmp_path):
    log = tmp_path / "session.jsonl"
    with SessionRecorder(subscription=None, path=log) as rec:
        for event in _stream():
            assert rec.write(event) is True
    restored = list(replay(log))
    original = _stream()
    assert len(restored) == len(original)
    for got, want in zip(restored, original, strict=True):
        assert got.modality == want.modality
        assert got.channel == want.channel
        assert got.payload == want.payload
        assert got.grade == want.grade
        assert got.timestamp == want.timestamp


def test_pause_suppresses_writes(tmp_path):
    log = tmp_path / "paused.jsonl"
    with SessionRecorder(subscription=None, path=log) as rec:
        rec.pause()
        assert rec.write(_stream()[0]) is False
        rec.resume()
        assert rec.write(_stream()[0]) is True
    assert len(list(replay(log))) == 1


def test_write_before_start_is_noop(tmp_path):
    rec = SessionRecorder(subscription=None, path=tmp_path / "x.jsonl")
    assert rec.write(_stream()[0]) is False
    assert rec.written == 0


async def test_run_drains_subscription(tmp_path):
    from drhm.sensory.events import EventBus

    bus = EventBus(maxsize=0)
    sub = bus.subscribe()
    log = tmp_path / "live.jsonl"
    rec = SessionRecorder(subscription=sub, path=log).start()

    import asyncio

    task = asyncio.create_task(rec.run())
    for event in _stream():
        await bus.publish(event)
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    rec.close()
    assert len(list(replay(log))) == 3
