#!/usr/bin/env python3
"""Phase-1 manual demo: screen + audio + peripherals -> attention -> recorder.

Runs digital sensory sources, applies the attention front-end, and records
every surviving spike to a JSONL session log.  A live stats line prints to
stdout so you can watch spikes arrive in real time.

Usage:
    .venv/bin/python demo.py [options]

Options:
    --session PATH      JSONL output (default ~/.drhm/sessions/demo.jsonl)
    --monitor N         mss monitor index to capture (default 1 = primary).
                        Use --monitor 0 to capture all monitors as one surface,
                        or --monitor 2 for the second display.
    --no-audio          Skip loopback audio source (default: on)
    --no-attention      Skip attention front-end (raw ungraded DVS spikes)
    --replay PATH       Summarise a recorded session and exit

Try scrolling, switching windows, typing, playing audio — proprioception
spikes appear on keys/clicks/scroll; vision spikes on screen change; audition
spikes on sound.  Ctrl-C to stop cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import time
from pathlib import Path

from drhm.recording.session import SessionRecorder, replay
from drhm.sensory.digital.loopback import LoopbackAuditionSource
from drhm.sensory.digital.peripherals import PeripheralSource
from drhm.sensory.digital.screen import ScreenVisionSource
from drhm.sensory.events import ArammanaGrade, EventBus, Modality
from drhm import config


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="~/.drhm/sessions/demo.jsonl", help="JSONL output path")
    p.add_argument(
        "--monitor", type=int, default=None,
        help="mss monitor index: 0=all-combined, 1=primary (default), 2=second display, …",
    )
    p.add_argument("--no-audio", action="store_true", help="Skip loopback audio source")
    p.add_argument("--no-attention", action="store_true", help="Skip attention front-end (raw DVS spikes)")
    p.add_argument("--replay", metavar="PATH", help="Summarise a recorded session and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Replay / inspect mode
# ---------------------------------------------------------------------------

def run_replay(path: str) -> None:
    spikes = list(replay(path))
    if not spikes:
        print("Session is empty.")
        return
    t0 = spikes[0].timestamp
    counts = {m.name: 0 for m in Modality}
    grades = {g.name: 0 for g in ArammanaGrade}
    for e in spikes:
        counts[e.modality.name] += 1
        grades[e.grade.name] += 1
    duration = spikes[-1].timestamp - t0
    print(f"Session:  {path}")
    print(f"Duration: {duration:.1f}s   total spikes: {len(spikes)}")
    print(f"By modality:  {counts}")
    print(f"By grade:     {grades}")
    print(f"\nFirst 5 spikes:")
    for e in spikes[:5]:
        print(f"  t={e.timestamp - t0:7.3f}s  {e.modality.name:<14} ch={e.channel:4d}  "
              f"p={e.payload:+.2f}  grade={e.grade.name}")


# ---------------------------------------------------------------------------
# Live stats display
# ---------------------------------------------------------------------------

async def _stats(sub, recorder, interval: float = 0.5) -> None:
    counts = {m.name: 0 for m in Modality}
    grades = {g.name: 0 for g in ArammanaGrade}
    total = 0
    t0 = time.monotonic()
    last = t0

    async for event in sub:
        total += 1
        counts[event.modality.name] += 1
        grades[event.grade.name] += 1
        now = time.monotonic()
        if now - last >= interval:
            last = now
            elapsed = now - t0
            rate = total / max(elapsed, 1e-3)
            print(
                f"\r[{elapsed:5.0f}s]  spikes={total:6d} ({rate:5.1f}/s)  "
                f"VIS={counts['VISION']:5d}  AUD={counts['AUDITION']:4d}  PROP={counts['PROPRIOCEPTION']:4d}  |  "
                f"PAR={grades['PARITTA']}  MAH={grades['MAHANTA']}  ATI_MAH={grades['ATI_MAHANTA']}  "
                f"rec={recorder.written}",
                end="",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Live demo
# ---------------------------------------------------------------------------

async def run_live(
    session_path: str,
    monitor: int,
    use_attention: bool,
    use_audio: bool,
) -> None:
    bus = EventBus()

    rec_sub = bus.subscribe()
    stats_sub = bus.subscribe()

    session = Path(session_path).expanduser()
    recorder = SessionRecorder(rec_sub, session).start()

    attention = None
    if use_attention:
        from drhm.attention.frontend import AttentionFrontend
        attention = AttentionFrontend()

    screen = ScreenVisionSource(bus, attention=attention)
    # Override the monitor index from config if the user passed --monitor.
    screen._monitor_idx = monitor
    periph = PeripheralSource(bus)
    audio = LoopbackAuditionSource(bus) if use_audio else None

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGTERM,
        lambda: [t.cancel() for t in asyncio.all_tasks(loop)],
    )

    monitor_label = {0: "all monitors combined", 1: "primary"}.get(monitor, f"monitor {monitor}")
    print(f"Recording   ->  {session}")
    print(f"Screen         monitor {monitor} ({monitor_label})")
    print(f"Audio loopback {'ON' if use_audio else 'OFF'}")
    print(f"Attention      {'ON (habituating, grading)' if use_attention else 'OFF (raw DVS)'}")
    print("Keys/clicks/scroll → PROP  |  screen change → VIS  |  sound → AUD\n")

    async def _screen_run():
        """Run mss capture using the chosen monitor index."""
        import warnings
        import mss
        import numpy as np

        _running = True
        period = 1.0 / config.SCREEN_CAPTURE_FPS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sct = mss.mss()
        with sct:
            mon = sct.monitors[screen._monitor_idx]

            def grab() -> np.ndarray:
                shot = sct.grab(mon)
                return np.asarray(shot)[:, :, :3]

            while _running:
                frame = await loop.run_in_executor(None, grab)
                screen.update_focus_from_active_window()
                events = screen.encoder.encode(frame)
                if screen.attention is not None:
                    events = screen.attention.process(events)
                for ev in events:
                    await bus.publish(ev)
                await asyncio.sleep(period)

    async def _record():
        await recorder.run()

    tasks = [
        asyncio.create_task(_screen_run(), name="screen"),
        asyncio.create_task(periph.run(), name="peripherals"),
        asyncio.create_task(_record(), name="recorder"),
        asyncio.create_task(_stats(stats_sub, recorder), name="stats"),
    ]
    if audio is not None:
        tasks.append(asyncio.create_task(audio.run(), name="audio"))

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Stop pynput listener threads first — they hold C-level Xlib state and
        # will segfault if the event loop tears them down while they are blocked.
        periph.stop()
        if audio:
            audio.stop()
        screen.stop()
        # Give the threads a moment to exit before cancelling the asyncio tasks.
        await asyncio.sleep(0.1)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        recorder.close()
        print(f"\n\nStopped.  {recorder.written} spikes written to {session}")
        print(f"Replay later:  .venv/bin/python demo.py --replay {session}")


# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse()
    if args.replay:
        run_replay(args.replay)
        return
    monitor = args.monitor if args.monitor is not None else config.SCREEN_MONITOR
    try:
        asyncio.run(run_live(
            args.session,
            monitor=monitor,
            use_attention=not args.no_attention,
            use_audio=not args.no_audio,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
