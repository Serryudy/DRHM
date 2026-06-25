"""Spike-stream session recorder (CLAUDE.md §8.5).

Subscribes to the ``EventBus`` and appends each spike to a local JSONL file:
one line per spike with ``timestamp, modality, channel, payload, source, grade``.
The same stream can be ``replay``-ed back into ``SpikeEvent``s once the learning
substrate exists.

Privacy controls: an explicit ``start()`` (opt-in), a ``pause()/resume()`` toggle
for sensitive moments, and a local-only path. Keystrokes arrive already hashed
from the proprioception source.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent, Subscription


class SessionRecorder:
    """Append a bus's spike stream to a local JSONL session log."""

    def __init__(
        self,
        subscription: Subscription,
        path: str | os.PathLike,
    ) -> None:
        self.subscription = subscription
        self.path = Path(os.path.expanduser(str(path)))
        self._fh = None
        self._paused = False
        self.written = 0

    # --- lifecycle / privacy controls --------------------------------------
    def start(self) -> SessionRecorder:
        """Open the log for appending (opt-in)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)  # line-buffered: each \n hits disk
        return self

    def pause(self) -> None:
        """Stop persisting spikes without tearing down the file (privacy)."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> SessionRecorder:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- writing -----------------------------------------------------------
    def write(self, event: SpikeEvent) -> bool:
        """Persist one spike. Returns False if paused/closed (nothing written)."""
        if self._fh is None or self._paused:
            return False
        self._fh.write(
            json.dumps(
                {
                    "t": event.timestamp,
                    "m": int(event.modality),
                    "c": event.channel,
                    "p": event.payload,
                    "s": event.source,
                    "g": int(event.grade),
                }
            )
            + "\n"
        )
        self.written += 1
        return True

    async def run(self) -> None:
        """Drain the subscription into the log until cancelled."""
        async for event in self.subscription:
            self.write(event)


def replay(path: str | os.PathLike) -> Iterator[SpikeEvent]:
    """Yield the ``SpikeEvent``s recorded in a JSONL session log, in order."""
    p = Path(os.path.expanduser(str(path)))
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield SpikeEvent(
                r["t"],
                Modality(r["m"]),
                r["c"],
                r["p"],
                r.get("s", ""),
                ArammanaGrade(r.get("g", int(ArammanaGrade.MAHANTA))),
            )
