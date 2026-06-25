"""Mouse & keyboard as proprioception (CLAUDE.md §8.5).

If the agent lives inside the computer, the peripherals are how it senses *your*
actions. Tracking clicks and keystrokes lets a later stage learn the causal link
between an action and the screen change it produces — the grounding Active
Inference (M4) needs.

Each input becomes a ``Modality.PROPRIOCEPTION`` spike. **Privacy:** by default a
key is encoded only as a stable hash (a channel id in ``[0, PROPRIOCEPTION_CHANNELS)``),
never the literal character — the agent learns "some key was pressed and it
correlated with this screen change" without the literal text being stored.

Backend (lazy): ``pynput`` global listeners. ``feed_*`` methods drive the source
synthetically for tests.
"""

from __future__ import annotations

import hashlib

from drhm import config
from drhm.sensory.events import EventBus, Modality, SpikeEvent

# Mouse buttons/scroll occupy the top of the channel space; key hashes fill the
# rest, so the two never collide.
_RESERVED = 8


def key_channel(key: str, hash_keys: bool | None = None) -> int:
    """Stable channel id for a key. Hashed (privacy) unless explicitly disabled."""
    hash_keys = config.PROPRIOCEPTION_HASH_KEYS if hash_keys is None else hash_keys
    span = config.PROPRIOCEPTION_CHANNELS - _RESERVED
    if not hash_keys:
        return _RESERVED + (sum(ord(c) for c in key) % span)
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest()
    return _RESERVED + (int.from_bytes(digest, "big") % span)


class PeripheralSource:
    """Turn mouse/keyboard activity into proprioceptive spikes."""

    def __init__(
        self,
        bus: EventBus,
        hash_keys: bool | None = None,
    ) -> None:
        self.bus = bus
        self.hash_keys = config.PROPRIOCEPTION_HASH_KEYS if hash_keys is None else hash_keys
        self._listeners: list = []

    def _emit(self, channel: int, payload: float) -> SpikeEvent:
        event = SpikeEvent.now(Modality.PROPRIOCEPTION, channel, payload, "peripheral")
        self.bus.publish_nowait(event)
        return event

    def feed_key(self, key: str) -> SpikeEvent:
        """A key press -> a hashed proprioceptive spike (privacy-preserving)."""
        return self._emit(key_channel(key, self.hash_keys), 1.0)

    def feed_button(self, button: int, pressed: bool = True) -> SpikeEvent:
        """A mouse button event -> a reserved-channel spike."""
        return self._emit(button % _RESERVED, 1.0 if pressed else -1.0)

    def feed_scroll(self, dy: float) -> SpikeEvent:
        """A scroll tick -> a reserved-channel spike carrying direction."""
        return self._emit(_RESERVED - 1, float(dy))

    async def run(self) -> None:  # pragma: no cover - needs input devices
        """Attach global pynput listeners that publish proprioceptive spikes."""
        import asyncio

        from pynput import keyboard, mouse

        loop = asyncio.get_running_loop()

        def on_press(key) -> None:
            name = getattr(key, "char", None) or str(key)
            loop.call_soon_threadsafe(self.feed_key, name)

        def on_click(x, y, button, pressed) -> None:
            loop.call_soon_threadsafe(self.feed_button, hash(button) & 7, pressed)

        def on_scroll(x, y, dx, dy) -> None:
            loop.call_soon_threadsafe(self.feed_scroll, dy)

        kb = keyboard.Listener(on_press=on_press)
        ms = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        kb.start()
        ms.start()
        self._listeners = [kb, ms]
        try:
            await asyncio.Event().wait()
        finally:
            self.stop()

    def stop(self) -> None:  # pragma: no cover - needs input devices
        for listener in self._listeners:
            listener.stop()
        self._listeners = []
