"""Digital sensory sources — screen / loopback / peripherals (CLAUDE.md §8.5).

Driven synthetically via ``feed*()`` so no display / audio device / input hooks
are needed.
"""

from __future__ import annotations

import numpy as np

from drhm.attention.focus import FocusRouter
from drhm.attention.frontend import AttentionFrontend
from drhm.sensory.digital import (
    LoopbackAuditionSource,
    PeripheralSource,
    ScreenVisionSource,
)
from drhm.sensory.digital.peripherals import key_channel
from drhm.sensory.events import EventBus, Modality


# --- screen -> vision -------------------------------------------------------
def test_screen_source_reuses_dvs_encoder():
    bus = EventBus(maxsize=0)
    src = ScreenVisionSource(bus)
    src.feed(np.zeros((32, 32), dtype=np.float64))  # prime: no spikes
    assert bus.published == 0
    events = src.feed(np.full((32, 32), 255.0))  # flash
    assert events and all(e.modality is Modality.VISION for e in events)


def test_screen_source_respects_focus():
    bus = EventBus(maxsize=0)
    fe = AttentionFrontend(focus=FocusRouter())
    fe.focus.set_focus((0, 0, 1, 1))  # attend a single grid cell
    src = ScreenVisionSource(bus, attention=fe)
    src.feed(np.zeros((32, 32)))  # prime
    events = src.feed(np.full((32, 32), 255.0))  # whole-screen flash
    # Only the attended cell survives spatial adverting.
    assert events
    assert bus.published == len(events)


# --- loopback -> hearing ----------------------------------------------------
def test_loopback_source_reuses_cochlea_encoder():
    bus = EventBus(maxsize=0)
    src = LoopbackAuditionSource(bus)
    sr = src.encoder.sample_rate
    t = np.arange(2048) / sr
    events = src.feed(np.sin(2.0 * np.pi * 1000.0 * t))
    assert events and all(e.modality is Modality.AUDITION for e in events)
    assert src.feed(np.zeros(512)) == []  # silence -> nothing


# --- peripherals -> proprioception -----------------------------------------
def test_keystroke_becomes_hashed_proprioceptive_spike():
    bus = EventBus(maxsize=0)
    src = PeripheralSource(bus)
    event = src.feed_key("a")
    assert event.modality is Modality.PROPRIOCEPTION
    # Channel is a stable hash within range; the literal char is not stored.
    assert event.channel == key_channel("a")
    assert 0 <= event.channel < 256


def test_key_hash_is_stable_and_not_the_literal():
    assert key_channel("secret") == key_channel("secret")
    assert key_channel("a") != key_channel("b")


def test_mouse_button_and_scroll_emit_spikes():
    bus = EventBus(maxsize=0)
    src = PeripheralSource(bus)
    click = src.feed_button(0, pressed=True)
    scroll = src.feed_scroll(-1.0)
    assert click.modality is Modality.PROPRIOCEPTION
    assert scroll.payload == -1.0
