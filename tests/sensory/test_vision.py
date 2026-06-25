"""DVS vision encoder tests (CLAUDE.md §8 M1 supporting)."""

from __future__ import annotations

import numpy as np

from drhm.sensory.events import Modality
from drhm.sensory.vision import DvsEncoder


def _frame(value: float, shape=(32, 32)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float64)


def test_first_frame_only_primes_reference():
    enc = DvsEncoder(resolution=(8, 8))
    assert enc.encode(_frame(100.0, (8, 8))) == []


def test_static_scene_emits_no_spikes():
    enc = DvsEncoder(resolution=(8, 8))
    enc.encode(_frame(100.0, (8, 8)))  # prime
    for _ in range(10):
        assert enc.encode(_frame(100.0, (8, 8))) == []


def test_brightening_emits_on_spikes():
    enc = DvsEncoder(resolution=(8, 8), contrast_threshold=0.2)
    enc.encode(_frame(10.0, (8, 8)))  # prime at low intensity
    events = enc.encode(_frame(255.0, (8, 8)))  # sudden flash
    assert events, "a large brightness jump must produce spikes"
    assert all(e.modality is Modality.VISION for e in events)
    assert all(e.payload == 1.0 for e in events)  # ON polarity


def test_darkening_emits_off_spikes():
    enc = DvsEncoder(resolution=(8, 8), contrast_threshold=0.2)
    enc.encode(_frame(255.0, (8, 8)))  # prime bright
    events = enc.encode(_frame(10.0, (8, 8)))  # sudden darkening
    assert events
    assert all(e.payload == -1.0 for e in events)  # OFF polarity


def test_output_is_sparse_for_localized_change():
    enc = DvsEncoder(resolution=(8, 8), contrast_threshold=0.2)
    base = _frame(50.0, (8, 8))
    enc.encode(base)
    changed = base.copy()
    changed[0, 0] = 255.0  # one pixel changes
    events = enc.encode(changed)
    assert len(events) == 1
    assert events[0].channel == 0


def test_color_frame_is_reduced_to_intensity():
    enc = DvsEncoder(resolution=(4, 4))
    rgb = np.zeros((4, 4, 3), dtype=np.float64)
    assert enc.encode(rgb) == []  # primes without error on a 3-channel frame
    rgb[:] = 200.0
    assert enc.encode(rgb)  # brightening across channels spikes
