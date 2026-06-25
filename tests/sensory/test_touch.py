"""Capacitive tactile encoder tests (CLAUDE.md §8 M1 supporting)."""

from __future__ import annotations

from drhm.sensory.events import Modality
from drhm.sensory.touch import TouchEncoder


def test_subthreshold_contact_emits_nothing():
    enc = TouchEncoder(grid=(8, 8), pressure_threshold=0.1)
    assert enc.encode_contact(0.5, 0.5, pressure=0.01, timestamp=0.0) == []


def test_contact_emits_spikes_on_addressed_cell():
    enc = TouchEncoder(grid=(8, 8), pressure_threshold=0.1)
    events = enc.encode_contact(0.0, 0.0, pressure=0.2, timestamp=0.0)
    assert events
    assert all(e.modality is Modality.TOUCH for e in events)
    assert all(e.channel == 0 for e in events)  # top-left cell


def test_cell_index_maps_corners():
    enc = TouchEncoder(grid=(8, 8))
    assert enc.cell_index(0.0, 0.0) == 0
    assert enc.cell_index(0.99, 0.99) == 63  # bottom-right cell


def test_harder_press_rate_codes_more_spikes():
    enc = TouchEncoder(grid=(8, 8), pressure_threshold=0.1, max_spikes=8)
    light = enc.encode_contact(0.5, 0.5, pressure=0.1, timestamp=0.0)
    enc.reset()
    hard = enc.encode_contact(0.5, 0.5, pressure=0.8, timestamp=0.0)
    assert len(hard) > len(light)


def test_payload_carries_pressure():
    enc = TouchEncoder(grid=(4, 4), pressure_threshold=0.05)
    events = enc.encode_contact(0.5, 0.5, pressure=0.42, timestamp=0.0)
    assert events and all(e.payload == 0.42 for e in events)
