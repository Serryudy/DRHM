"""Silicon-cochlea audition encoder tests (CLAUDE.md §8 M1 supporting)."""

from __future__ import annotations

import numpy as np

from drhm.sensory.audition import CochleaEncoder
from drhm.sensory.events import Modality


def _tone(freq: float, sr: int, n: int) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2.0 * np.pi * freq * t)


def test_silence_emits_no_spikes():
    enc = CochleaEncoder()
    assert enc.encode(np.zeros(512)) == []


def test_empty_block_is_safe():
    enc = CochleaEncoder()
    assert enc.encode(np.array([])) == []


def test_tone_fires_the_band_containing_its_frequency():
    sr = 16_000
    enc = CochleaEncoder(sample_rate=sr, n_channels=16, fmin=80.0, fmax=8000.0)
    freq = 1000.0
    events = enc.encode(_tone(freq, sr, 2048))
    assert events, "a pure tone must produce at least one spike"
    assert all(e.modality is Modality.AUDITION for e in events)
    fired = {e.channel for e in events}
    assert enc.band_for_frequency(freq) in fired


def test_distant_bands_stay_quiet_for_a_pure_tone():
    sr = 16_000
    enc = CochleaEncoder(sample_rate=sr, n_channels=16, fmin=80.0, fmax=8000.0)
    freq = 1000.0
    fired = {e.channel for e in enc.encode(_tone(freq, sr, 2048))}
    target = enc.band_for_frequency(freq)
    # No spikes more than one band away from the tone's band.
    assert all(abs(ch - target) <= 1 for ch in fired)


def test_louder_tone_yields_more_band_energy():
    sr = 16_000
    enc = CochleaEncoder(sample_rate=sr)
    quiet = enc.encode(0.1 * _tone(1000.0, sr, 2048))
    loud = enc.encode(1.0 * _tone(1000.0, sr, 2048))
    assert max(e.payload for e in loud) > max(e.payload for e in quiet)
