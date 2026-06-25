"""Vocalization (articulatory synth) tests (CLAUDE.md §8 M1 supporting)."""

from __future__ import annotations

import numpy as np

from drhm.sensory.events import Modality, SpikeEvent
from drhm.sensory.vocalization import channel_frequency, render_motor_spikes


def test_no_motor_spikes_is_silence():
    waveform = render_motor_spikes([], sample_rate=16_000, duration=0.1)
    assert waveform.shape[0] == 1600
    assert np.allclose(waveform, 0.0)


def test_non_motor_events_are_ignored():
    afferent = [SpikeEvent(0.0, Modality.VISION, channel=0, payload=1.0)]
    assert np.allclose(render_motor_spikes(afferent, duration=0.05), 0.0)


def test_motor_spike_produces_acoustic_energy():
    spike = [SpikeEvent(0.0, Modality.MOTOR, channel=2, payload=1.0)]
    waveform = render_motor_spikes(spike, sample_rate=16_000)
    assert np.abs(waveform).sum() > 0.0


def test_more_spikes_more_energy():
    one = render_motor_spikes(
        [SpikeEvent(0.0, Modality.MOTOR, channel=1)], sample_rate=16_000, duration=0.2
    )
    many = render_motor_spikes(
        [SpikeEvent(i * 0.02, Modality.MOTOR, channel=1) for i in range(5)],
        sample_rate=16_000,
        duration=0.2,
    )
    assert np.abs(many).sum() > np.abs(one).sum()


def test_output_is_bounded():
    spikes = [SpikeEvent(0.0, Modality.MOTOR, channel=c) for c in range(8)]
    waveform = render_motor_spikes(spikes, sample_rate=16_000)
    assert np.abs(waveform).max() <= 1.0


def test_channel_frequency_increases_with_channel():
    assert channel_frequency(0) < channel_frequency(1) < channel_frequency(5)
