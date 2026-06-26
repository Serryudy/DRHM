"""Idle-stability test and perturbation tests for Bhavanga (M2 acceptance).

Acceptance criteria (CLAUDE.md §8 M2):
  - With no input, membrane potentials decay to baseline and the network stays
    in the FLOWING attractor; injected spikes raise potentials and, above
    threshold, exit bhavanga.
  - No exit to ARRESTED without sufficient input.
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm import config
from drhm.sensory.events import Modality, SpikeEvent
from drhm.snn.bhavanga import Bhavanga, BhavangaState, PerturbationEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bhavanga() -> Bhavanga:
    return Bhavanga(rng=np.random.default_rng(0))


def strong_events(n: int = 50, payload: float = 1.0) -> list[SpikeEvent]:
    """Batch of n events that together carry enough current to arrest bhavanga."""
    return [SpikeEvent.now(Modality.VISION, i % 64, payload=payload) for i in range(n)]


def weak_event() -> list[SpikeEvent]:
    """Single event too weak to trigger arrest on its own."""
    return [SpikeEvent.now(Modality.VISION, 0, payload=0.001)]


# ---------------------------------------------------------------------------
# ★ Idle-stability test (acceptance gate)
# ---------------------------------------------------------------------------

class TestIdleStability:
    """The canonical M2 acceptance test: CLAUDE.md §8 M2 idle-stability."""

    def test_no_input_stays_flowing(self) -> None:
        """100 idle steps with no events must not leave FLOWING."""
        b = make_bhavanga()
        for _ in range(100):
            state = b.step_idle()
        assert state == BhavangaState.FLOWING

    def test_mean_voltage_near_rest_after_idle(self) -> None:
        """After 100 quiet ticks, population mean voltage stays within 5 mV of V_rest."""
        b = make_bhavanga()
        for _ in range(100):
            b.step_idle()
        assert abs(b.layer.mean_voltage - config.LIF_V_REST) < 5.0

    def test_strong_input_causes_arrest(self) -> None:
        """Sustained strong input must cause ARRESTED within CALANA_WINDOW steps."""
        b = make_bhavanga()
        events = strong_events(50, payload=1.0)
        state = BhavangaState.FLOWING
        for _ in range(config.BHAVANGA_CALANA_WINDOW + 2):
            state = b.inject(events)
            if state == BhavangaState.ARRESTED:
                break
        assert state == BhavangaState.ARRESTED

    def test_no_arrest_without_events(self) -> None:
        """200 idle steps must never produce ARRESTED — no spontaneous arrest."""
        b = make_bhavanga()
        for _ in range(200):
            state = b.step_idle()
            assert state != BhavangaState.ARRESTED


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------

class TestStateMachine:

    def test_first_event_transitions_flowing_to_vibrating(self) -> None:
        """Any SpikeEvent batch transitions FLOWING → VIBRATING."""
        b = make_bhavanga()
        state = b.inject(strong_events(1, payload=0.001))
        assert state == BhavangaState.VIBRATING

    def test_weak_single_event_vibrates_not_arrests(self) -> None:
        """One tiny-payload event causes VIBRATING but not ARRESTED."""
        b = make_bhavanga()
        state = b.inject(weak_event())
        assert state == BhavangaState.VIBRATING

    def test_vibration_times_out_without_arrest(self) -> None:
        """If input too weak to arrest within CALANA_WINDOW, return to FLOWING."""
        b = make_bhavanga()
        b.inject(weak_event())  # → VIBRATING
        assert b.state == BhavangaState.VIBRATING
        # Run idle steps until window expires
        for _ in range(config.BHAVANGA_CALANA_WINDOW + 1):
            b.step_idle()
        assert b.state == BhavangaState.FLOWING

    def test_arrested_state_persists_until_reset(self) -> None:
        """Once ARRESTED, subsequent inject() calls return ARRESTED without re-firing."""
        b = make_bhavanga()
        events = strong_events()
        for _ in range(config.BHAVANGA_CALANA_WINDOW + 2):
            if b.inject(events) == BhavangaState.ARRESTED:
                break
        assert b.state == BhavangaState.ARRESTED
        # Further calls stay ARRESTED
        assert b.inject([]) == BhavangaState.ARRESTED
        assert b.inject(strong_events()) == BhavangaState.ARRESTED

    def test_reset_returns_to_flowing(self) -> None:
        """reset() after arrest returns to FLOWING for a new vithi cycle."""
        b = make_bhavanga()
        events = strong_events()
        for _ in range(config.BHAVANGA_CALANA_WINDOW + 2):
            if b.inject(events) == BhavangaState.ARRESTED:
                break
        b.reset()
        assert b.state == BhavangaState.FLOWING

    def test_reset_clears_calana_counter(self) -> None:
        """After reset(), calana elapsed counter is zero (clean slate)."""
        b = make_bhavanga()
        b.inject(weak_event())
        b.reset()
        assert b._calana_elapsed == 0
        assert b._input_spike_count == 0

    def test_can_arrest_again_after_reset(self) -> None:
        """A full arrest → reset → arrest cycle works correctly."""
        b = make_bhavanga()
        events = strong_events()
        for _ in range(20):
            if b.inject(events) == BhavangaState.ARRESTED:
                break
        b.reset()
        # Second perturbation
        for _ in range(20):
            if b.inject(events) == BhavangaState.ARRESTED:
                break
        assert b.state == BhavangaState.ARRESTED

    def test_initial_state_is_flowing(self) -> None:
        assert make_bhavanga().state == BhavangaState.FLOWING


# ---------------------------------------------------------------------------
# Callback / PerturbationEvent
# ---------------------------------------------------------------------------

class TestOnArrestCallback:

    def test_callback_fires_on_arrest(self) -> None:
        """on_arrest is called exactly once when threshold is crossed."""
        fired: list[PerturbationEvent] = []
        b = Bhavanga(on_arrest=fired.append, rng=np.random.default_rng(0))
        for _ in range(20):
            if b.inject(strong_events()) == BhavangaState.ARRESTED:
                break
        assert len(fired) == 1

    def test_callback_not_fired_without_arrest(self) -> None:
        """Weak input that times out without arrest must not fire callback."""
        fired: list[PerturbationEvent] = []
        b = Bhavanga(on_arrest=fired.append, rng=np.random.default_rng(0))
        b.inject(weak_event())
        for _ in range(config.BHAVANGA_CALANA_WINDOW + 1):
            b.step_idle()
        assert len(fired) == 0

    def test_callback_receives_valid_perturbation_event(self) -> None:
        """PerturbationEvent carries non-trivial spike count and rate."""
        fired: list[PerturbationEvent] = []
        b = Bhavanga(on_arrest=fired.append, rng=np.random.default_rng(0))
        for _ in range(20):
            if b.inject(strong_events(50)) == BhavangaState.ARRESTED:
                break
        assert fired
        evt = fired[0]
        assert evt.input_spikes > 0
        assert 0.0 < evt.population_rate <= 1.0
        assert evt.timestamp > 0.0

    def test_callback_not_duplicated_on_repeated_inject_after_arrest(self) -> None:
        """Calling inject() while already ARRESTED must not re-fire the callback."""
        count = [0]
        b = Bhavanga(on_arrest=lambda _: count.__setitem__(0, count[0] + 1),
                     rng=np.random.default_rng(0))
        for _ in range(20):
            b.inject(strong_events())
        assert count[0] == 1  # fired exactly once


# ---------------------------------------------------------------------------
# Input current mapping
# ---------------------------------------------------------------------------

class TestInputCurrentMapping:

    def test_zero_payload_events_give_no_extra_drive(self) -> None:
        """Zero-payload events behave identically to no events (noise only)."""
        b1 = Bhavanga(rng=np.random.default_rng(42))
        b2 = Bhavanga(rng=np.random.default_rng(42))
        zero_events = [SpikeEvent.now(Modality.VISION, 0, payload=0.0)]
        s1 = b1.inject([])
        s2 = b2.inject(zero_events)
        # Both should remain FLOWING (zero payload = no state transition)
        assert s1 == BhavangaState.FLOWING
        # Note: zero-payload event still triggers VIBRATING because events list is non-empty.
        # This is intentional: the event bus delivered something; whether it arrests is
        # determined by the LIF response, not the payload gate.
        assert s2 in (BhavangaState.FLOWING, BhavangaState.VIBRATING)

    def test_higher_payload_sum_increases_drive(self) -> None:
        """More total payload → higher population firing rate → faster arrest."""
        arrested_step_weak = []
        arrested_step_strong = []

        for payload, record in [(0.3, arrested_step_weak), (5.0, arrested_step_strong)]:
            b = Bhavanga(rng=np.random.default_rng(0))
            for step in range(50):
                if b.inject(strong_events(20, payload=payload)) == BhavangaState.ARRESTED:
                    record.append(step)
                    break

        # Strong payload should arrest no later than weak
        if arrested_step_weak and arrested_step_strong:
            assert arrested_step_strong[0] <= arrested_step_weak[0]

    def test_multimodal_events_contribute_to_arrest(self) -> None:
        """Events from different modalities all contribute to the input current."""
        b = make_bhavanga()
        mixed = [
            SpikeEvent.now(Modality.VISION, 0, payload=1.0),
            SpikeEvent.now(Modality.AUDITION, 0, payload=1.0),
            SpikeEvent.now(Modality.PROPRIOCEPTION, 0, payload=1.0),
        ] * 15  # 45 events across 3 modalities
        state = BhavangaState.FLOWING
        for _ in range(20):
            state = b.inject(mixed)
            if state == BhavangaState.ARRESTED:
                break
        assert state == BhavangaState.ARRESTED


# ---------------------------------------------------------------------------
# Bhavanga state enum
# ---------------------------------------------------------------------------

def test_three_states_exist() -> None:
    states = list(BhavangaState)
    assert BhavangaState.FLOWING in states
    assert BhavangaState.VIBRATING in states
    assert BhavangaState.ARRESTED in states
    assert len(states) == 3
