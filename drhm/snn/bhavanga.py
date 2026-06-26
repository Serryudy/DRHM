"""Bhavanga resting attractor and perturbation detection (moments 1–3).

Abhidhamma account (Blueprint §3.2; CLAUDE.md §3, §10 phase I)
──────────────────────────────────────────────────────────────
The bhavanga ('ground of becoming') is the resting life-continuum — the mind's
default state between active cognitive episodes. It is not nothing; it is a
continuously flowing substrate that maintains the stream of consciousness.

Three perturbation moments precede every citta-vithi:

  1. bhavanga-sota (FLOWING)    — life-continuum flows undisturbed; the LIF
                                   population rests near V_rest, driven only by
                                   background synaptic noise.
  2. bhavanga-calana (VIBRATING)— a sensory impact reaches the mind-door; the
                                   network is 'shaken' but not yet arrested.
                                   In code: the first SpikeEvent batch that
                                   arrives starts the calana window.
  3. bhavanga-upaccheda (ARRESTED)— the life-continuum is 'cut'; the citta-vithi
                                    begins. Triggered when the population firing
                                    rate in the calana window exceeds the arrest
                                    threshold.

If the stimulus is too weak to arrest (pop_rate < BHAVANGA_PERTURB_THRESHOLD
within BHAVANGA_CALANA_WINDOW steps), the vibration decays and bhavanga returns
to FLOWING — the energy-proportionality path: no vithi fires for noise.

This maps to the CLAUDE.md §6.1 invariant: the system is perturbed, never polled.
Idle CPU ≈ 0; the LIF layer is only stepped when inject() is called by the async
event bus (sensory spikes) or step_idle() for background maintenance.

Reference: Blueprint §3.2; Bhikkhu Bodhi, *Comprehensive Manual* ch. 3.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from drhm import config
from drhm.sensory.events import SpikeEvent
from drhm.snn.neurons import LIFLayer


class BhavangaState(enum.Enum):
    """The three bhavanga moments (CLAUDE.md §3)."""

    FLOWING = "flowing"    # moment 1: bhavanga-sota — undisturbed resting state
    VIBRATING = "vibrating"  # moment 2: bhavanga-calana — shaken by sensory input
    ARRESTED = "arrested"  # moment 3: bhavanga-upaccheda — life-continuum cut


@dataclass(frozen=True)
class PerturbationEvent:
    """Emitted once per arrest (moment 3 complete).

    Attributes:
        timestamp:       Monotonic clock at the moment of arrest.
        input_spikes:    Total external SpikeEvents that drove this arrest.
        population_rate: Fraction of neurons that fired in the triggering step.
    """

    timestamp: float
    input_spikes: int
    population_rate: float


class Bhavanga:
    """Resting attractor and moments 1–3 perturbation detector.

    A :class:`~drhm.snn.neurons.LIFLayer` population models the neural substrate
    of bhavanga. Sensory :class:`~drhm.sensory.events.SpikeEvent` objects are
    converted to per-neuron input current and injected into the population.

    State machine
    ─────────────
      FLOWING → (any SpikeEvent arrives)         → VIBRATING
      VIBRATING → (pop_rate ≥ threshold in window) → ARRESTED  → on_arrest()
      VIBRATING → (calana window expires, no arrest) → FLOWING
      ARRESTED  → (citta-vithi calls reset())       → FLOWING

    The :attr:`on_arrest` callback fires synchronously at the moment of arrest;
    it should schedule the citta-vithi without blocking.

    Args:
        on_arrest: Optional callback receiving a :class:`PerturbationEvent`.
        rng:       Seeded numpy Generator; defaults to ``config.SEED``.
    """

    def __init__(
        self,
        on_arrest: Callable[[PerturbationEvent], None] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rng = rng or np.random.default_rng(config.SEED)
        self.layer = LIFLayer(config.BHAVANGA_N_NEURONS, rng=self._rng)
        self.state = BhavangaState.FLOWING
        self.on_arrest = on_arrest

        # Internal calana tracking
        self._calana_elapsed: int = 0      # steps spent in VIBRATING
        self._input_spike_count: int = 0   # total external spikes since calana began

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, events: list[SpikeEvent]) -> BhavangaState:
        """Inject sensory spike events and advance the 3-moment state machine.

        Each call represents one :data:`config.LIF_DT` ms tick. The LIF layer is
        always stepped (even with an empty event list) so that membrane dynamics
        remain realistic — background noise keeps the resting state alive.

        Args:
            events: Zero or more SpikeEvents from the current tick.

        Returns:
            The new :class:`BhavangaState` after processing.
        """
        if self.state == BhavangaState.ARRESTED:
            # Arrested — step the LIF quietly but don't change state until reset()
            self._step_lif(events)
            return self.state

        pop_rate = self._step_lif(events)

        if events:
            self._input_spike_count += len(events)
            if self.state == BhavangaState.FLOWING:
                # First input: transition to bhavanga-calana (moment 2)
                self.state = BhavangaState.VIBRATING
                self._calana_elapsed = 0

        if self.state == BhavangaState.VIBRATING:
            self._calana_elapsed += 1
            if pop_rate >= config.BHAVANGA_PERTURB_THRESHOLD:
                # Threshold crossed: bhavanga-upaccheda (moment 3) — arrest
                self.state = BhavangaState.ARRESTED
                if self.on_arrest is not None:
                    self.on_arrest(
                        PerturbationEvent(
                            timestamp=time.monotonic(),
                            input_spikes=self._input_spike_count,
                            population_rate=pop_rate,
                        )
                    )
            elif self._calana_elapsed >= config.BHAVANGA_CALANA_WINDOW:
                # Window expired without arrest — insufficient perturbation; decay back
                self.state = BhavangaState.FLOWING
                self._calana_elapsed = 0
                self._input_spike_count = 0

        return self.state

    def step_idle(self) -> BhavangaState:
        """Advance one tick with no sensory input (background noise only).

        Call this from the async event bus when the sensory queue is quiet so
        that membrane dynamics stay realistic without triggering a citta-vithi.
        Always returns FLOWING (cannot arrest without external SpikeEvents).
        """
        return self.inject([])

    def reset(self) -> None:
        """Return to FLOWING after a citta-vithi cycle completes.

        The LIF membrane potentials are *not* reset — they decay naturally back
        to V_rest through the leak term, preserving realistic after-spike dynamics.
        """
        self.state = BhavangaState.FLOWING
        self._calana_elapsed = 0
        self._input_spike_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_lif(self, events: list[SpikeEvent]) -> float:
        """Step the LIF layer and return the population firing rate [0, 1]."""
        total_input = sum(abs(e.payload) for e in events) if events else 0.0
        current_per_neuron = (
            total_input * config.BHAVANGA_INPUT_GAIN / config.BHAVANGA_N_NEURONS
        )
        current = np.full(self.layer.n, current_per_neuron)

        # Per-neuron background noise (models spontaneous synaptic activity)
        noise = self._rng.normal(0.0, config.BHAVANGA_NOISE_STD, self.layer.n)

        spikes = self.layer.step(current + noise)
        return float(spikes.sum()) / self.layer.n
