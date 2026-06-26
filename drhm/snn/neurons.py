"""LIF (Leaky Integrate-and-Fire) neuron population — the SNN substrate.

Architecture decision record (ADR-SNN-001)
─────────────────────────────────────────
Nengo and BindsNET were evaluated. Both were rejected for M2:

  Nengo    — high-level, requires compiled backend (nengo-dl / nengo-ocl). Excellent
              for academic modelling but adds a heavy dependency chain and hides the
              membrane dynamics we need to inspect in tests.
  BindsNET — PyTorch-based, pulls in torch (~2 GB). Good for GPU-accelerated SNN
              research but exceeds the project's "small, auditable, edge-deployable"
              footprint target (CLAUDE.md §5, §8 energy proportionality).

Decision: pure-numpy LIF. The population is small (≤512 neurons for bhavanga), the
dynamics are a single vectorised Euler step, and the code is readable and unit-testable
without a framework. If Phase 4 learning requires GPU acceleration for STDP across a
larger population, BindsNET can be introduced then — quarantined in drhm/learning/.

LIF equation (discrete Euler, dt = config.LIF_DT ms)
─────────────────────────────────────────────────────
  V[t+1] = V[t] + (dt / τ_m) × (−(V[t] − V_rest) + R_m × I[t])

  Spike condition: V[t+1] ≥ V_threshold  →  emit spike, clamp to V_reset, enter refractory.
  Refractory: neuron is clamped at V_reset for T_refractory / dt steps; no integration.

Reference: Dayan & Abbott, *Theoretical Neuroscience* ch. 5; CLAUDE.md §8 M2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drhm import config


@dataclass
class LIFState:
    """Membrane state snapshot for a population of N neurons.

    Attributes:
        v:          Membrane voltages (N,) in mV.
        refractory: Remaining refractory steps per neuron (N,); 0 = active.
    """

    v: np.ndarray
    refractory: np.ndarray  # dtype int, counts down to 0

    @classmethod
    def resting(cls, n: int) -> LIFState:
        """All neurons at rest potential, none refractory."""
        return cls(
            v=np.full(n, config.LIF_V_REST, dtype=float),
            refractory=np.zeros(n, dtype=int),
        )


class LIFLayer:
    """Vectorised LIF population — the elemental neural substrate.

    All arithmetic is batched over N neurons using numpy; one call to
    :meth:`step` advances the population by one :data:`config.LIF_DT` ms tick.

    Args:
        n:   Number of neurons in the population.
        rng: Seeded numpy Generator; defaults to ``config.SEED``.
    """

    def __init__(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n = n
        self._rng = rng or np.random.default_rng(config.SEED)
        self.state = LIFState.resting(n)
        self._refractory_steps = int(config.LIF_T_REFRACTORY / config.LIF_DT)

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(self, current: np.ndarray) -> np.ndarray:
        """Advance one dt tick under external current *current* (N,) [nA].

        Returns:
            Boolean spike mask (N,): True where a neuron fired this tick.
        """
        v = self.state.v
        ref = self.state.refractory
        active = ref <= 0  # neurons not in refractory

        # Euler integration (only for active neurons; refractory stay at reset)
        dv = (config.LIF_DT / config.LIF_TAU_M) * (
            -(v - config.LIF_V_REST) + config.LIF_R_MEMBRANE * current
        )
        v_new = np.where(active, v + dv, config.LIF_V_RESET)

        # Threshold and fire
        spikes: np.ndarray = active & (v_new >= config.LIF_V_THRESHOLD)

        # Reset fired neurons; decrement refractory counters
        v_new = np.where(spikes, config.LIF_V_RESET, v_new)
        ref_new = np.where(
            spikes,
            self._refractory_steps,
            np.maximum(0, ref - 1),
        )

        self.state = LIFState(v=v_new, refractory=ref_new)
        return spikes

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return all neurons to resting state (used when bhavanga resets)."""
        self.state = LIFState.resting(self.n)

    @property
    def mean_voltage(self) -> float:
        """Population-mean membrane voltage (mV)."""
        return float(self.state.v.mean())

    @property
    def active_fraction(self) -> float:
        """Fraction of neurons not currently in refractory."""
        return float((self.state.refractory <= 0).mean())
