"""Generative world model for Active Inference (CLAUDE.md §8 M5).

Theory (Friston 2010, "The free-energy principle: a unified brain theory"):
  The agent maintains beliefs Q(s) over hidden states s that generated sensory
  observations o.  Variational Free Energy (VFE) is an upper bound on surprise:

      F ≥ −log P(o)   (surprise / negative log-evidence)
      F = KL[Q(s) || P(s|o)]  =  prediction error + complexity cost

  In the simplified setting used here the world model is an exponential moving
  average (EMA) over observed X_t hypervectors:

      μ  ←  (1 − η) μ  +  η X_t       (belief update)
      surprise(X_t)  =  1 − cos(X_t, μ)   ∈ [0, 2]

  When the model has no observations yet (n_obs = 0) it returns surprise = 1.0
  (maximum uncertainty — every new stimulus is fully novel).

Connection to the citta-vithi (CLAUDE.md §10 phase IV):
  At moment 8 (votthapana) the JavanaDeterminer (drhm/inference/javana.py) calls
  ``surprise()`` to decide whether to open the EFE gate or return early (energy
  proportionality).  The model is updated via ``observe()`` at moments 16–17
  (tadarammana/registration), mirroring the Abhidhamma's plasticity commit.

Reference: Friston 2010 §2; docs/Architecture of X.md §5 (the model as anusaya).
"""

from __future__ import annotations

import numpy as np

from drhm import config


class GenerativeModel:
    """Exponential-moving-average predictive model over sensory hypervectors.

    The belief μ is a float64 vector of the same dimension D as the agent's
    X_t hypervectors.  For the minimal 1-D case (vedanā scalar only) wrap the
    scalar as ``np.array([vedana_scalar])`` and set ``D=1``.

    Args:
        D:   Dimensionality of the observation space.  Must match the VSA
             dimension when X_t hypervectors are used; can be 1 for scalar use.
        eta: EMA learning rate η ∈ (0, 1].  Small η → slow adaptation (deep
             priors); large η → fast adaptation (shallow priors).

    Attributes:
        D:    Dimensionality.
        eta:  Learning rate.
    """

    def __init__(
        self,
        D: int,
        eta: float = config.GENERATIVE_MODEL_ETA,
    ) -> None:
        if D < 1:
            raise ValueError(f"D must be ≥ 1, got {D}")
        if not 0 < eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        self.D = D
        self.eta = eta
        self._mu: np.ndarray = np.zeros(D, dtype=np.float64)
        self._n_obs: int = 0

    # ── Core ──────────────────────────────────────────────────────────────────

    def surprise(self, x: np.ndarray) -> float:
        """Compute variational free energy ≈ 1 − cos(x, μ) ∈ [0, 2].

        Returns ``1.0`` (maximum uncertainty) when no observations have been
        made yet, or when the belief vector μ is effectively zero.

        Args:
            x: Observation vector, shape (D,).  May be bipolar int8 (X_t from
               VSA) or float64 (conceptual-space coordinates).

        Returns:
            Surprise in [0, 2].  0.0 = perfectly predicted; 2.0 = perfectly
            opposite prediction; 1.0 = orthogonal / no prior.
        """
        if self._n_obs == 0:
            return 1.0
        mu_norm = float(np.linalg.norm(self._mu))
        if mu_norm < 1e-9:
            return 1.0
        x_f = x.astype(np.float64)
        x_norm = float(np.linalg.norm(x_f))
        if x_norm < 1e-9:
            return 1.0
        cosine = float(np.dot(x_f, self._mu)) / (x_norm * mu_norm)
        return float(1.0 - np.clip(cosine, -1.0, 1.0))

    def observe(self, x: np.ndarray) -> float:
        """Update belief with a new observation; return surprise BEFORE update.

        Call this at moments 16–17 (tadarammana) when the citta-vithi commits
        the observation.  Calling at moment 8 instead would short-circuit the
        Abhidhamma temporal ordering (surprises the surpriser).

        Args:
            x: Observation vector, shape (D,).

        Returns:
            Surprise computed against the PRE-update belief (the agent was
            surprised by x before it learned from it).
        """
        s = self.surprise(x)
        self._mu = (1.0 - self.eta) * self._mu + self.eta * x.astype(np.float64)
        self._n_obs += 1
        return s

    def reset(self) -> None:
        """Reset belief to the uninformed prior (zeros, n_obs = 0)."""
        self._mu[:] = 0.0
        self._n_obs = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_obs(self) -> int:
        """Number of observations incorporated into the belief so far."""
        return self._n_obs

    @property
    def belief(self) -> np.ndarray:
        """A copy of the current belief vector μ, shape (D,), dtype float64."""
        return self._mu.copy()
