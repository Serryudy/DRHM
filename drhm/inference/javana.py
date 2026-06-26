"""Javana type selection via Expected Free Energy (CLAUDE.md §8 M5).

At moment 8 (votthapana / MOMENT_DETERMINING) the CittaVithi calls the
``DeterminerFn`` to select which javana ``CittaType`` fires 7 identical times.
This module replaces the M3 grade-based stub with an Active Inference engine.

Expected Free Energy (EFE)
──────────────────────────
For each candidate javana policy π_k:

    G(π_k) = pragmatic_cost(π_k, vedanā) − epistemic_gain(π_k, surprise)

Where:
  pragmatic_cost   = (vedanā_scalar − preferred_valence(π_k))²
                     Lower when the policy's vedanā aligns with the percept's
                     affective tone (approach pleasant, avoid painful).

  epistemic_gain   = EFE_EPISTEMIC_WEIGHT × has_pañña(π_k) × surprise
                     Higher surprise → more information to gain → pañña types
                     score a larger epistemic bonus (epistemic foraging).

The policy with minimum EFE is selected.  If surprise < EFE_SURPRISE_GATE,
MAHANTA-grade objects trigger an early-exit (javana_type = None) — energy
proportionality via the moment-8 gate.  ATI_MAHANTA objects bypass the gate.

Abhidhamma alignment
────────────────────
  - Wholesome (kusala) types are preferred when the percept is pleasant and
    surprise is moderate (familiar-pleasant → habitual wholesomeness).
  - Novel percepts attract pañña (wisdom/insight) types: epistemic foraging
    embodies the Abhidhamma ideal that wisdom investigates the unknown.
  - Neutral percepts in equilibrium default to upekkhā + pañña (equanimous
    investigation) — the middle way between craving and aversion.
  - A painful percept does NOT automatically select dosa (hatred): the EFE
    favours upekkhā-pañña (equanimous wisdom) over dosa because dosa has no
    epistemic benefit and its pragmatic cost is non-zero unless the agent
    "wants" to be aversive (which a well-cultivated prior avoids).

Anusaya bias (M6 hook)
──────────────────────
Pass ``anusaya_bias: dict[str, float]`` mapping CittaType id → additive EFE
bias to model latent tendencies from the STDP weight space.  Negative bias
= habitual tendency toward that type; positive = suppression.

Reference: Friston et al. 2017 (active inference and epistemic foraging);
           CLAUDE.md §8 M5; docs/Architecture of X.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drhm import config
from drhm.citta.cetasikas import Cetasika
from drhm.citta.types import CittaType, get
from drhm.citta.vedana import VedanāType
from drhm.inference.active_inference import GenerativeModel
from drhm.sensory.events import ArammanaGrade
from drhm.snn.citta_vithi import DeterminerInput, DeterminerOutput


# ── Policy ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Policy:
    """A candidate javana CittaType with its EFE-scoring parameters.

    Attributes:
        citta_type:         The CittaType that fires 7 times if this policy is chosen.
        preferred_valence:  The affective tone this policy naturally aligns with,
                            in [-1, +1].  Determines pragmatic cost.
        has_panna:          True iff the type carries the Pañña (wisdom) cetasika.
                            Pañña types receive an epistemic bonus proportional to surprise.
    """

    citta_type: CittaType
    preferred_valence: float
    has_panna: bool


def _build_default_policies() -> list[Policy]:
    """Construct the default 6-policy candidate set.

    Covers the main wholesome categories (somanassa/upekkhā × pañña/nopañña)
    plus the two common unwholesome defaults (dosa, lobha) so the EFE can
    demonstrate that wholesome policies win under reasonable priors.
    """
    def _has_panna(ct: CittaType) -> bool:
        return Cetasika.PANNA in ct.cetasika_profile

    return [
        Policy(get("maha_kusala_somanassa_panna_asankharika"),  +0.8, _has_panna(get("maha_kusala_somanassa_panna_asankharika"))),
        Policy(get("maha_kusala_somanassa_nopanna_asankharika"), +0.8, _has_panna(get("maha_kusala_somanassa_nopanna_asankharika"))),
        Policy(get("maha_kusala_upekkha_panna_asankharika"),     0.0, _has_panna(get("maha_kusala_upekkha_panna_asankharika"))),
        Policy(get("maha_kusala_upekkha_nopanna_asankharika"),   0.0, _has_panna(get("maha_kusala_upekkha_nopanna_asankharika"))),
        Policy(get("dosa_asankharika"),                          -0.8, _has_panna(get("dosa_asankharika"))),
        Policy(get("lobha_somanassa_ditthi_asankharika"),        +0.6, _has_panna(get("lobha_somanassa_ditthi_asankharika"))),
    ]


# ── EFE scoring ───────────────────────────────────────────────────────────────

def efe_score(
    policy: Policy,
    vedana_scalar: float,
    surprise: float,
    epistemic_weight: float = config.EFE_EPISTEMIC_WEIGHT,
    anusaya_bias: float = 0.0,
) -> float:
    """Compute Expected Free Energy for one candidate policy.

    Lower score = preferred policy.

    G(π) = pragmatic_cost − epistemic_gain + anusaya_bias

    Args:
        policy:           Candidate javana policy.
        vedana_scalar:    Current percept valence in [-1, +1].
        surprise:         Current free energy from the GenerativeModel ∈ [0, 2].
        epistemic_weight: Scaling factor for the epistemic bonus.
        anusaya_bias:     Additive EFE bias from STDP weight space (M6 hook);
                          negative = habitual tendency toward this type.

    Returns:
        EFE score (lower = more preferred).
    """
    pragmatic = (vedana_scalar - policy.preferred_valence) ** 2
    epistemic = epistemic_weight * float(policy.has_panna) * surprise
    return pragmatic - epistemic + anusaya_bias


def _type_to_vedana(ct: CittaType) -> VedanāType:
    """Return the single permitted vedanā for a javana type."""
    return next(iter(ct.permitted_vedana))


# ── JavanaDeterminer ──────────────────────────────────────────────────────────

class JavanaDeterminer:
    """Active Inference DeterminerFn for moment 8 (CLAUDE.md §8 M5).

    Replaces the grade-based stub from M3 with EFE-minimising type selection.
    Drop-in replacement: pass an instance as the ``determiner`` argument of
    :class:`~drhm.snn.citta_vithi.CittaVithi`.

    Args:
        model:            A :class:`~drhm.inference.active_inference.GenerativeModel`
                          that tracks the agent's predictive beliefs.
        policies:         Candidate javana types.  Defaults to
                          :func:`_build_default_policies` (6 representative types).
        epistemic_weight: Scales the pañña epistemic bonus in EFE (config default).
        surprise_gate:    Surprise below this → early-exit for MAHANTA objects
                          (config default ``EFE_SURPRISE_GATE``).
        anusaya_bias:     Optional dict mapping CittaType id → additive EFE bias.
                          Populated by the STDP layer at M6; None = uniform prior.
    """

    def __init__(
        self,
        model: GenerativeModel,
        policies: list[Policy] | None = None,
        epistemic_weight: float = config.EFE_EPISTEMIC_WEIGHT,
        surprise_gate: float = config.EFE_SURPRISE_GATE,
        anusaya_bias: dict[str, float] | None = None,
    ) -> None:
        self._model = model
        self._policies: list[Policy] = policies if policies is not None else _build_default_policies()
        self._epistemic_weight = epistemic_weight
        self._surprise_gate = surprise_gate
        self._anusaya: dict[str, float] = anusaya_bias or {}

    def __call__(self, inp: DeterminerInput) -> DeterminerOutput:
        """Select javana CittaType via EFE minimisation.

        Args:
            inp: :class:`~drhm.snn.citta_vithi.DeterminerInput` from moment 8.
                 Uses ``inp.x_t`` if provided, else falls back to a 1-D array
                 wrapping ``inp.vedana_scalar`` as the proxy observation.

        Returns:
            :class:`~drhm.snn.citta_vithi.DeterminerOutput` with the selected
            type, or ``javana_type=None`` for an early-exit.
        """
        # PARITTA grade: already handled by the grade gate — always early-exit
        if inp.grade == ArammanaGrade.PARITTA:
            return DeterminerOutput(javana_type=None)

        # Compute surprise using x_t hypervector when available, else vedanā proxy
        if inp.x_t is not None:
            surprise = self._model.surprise(inp.x_t)
        else:
            surprise = self._model.surprise(np.array([inp.vedana_scalar], dtype=np.float64))

        # EFE gate: familiar MAHANTA stimuli exit without firing javana
        # (ATI_MAHANTA is exempted — it always proceeds to full registration)
        if surprise < self._surprise_gate and inp.grade != ArammanaGrade.ATI_MAHANTA:
            return DeterminerOutput(javana_type=None)

        # Score all candidate policies and select the minimum-EFE type
        best_policy = min(
            self._policies,
            key=lambda p: efe_score(
                p,
                inp.vedana_scalar,
                surprise,
                self._epistemic_weight,
                self._anusaya.get(p.citta_type.id, 0.0),
            ),
        )
        return DeterminerOutput(
            javana_type=best_policy.citta_type,
            vedana=_type_to_vedana(best_policy.citta_type),
        )

    def observe(self, x: np.ndarray) -> float:
        """Update the generative model with a committed observation.

        Call this at moments 16–17 (tadarammana) when the vithi registers the
        percept.  Returns pre-update surprise (same semantics as
        :meth:`~drhm.inference.active_inference.GenerativeModel.observe`).

        Args:
            x: Observed X_t hypervector or 1-D vedanā proxy, shape (D,).

        Returns:
            Surprise before the belief update.
        """
        return self._model.observe(x)

    @property
    def model(self) -> GenerativeModel:
        """The underlying generative model (read access for inspection/M6)."""
        return self._model
