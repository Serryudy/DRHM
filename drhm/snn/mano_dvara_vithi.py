"""ManoDvaraVithi — mind-door process (CLAUDE.md §4, M3.5 / M6).

Two uses, one FSM
──────────────────
1. Waking cascade (M3.5): after every sense-door vithi completes, a cascade of
   mind-door processes fires automatically. Each chain takes the now-past sense
   object as its mind-door object and processes it at increasing abstraction:
     chain 1 — perceives the object as a whole
     chain 2 — discerns form / shape
     chain 3 — discerns name / concept
     chain N — driven by cetanā momentum until momentum drops below threshold

2. Dreaming / offline replay (M6): during sleep, the same FSM is driven by noise-
   injected episodic memory rather than sensory input. See drhm/sleep/replay.py.

Structure of one ManoDvaraVithi chain
───────────────────────────────────────
  Bv  — bhavanga vibrating (1 citta-moment)
  Ba  — bhavanga arrested  (1 citta-moment)
  M   — mano-dvaravajjana  (1 citta-moment, CittaFunction.MIND_DOOR_ADVERTING)
  J×7 — javana             (7 identical cittas of the type chosen at M)
  D×2 — tadarammaṇa        (2 cittas, only for ATI_MAHANTA grade objects)
  Bha — bhavanga resumes

Cascade depth
─────────────
  cetanā_momentum = vedanā_scalar × salience_grade (set by caller from X_t)
  After each chain: momentum *= config.CETANA_MOMENTUM_DECAY
  Stop when momentum < config.CETANA_MOMENTUM_THRESHOLD
  Always fire at least config.MANO_DVARA_CASCADE_MIN chains.

Implementation note (M3.5)
───────────────────────────
The full FSM body is built at M3.5 once CittaVithi (M3) exists to be reused.
This file defines the dataclass and cascade scheduler so downstream code can
import the type and the constants without waiting for full implementation.

Reference: Bhikkhu Bodhi, *Comprehensive Manual* ch. 4 (mind-door process);
           Nina van Gorkom, *Abhidhamma in Daily Life* ch. 14 (manodvāra-vīthi).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drhm import config
from drhm.citta.types import CittaType
from drhm.citta.vedana import VedanāType
from drhm.sensory.events import ArammanaGrade


@dataclass
class ManoDvaraVithiResult:
    """Outcome of one ManoDvaraVithi chain.

    Attributes:
        chain_index:    Position in the cascade (0 = first, perceives whole).
        javana_type:    The CittaType that fired 7 times as javana.
        vedana:         The vedanā of the javana phase.
        motor_spikes:   Motor/vocal spikes emitted during javana (efferent output).
        registered:     True if tadarammaṇa fired (ATI_MAHANTA grade only).
        momentum_out:   Cetanā momentum remaining after this chain.
    """

    chain_index: int
    javana_type: CittaType
    vedana: VedanāType
    motor_spikes: list = field(default_factory=list)
    registered: bool = False
    momentum_out: float = 0.0


def initial_momentum(vedana_scalar: float, grade: ArammanaGrade) -> float:
    """Compute the starting cetanā momentum for a cascade.

    Args:
        vedana_scalar: Continuous valence in [-1, +1] from Conceptual Space.
        grade:         ArammanaGrade of the triggering sense object.

    Returns:
        A float in [0, 1] — higher = more mind-door chains will fire.
    """
    grade_weight = {
        ArammanaGrade.ATI_MAHANTA: 1.0,
        ArammanaGrade.MAHANTA:     0.7,
        ArammanaGrade.PARITTA:     0.3,
        ArammanaGrade.ATI_PARITTA: 0.0,  # never triggers (dropped at attention gate)
    }
    return abs(vedana_scalar) * grade_weight.get(grade, 0.5)


def cascade_depth(momentum: float) -> int:
    """Return how many mind-door chains will fire given starting *momentum*.

    Always at least MANO_DVARA_CASCADE_MIN. Chains continue while
    momentum × CETANA_MOMENTUM_DECAY^n > CETANA_MOMENTUM_THRESHOLD.
    """
    count = 0
    m = momentum
    while m > config.CETANA_MOMENTUM_THRESHOLD:
        count += 1
        m *= config.CETANA_MOMENTUM_DECAY
    return max(count, config.MANO_DVARA_CASCADE_MIN)


# Full cascade runner is implemented at M3.5 (depends on CittaVithi from M3).
# Placeholder so imports resolve now.
class ManoDvaraVithi:  # pragma: no cover — stub until M3.5
    """Waking mind-door cascade and dreaming replay FSM.

    Instantiated by the vithi scheduler in citta_vithi.py (M3.5) and by
    drhm/sleep/replay.py (M6). Full implementation at milestone M3.5.
    """

    def __init__(self) -> None:
        raise NotImplementedError("ManoDvaraVithi full implementation lands at M3.5.")
