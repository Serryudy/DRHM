"""ManoDvaraVithi — mind-door process (CLAUDE.md §4, M3.5 / M6).

Two uses, one FSM
──────────────────
1. Waking cascade (M3.5): after every full sense-door vithi, a cascade of
   mind-door processes fires automatically. Each chain takes the now-past sense
   object as its mind-door object and processes it at increasing abstraction:
     chain 0 — perceives the object as a whole
     chain 1 — discerns form / shape
     chain 2 — discerns name / concept
     chain N — driven by cetanā momentum until momentum drops below threshold
   This is the mechanism of recognition, naming, and meaning-making — and, when
   momentum runs high, the overthinker's rumination.

2. Dreaming / offline replay (M6): during sleep, the same FSM is driven by noise-
   injected episodic memory rather than a sense object. The determiner is swapped
   for a replay-driven one; the chain structure is identical. See
   drhm/sleep/replay.py.

Structure of one ManoDvaraVithi chain
───────────────────────────────────────
  Bv  — bhavanga vibrating  (1 citta; MOMENT_BHAVANGA_CALANA)
  Ba  — bhavanga arrested   (1 citta; MOMENT_BHAVANGA_UPACCHEDA)
  M   — mano-dvaravajjana   (1 citta; MOMENT_MIND_DOOR_ADVERTING — selects javana)
  J×7 — javana              (7 identical cittas of the type chosen at M)
  D×2 — tadārammaṇa         (2 cittas, only for ATI_MAHANTA grade objects)
  [Bha — bhavanga resumes — implicit; the next chain's Bv/Ba is that bhavanga]

The leading Bv/Ba of each chain *is* the "brief bhavanga between chains": the
life-continuum flows momentarily, is arrested again by the persisting mental
object, and the next mind-door process begins.

Cascade depth
─────────────
  cetanā_momentum = |vedanā_scalar| × grade_weight   (set by the caller from X_t)
  After each chain: momentum *= config.CETANA_MOMENTUM_DECAY
  Stop when momentum < config.CETANA_MOMENTUM_THRESHOLD.
  Always fire at least config.MANO_DVARA_CASCADE_MIN chains (whole → form → name).
  Emotionally neutral objects (upekkhā) get the minimum 3 chains; charged objects
  (strong somanassa / domanassa) ruminate for many more.

Substrate note (M3.5)
─────────────────────
The mind-door chain is modelled symbolically here — a deterministic moment trace
over the citta-type registry, exactly like CittaVithi. It does not perturb the LIF
substrate (the object is internal/mental, not a fresh sensory spike). Driving the
mind-door process through recurrent SNN dynamics is a Phase 4+ refinement; the FSM
structure and the momentum-driven scheduling are what M3.5 establishes.

The ``ManoDeterminerFn`` at moment M is the Active Inference hook (M5), mirroring
the sense-door determiner at votthapana. A grade-based stub is provided for M3.5.

Reference: Bhikkhu Bodhi, *Comprehensive Manual* ch. 4 (mind-door process);
           Nina van Gorkom, *Abhidhamma in Daily Life* ch. 14 (manodvāra-vīthi).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from drhm import config
from drhm.citta.types import CittaType, get
from drhm.citta.vedana import VedanāType
from drhm.sensory.events import ArammanaGrade

# ── Moment trace ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ManoMomentRecord:
    """One citta-moment within a mind-door chain.

    Attributes:
        number:     Position within the chain (1-based; 1=Bv … up to 12 with D×2).
        citta_type: The CittaType active at this moment.
        phase:      Label matching the MOMENT_* vocabulary (CLAUDE.md §3).
    """

    number: int
    citta_type: CittaType
    phase: str


@dataclass
class ManoDvaraVithiResult:
    """Outcome of one ManoDvaraVithi chain.

    Attributes:
        chain_index:  Position in the cascade (0 = first, perceives the whole).
        javana_type:  The CittaType that fired 7 times as javana.
        vedana:       The vedanā of the javana phase.
        moments:      Ordered moment trace for this chain.
        motor_spikes: Motor/vocal spikes emitted during javana (efferent output; M5).
        registered:   True if tadārammaṇa fired (ATI_MAHANTA grade only).
        momentum_out: Cetanā momentum remaining after this chain (input × decay).
    """

    chain_index: int
    javana_type: CittaType
    vedana: VedanāType
    moments: list[ManoMomentRecord] = field(default_factory=list)
    motor_spikes: list = field(default_factory=list)
    registered: bool = False
    momentum_out: float = 0.0

    @property
    def javana_moments(self) -> list[ManoMomentRecord]:
        """The 7 javana moment records of this chain."""
        return [m for m in self.moments if m.phase == "MOMENT_JAVANA"]

    @property
    def starts_with_bhavanga(self) -> bool:
        """True iff the chain opens with the brief bhavanga (Bv then Ba)."""
        return (
            len(self.moments) >= 2
            and self.moments[0].phase == "MOMENT_BHAVANGA_CALANA"
            and self.moments[1].phase == "MOMENT_BHAVANGA_UPACCHEDA"
        )


@dataclass
class CascadeResult:
    """The full mind-door cascade following one sense-door vithi (M3.5).

    Attributes:
        chains: Ordered list of every ManoDvaraVithiResult that fired.
    """

    chains: list[ManoDvaraVithiResult]

    @property
    def chain_count(self) -> int:
        return len(self.chains)

    @property
    def total_javana_cittas(self) -> int:
        return sum(len(c.javana_moments) for c in self.chains)

    @property
    def momenta(self) -> list[float]:
        """Outgoing momentum after each chain — must be monotonically decreasing."""
        return [c.momentum_out for c in self.chains]


# ── Determiner (Active Inference hook at moment M — M5) ───────────────────────

@dataclass
class ManoDeterminerInput:
    """Context passed to the mano-dvaravajjana (moment M) determiner.

    Attributes:
        chain_index:   Cascade position (0=whole, 1=form, 2=name, …).
        momentum:      Current cetanā momentum entering this chain.
        vedana_scalar: Continuous valence of the mind-door object in [-1, +1].
    """

    chain_index: int
    momentum: float
    vedana_scalar: float


@dataclass
class ManoDeterminerOutput:
    """Result from the moment-M determiner.

    Attributes:
        javana_type: CittaType to fire 7 times as javana (never None — a mind-door
                     object that reached adverting always apperceives).
        vedana:      The vedanā of the selected javana type.
    """

    javana_type: CittaType
    vedana: VedanāType = VedanāType.UPEKKHA


ManoDeterminerFn = Callable[[ManoDeterminerInput], ManoDeterminerOutput]


# ── Momentum scheduling helpers ───────────────────────────────────────────────

_GRADE_WEIGHT: dict[ArammanaGrade, float] = {
    ArammanaGrade.ATI_MAHANTA: 1.0,
    ArammanaGrade.MAHANTA:     0.7,
    ArammanaGrade.PARITTA:     0.3,
    ArammanaGrade.ATI_PARITTA: 0.0,  # never triggers (dropped at the attention gate)
}


def initial_momentum(vedana_scalar: float, grade: ArammanaGrade) -> float:
    """Compute the starting cetanā momentum for a cascade.

    Args:
        vedana_scalar: Continuous valence in [-1, +1] from Conceptual Space.
        grade:         ArammanaGrade of the triggering sense object.

    Returns:
        A float in [0, 1] — higher = more mind-door chains will fire.
    """
    return abs(vedana_scalar) * _GRADE_WEIGHT.get(grade, 0.5)


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


def vedana_to_scalar(vedana: VedanāType) -> float:
    """Map a discrete vedanā tone to a representative valence scalar in [-1, +1].

    The inverse of :func:`drhm.citta.vedana.categorise` for momentum purposes:
    pleasant tones are strongly positive, painful strongly negative, neutral zero.
    Used to seed cascade momentum from a sense-door javana's vedanā.
    """
    return {
        VedanāType.SOMANASSA: 0.8,
        VedanāType.SUKHA:     0.8,
        VedanāType.UPEKKHA:   0.0,
        VedanāType.DOMANASSA: -0.8,
        VedanāType.DUKKHA:    -0.8,
    }[vedana]


# ── The ManoDvaraVithi FSM (one chain) ────────────────────────────────────────

class ManoDvaraVithi:
    """Waking mind-door cascade and dreaming-replay FSM (one chain per process()).

    Instantiated by :func:`run_cascade` (waking, M3.5) and by
    drhm/sleep/replay.py (dreaming, M6). The two uses differ only in the
    *determiner* supplied; the chain structure is identical.

    Args:
        determiner: Optional moment-M javana selector. Defaults to the grade-based
                    M3.5 stub; replaced by Active Inference at M5 and by a
                    replay-driven determiner at M6.
    """

    def __init__(self, determiner: ManoDeterminerFn | None = None) -> None:
        self._determiner: ManoDeterminerFn = determiner or _stub_mano_determiner

    def process(
        self,
        chain_index: int,
        momentum: float,
        vedana_scalar: float,
        grade: ArammanaGrade,
    ) -> ManoDvaraVithiResult:
        """Run one mind-door chain and return its moment trace.

        Args:
            chain_index:   Cascade position (0 = whole, 1 = form, 2 = name, …).
            momentum:      Cetanā momentum entering this chain.
            vedana_scalar: Valence of the mind-door object in [-1, +1].
            grade:         ArammanaGrade (governs whether tadārammaṇa fires).

        Returns:
            :class:`ManoDvaraVithiResult` with moments Bv-Ba-M-J×7-[D×2].
        """
        moments: list[ManoMomentRecord] = []
        bhav = get("bhavanga")

        # Bv / Ba — the brief bhavanga that precedes the mind-door process
        moments.append(ManoMomentRecord(1, bhav, "MOMENT_BHAVANGA_CALANA"))
        moments.append(ManoMomentRecord(2, bhav, "MOMENT_BHAVANGA_UPACCHEDA"))

        # M — mano-dvaravajjana: selects the javana type (Active Inference hook)
        moments.append(
            ManoMomentRecord(3, get("manodvaravajjana"), "MOMENT_MIND_DOOR_ADVERTING")
        )
        out = self._determiner(
            ManoDeterminerInput(
                chain_index=chain_index,
                momentum=momentum,
                vedana_scalar=vedana_scalar,
            )
        )

        # J×7 — ONE CittaType repeated exactly JAVANA_COUNT times
        for rep in range(config.JAVANA_COUNT):
            moments.append(ManoMomentRecord(4 + rep, out.javana_type, "MOMENT_JAVANA"))

        # D×2 — tadārammaṇa, only for very-great (ATI_MAHANTA) objects
        registered = grade == ArammanaGrade.ATI_MAHANTA
        if registered:
            tad = _select_tadarammana(out.vedana)
            base = 4 + config.JAVANA_COUNT  # = 11
            moments.append(ManoMomentRecord(base, tad, "MOMENT_REGISTERING"))
            moments.append(ManoMomentRecord(base + 1, tad, "MOMENT_REGISTERING"))

        return ManoDvaraVithiResult(
            chain_index=chain_index,
            javana_type=out.javana_type,
            vedana=out.vedana,
            moments=moments,
            registered=registered,
            momentum_out=momentum * config.CETANA_MOMENTUM_DECAY,
        )


# ── Cascade runner (the vithi scheduler's mind-door half) ─────────────────────

def run_cascade(
    vedana_scalar: float,
    grade: ArammanaGrade,
    determiner: ManoDeterminerFn | None = None,
) -> CascadeResult:
    """Fire the full mind-door cascade following one sense-door vithi (M3.5).

    Computes the starting momentum from *vedana_scalar* and *grade*, derives the
    chain count (≥ MANO_DVARA_CASCADE_MIN), and runs that many ManoDvaraVithi
    chains, decaying momentum after each.

    Args:
        vedana_scalar: Valence of the just-cognized sense object in [-1, +1].
        grade:         ArammanaGrade of that object.
        determiner:    Optional moment-M selector (M5 Active Inference / M6 replay).

    Returns:
        :class:`CascadeResult` holding every chain that fired, in order.
    """
    momentum = initial_momentum(vedana_scalar, grade)
    depth = cascade_depth(momentum)
    vithi = ManoDvaraVithi(determiner=determiner)

    chains: list[ManoDvaraVithiResult] = []
    m = momentum
    for i in range(depth):
        result = vithi.process(
            chain_index=i,
            momentum=m,
            vedana_scalar=vedana_scalar,
            grade=grade,
        )
        chains.append(result)
        m = result.momentum_out

    return CascadeResult(chains=chains)


# ── Stub determiner (replaced by Active Inference at M5) ──────────────────────

def _stub_mano_determiner(inp: ManoDeterminerInput) -> ManoDeterminerOutput:
    """Grade-free M3.5 determiner: selects a javana type by object valence.

    Pleasant object  → wholesome somanassa apperception (mahā-kusala, paññā).
    Painful object   → aversive rumination (dosa-rooted) — the overthinker's loop.
    Neutral object   → equanimous wholesome apperception (mahā-kusala, upekkhā).

    Active Inference (M5) replaces this with EFE-minimising type selection over
    vedanā, anusaya (STDP weight space), and salience grade.
    """
    if inp.vedana_scalar >= config.VEDANA_PLEASANT_THRESHOLD:
        return ManoDeterminerOutput(
            javana_type=get("maha_kusala_somanassa_panna_asankharika"),
            vedana=VedanāType.SOMANASSA,
        )
    if inp.vedana_scalar <= config.VEDANA_PAINFUL_THRESHOLD:
        return ManoDeterminerOutput(
            javana_type=get("dosa_asankharika"),
            vedana=VedanāType.DOMANASSA,
        )
    return ManoDeterminerOutput(
        javana_type=get("maha_kusala_upekkha_nopanna_asankharika"),
        vedana=VedanāType.UPEKKHA,
    )


def _select_tadarammana(vedana: VedanāType) -> CittaType:
    """Tadārammaṇa follows the javana's vedanā (mind-door registration)."""
    if vedana == VedanāType.SOMANASSA:
        return get("tadarammana_somanassa")
    return get("tadarammana_upekkha")
