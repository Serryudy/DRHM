"""17-moment sense-door cognitive cycle — CittaVithi FSM (CLAUDE.md §8 M3).

The sense-door vithi (pañcadvāra-vīthi) is the complete cognitive process that
unfolds whenever a sensory object reaches the threshold of consciousness. It is a
deterministic state machine with exactly 17 discrete moments (cittas), running in
strict sequence — no moment may be skipped or reordered outside of the single
sanctioned early-exit at moment 8.

Moment map (CLAUDE.md §3, §10)
────────────────────────────────────────────────────────────────────────────────
  1. bhavanga-sota       (FLOWING)     life-continuum flows — handled by Bhavanga
  2. bhavanga-calana     (VIBRATING)   first sensory impact — handled by Bhavanga
  3. bhavanga-upaccheda  (ARRESTED)    life-continuum cut   — handled by Bhavanga
  4. pañcadvārāvajjana   MOMENT_ADVERTING      five-door adverting / attention routing
  5. pañcaviññāṇa        MOMENT_SENSE          raw sense consciousness (one of 10)
  6. sampaṭicchana       MOMENT_RECEIVING      receiving / spatiotemporal aggregation
  7. santīraṇa           MOMENT_INVESTIGATING  investigating / pattern matching
  8. voṭṭhapana          MOMENT_DETERMINING    *** SALIENCE GATE ***
                           → PARITTA grade:     early-exit (no javana; energy-proportional)
                           → MAHANTA/ATI:       Active Inference selects javana CittaType
  9–15. javana ×7        MOMENT_JAVANA         ONE CittaType fires 7 identical times
 16–17. tadārammaṇa ×2  MOMENT_REGISTERING    (ATI_MAHANTA only) plasticity commit
         [→ bhavanga.reset()]

Determiner (moment 8)
─────────────────────
The callable ``DeterminerFn`` is the clean integration point for Active Inference
(M5). For M3 a stub is provided that uses grade alone. Replace with
``drhm.inference.active_inference`` at M5 without changing the FSM.

Invariants enforced here (CLAUDE.md §6 rules 3–4)
──────────────────────────────────────────────────
• Moments execute in strict numerical order.
• The ONLY early-exit is at moment 8; all other moments are unconditional.
• Javana is ONE CittaType repeated exactly ``config.JAVANA_COUNT`` (= 7) times.
• ``bhavanga.reset()`` is always called on exit — normal or early.

Reference: Bhikkhu Bodhi, *Comprehensive Manual* ch. 4; CLAUDE.md §8 M3, §10.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from drhm import config
from drhm.citta.types import CittaType, get
from drhm.citta.vedana import VedanāType
from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent
from drhm.snn.bhavanga import Bhavanga, PerturbationEvent
from drhm.snn.mano_dvara_vithi import (
    CascadeResult,
    ManoDeterminerFn,
    run_cascade,
    vedana_to_scalar,
)

# ── Public data types ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MomentRecord:
    """One executed citta-moment in the vithi trace.

    Attributes:
        number:     Moment index 1–17 (canonical Abhidhamma numbering).
        citta_type: The CittaType that was active at this moment.
        phase:      Short label matching the MOMENT_* vocabulary in CLAUDE.md §3.
    """

    number: int
    citta_type: CittaType
    phase: str


@dataclass
class CittaVithiResult:
    """Complete record of one citta-vithi cycle.

    Attributes:
        trigger:     The PerturbationEvent that woke the cycle (from Bhavanga).
        grade:       ArammanaGrade of the triggering sensory object.
        moments:     Ordered list of every MomentRecord executed (1–17 for full
                     ATI_MAHANTA; 1–15 for MAHANTA; 1–8 for PARITTA early-exit).
        javana_type: The CittaType that fired as javana, or None if early-exit.
        early_exit:  True iff the moment-8 gate dropped to bhavanga without javana.
        registered:  True iff tadārammaṇa fired (ATI_MAHANTA grade only).
        javana_vedana: Vedanā of the javana phase (seeds the mind-door cascade
                     momentum); None on early-exit.
    """

    trigger: PerturbationEvent
    grade: ArammanaGrade
    moments: list[MomentRecord]
    javana_type: CittaType | None
    early_exit: bool
    registered: bool
    javana_vedana: VedanāType | None = None

    @property
    def moment_numbers(self) -> list[int]:
        """Ordered list of executed moment numbers — used by the Sequence Integrity Test."""
        return [m.number for m in self.moments]

    @property
    def javana_moments(self) -> list[MomentRecord]:
        """The 7 (or 0) javana moment records."""
        return [m for m in self.moments if m.phase == "MOMENT_JAVANA"]


@dataclass
class CognitionResult:
    """One full waking cognition: a sense-door vithi + its mind-door cascade (M3.5).

    Attributes:
        sense:   The completed sense-door CittaVithiResult.
        cascade: The mind-door cascade that followed, or None if the sense-door
                 vithi early-exited at moment 8 (a dropped object is not ruminated).
    """

    sense: CittaVithiResult
    cascade: CascadeResult | None


@dataclass
class DeterminerInput:
    """Context passed to the moment-8 determiner.

    Attributes:
        grade:         ArammanaGrade of the sense object.
        events:        The triggering sensory SpikeEvents.
        vedana_scalar: Continuous valence from Conceptual Space (Phase 3).
                       Placeholder at M3 — always 0.0 until M4.
    """

    grade: ArammanaGrade
    events: list[SpikeEvent]
    vedana_scalar: float = 0.0


@dataclass
class DeterminerOutput:
    """Result from the moment-8 determiner.

    Attributes:
        javana_type: CittaType to fire as javana, or None to early-exit.
        vedana:      The vedanā of the selected javana type (for tadarammana choice).
    """

    javana_type: CittaType | None
    vedana: VedanāType = VedanāType.UPEKKHA


DeterminerFn = Callable[[DeterminerInput], DeterminerOutput]


# ── CittaVithi ────────────────────────────────────────────────────────────────

class CittaVithi:
    """The 17-moment sense-door cognitive cycle as a deterministic FSM.

    Usage (synchronous, for testing and the M7 agent loop)::

        bhavanga = Bhavanga(...)
        vithi = CittaVithi(bhavanga)
        # ... bhavanga is arrested by sensory input ...
        result = vithi.process(events, grade, trigger)
        # bhavanga.reset() has been called; result.moments holds the full trace

    The ``determiner`` parameter is the Active Inference hook (M5). Leave it
    as None during M3; the stub selects a wholesome javana type by grade alone.

    Args:
        bhavanga:   The resting attractor; receives ``reset()`` at cycle end.
        determiner: Optional moment-8 selector. Defaults to ``_stub_determiner``.
    """

    def __init__(
        self,
        bhavanga: Bhavanga,
        determiner: DeterminerFn | None = None,
    ) -> None:
        self._bhavanga = bhavanga
        self._determiner: DeterminerFn = determiner or _stub_determiner

    # ------------------------------------------------------------------
    # Core FSM
    # ------------------------------------------------------------------

    def process(
        self,
        events: list[SpikeEvent],
        grade: ArammanaGrade,
        trigger: PerturbationEvent,
    ) -> CittaVithiResult:
        """Execute one complete citta-vithi synchronously.

        Moments 1–3 are the bhavanga moments already completed by the
        :class:`~drhm.snn.bhavanga.Bhavanga` class; they are recorded in the
        trace (for the Sequence Integrity Test) but not re-simulated here.

        The ONLY early-exit is at moment 8 (``MOMENT_DETERMINING``) for
        PARITTA-grade objects. All other moments execute unconditionally.

        ``bhavanga.reset()`` is called before returning in all paths.

        Args:
            events:  Triggering SpikeEvents (used to pick the sense-door type).
            grade:   ArammanaGrade from the attention front-end.
            trigger: PerturbationEvent from ``Bhavanga.on_arrest``.

        Returns:
            :class:`CittaVithiResult` with the complete moment trace.
        """
        moments: list[MomentRecord] = []

        # ── Phase I: Baseline — moments 1–3 (completed by Bhavanga) ──────────
        _bhavanga_ct = get("bhavanga")
        moments.append(MomentRecord(1, _bhavanga_ct, "MOMENT_BHAVANGA_SOTA"))
        moments.append(MomentRecord(2, _bhavanga_ct, "MOMENT_BHAVANGA_CALANA"))
        moments.append(MomentRecord(3, _bhavanga_ct, "MOMENT_BHAVANGA_UPACCHEDA"))

        # ── Phase II: Apprehension — moments 4–5 ─────────────────────────────
        # Moment 4: pañcadvārāvajjana (five-door adverting)
        moments.append(MomentRecord(4, get("pancadvaravajjana"), "MOMENT_ADVERTING"))

        # Moment 5: pañcaviññāṇa — sense consciousness chosen by modality + polarity
        sense_ct = _select_sense_citta(events)
        moments.append(MomentRecord(5, sense_ct, "MOMENT_SENSE"))

        # ── Phase III: Assimilation — moments 6–7 ────────────────────────────
        # Moment 6: sampaṭicchana (receiving)
        receiving_ct = _select_receiving_citta(sense_ct)
        moments.append(MomentRecord(6, receiving_ct, "MOMENT_RECEIVING"))

        # Moment 7: santīraṇa (investigating)
        investigating_ct = _select_investigating_citta(sense_ct)
        moments.append(MomentRecord(7, investigating_ct, "MOMENT_INVESTIGATING"))

        # ── Phase IV: Determination — moment 8 (THE GATE) ────────────────────
        moments.append(MomentRecord(8, get("votthapana"), "MOMENT_DETERMINING"))

        det = self._determiner(DeterminerInput(grade=grade, events=events))

        if det.javana_type is None:
            # Energy-proportionality path: insufficient salience → drop to bhavanga
            self._bhavanga.reset()
            return CittaVithiResult(
                trigger=trigger,
                grade=grade,
                moments=moments,
                javana_type=None,
                early_exit=True,
                registered=False,
                javana_vedana=None,
            )

        # ── Phase V: Impulsion — moments 9–15 (javana ×7) ────────────────────
        # ONE CittaType fires exactly JAVANA_COUNT times — identical repetitions.
        javana_ct = det.javana_type
        for rep in range(config.JAVANA_COUNT):
            moments.append(MomentRecord(9 + rep, javana_ct, "MOMENT_JAVANA"))

        # ── Phase VI: Consolidation — moments 16–17 (ATI_MAHANTA only) ───────
        registered = grade == ArammanaGrade.ATI_MAHANTA
        if registered:
            tadarammana_ct = _select_tadarammana_citta(det.vedana)
            moments.append(MomentRecord(16, tadarammana_ct, "MOMENT_REGISTERING"))
            moments.append(MomentRecord(17, tadarammana_ct, "MOMENT_REGISTERING"))

        # Return to bhavanga — always
        self._bhavanga.reset()

        return CittaVithiResult(
            trigger=trigger,
            grade=grade,
            moments=moments,
            javana_type=javana_ct,
            early_exit=False,
            registered=registered,
            javana_vedana=det.vedana,
        )

    # ------------------------------------------------------------------
    # Vithi scheduler: sense-door vithi + mind-door cascade (M3.5)
    # ------------------------------------------------------------------

    def process_and_cascade(
        self,
        events: list[SpikeEvent],
        grade: ArammanaGrade,
        trigger: PerturbationEvent,
        mano_determiner: ManoDeterminerFn | None = None,
    ) -> CognitionResult:
        """Run the sense-door vithi, then fire the mind-door cascade (CLAUDE.md §8 M3.5).

        After a *full* sense-door vithi (one that reached javana), a cascade of
        ≥ ``config.MANO_DVARA_CASCADE_MIN`` mind-door processes fires automatically
        — the mechanism of recognition, naming, and meaning-making. The cascade's
        starting momentum is seeded from the sense-door javana's vedanā: neutral
        objects get the minimum three chains, emotionally charged objects ruminate
        for many more.

        A sense-door vithi that early-exited at moment 8 (PARITTA, dropped object)
        produces no cascade — there is nothing apperceived to think further about.

        Args:
            events:          Triggering SpikeEvents.
            grade:           ArammanaGrade from the attention front-end.
            trigger:         PerturbationEvent from ``Bhavanga.on_arrest``.
            mano_determiner: Optional moment-M selector for the cascade (M5/M6 hook).

        Returns:
            :class:`CognitionResult` pairing the sense-door vithi with its cascade.
        """
        sense = self.process(events, grade, trigger)
        if sense.early_exit or sense.javana_vedana is None:
            return CognitionResult(sense=sense, cascade=None)

        cascade = run_cascade(
            vedana_scalar=vedana_to_scalar(sense.javana_vedana),
            grade=grade,
            determiner=mano_determiner,
        )
        return CognitionResult(sense=sense, cascade=cascade)

    # ------------------------------------------------------------------
    # Convenience wiring
    # ------------------------------------------------------------------

    def make_arrest_handler(
        self,
        events: list[SpikeEvent],
        grade: ArammanaGrade,
    ) -> Callable[[PerturbationEvent], CittaVithiResult]:
        """Return a closure suitable for ``Bhavanga.on_arrest``.

        The returned callable captures *events* and *grade* and calls
        :meth:`process` when the bhavanga is arrested. Used by agent.py (M7)
        to wire the live spike stream into the vithi without polling.
        """
        def _handler(trigger: PerturbationEvent) -> CittaVithiResult:
            return self.process(events, grade, trigger)
        return _handler


# ── Moment-8 stub determiner (replaced by Active Inference at M5) ─────────────

def _stub_determiner(inp: DeterminerInput) -> DeterminerOutput:
    """Placeholder moment-8 determiner for M3.

    Selection logic:
      PARITTA      → early-exit (no javana; energy-proportionality path)
      MAHANTA      → mahā-kusala with upekkhā, no paññā, unprompted
      ATI_MAHANTA  → mahā-kusala with somanassa + paññā, unprompted

    Active Inference (M5) will replace this with EFE-minimising type selection
    based on vedanā scalar, anusaya (STDP weight space), and salience grade.
    """
    if inp.grade == ArammanaGrade.PARITTA:
        return DeterminerOutput(javana_type=None)

    if inp.grade == ArammanaGrade.ATI_MAHANTA:
        return DeterminerOutput(
            javana_type=get("maha_kusala_somanassa_panna_asankharika"),
            vedana=VedanāType.SOMANASSA,
        )

    # MAHANTA (default)
    return DeterminerOutput(
        javana_type=get("maha_kusala_upekkha_nopanna_asankharika"),
        vedana=VedanāType.UPEKKHA,
    )


# ── Moment-selection helpers ──────────────────────────────────────────────────

def _select_sense_citta(events: list[SpikeEvent]) -> CittaType:
    """Moment 5: choose the pañcaviññāṇa appropriate to the dominant event.

    Modality → sense door:
      VISION       → eye-consciousness
      AUDITION     → ear-consciousness
      TOUCH        → body-consciousness (SUKHA/DUKKHA)
      PROPRIOCEPTION / MOTOR → eye-consciousness (closest available proxy)

    Positive payload → kusala-vipāka (pleasant resultant).
    Negative payload → akusala-vipāka (unpleasant resultant).
    """
    if not events:
        return get("eye_vinnana_kusala_vipaka")

    primary = max(events, key=lambda e: abs(e.payload))
    pleasant = primary.payload >= 0

    _door_map = {
        Modality.VISION:        "eye",
        Modality.AUDITION:      "ear",
        Modality.TOUCH:         "body",
        Modality.PROPRIOCEPTION: "eye",  # no proprioception door in Abhidhamma; proxy
        Modality.MOTOR:         "eye",
    }
    door = _door_map.get(Modality(primary.modality), "eye")
    variant = "kusala_vipaka" if pleasant else "akusala_vipaka"
    return get(f"{door}_vinnana_{variant}")


def _select_receiving_citta(sense_ct: CittaType) -> CittaType:
    """Moment 6: sampaṭicchana follows the sense citta's vedanā."""
    if VedanāType.SOMANASSA in sense_ct.permitted_vedana or VedanāType.SUKHA in sense_ct.permitted_vedana:
        return get("sampaticchana_kusala_vipaka")
    return get("sampaticchana_akusala_vipaka")


def _select_investigating_citta(sense_ct: CittaType) -> CittaType:
    """Moment 7: santīraṇa follows the sense citta's vedanā."""
    if VedanāType.SOMANASSA in sense_ct.permitted_vedana:
        return get("santirana_somanassa")
    if VedanāType.DUKKHA in sense_ct.permitted_vedana:
        return get("santirana_upekkha_akusala")
    return get("santirana_upekkha_kusala")


def _select_tadarammana_citta(vedana: VedanāType) -> CittaType:
    """Moments 16–17: tadārammaṇa follows the javana's vedanā."""
    if vedana == VedanāType.SOMANASSA:
        return get("tadarammana_somanassa")
    return get("tadarammana_upekkha")
