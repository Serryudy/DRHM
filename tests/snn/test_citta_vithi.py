"""Sequence Integrity Test and supporting tests for CittaVithi FSM (M3 acceptance).

Canonical acceptance test (CLAUDE.md §8 M3):
  ★ Inject an artificial spike train; trace the cascade; assert it passes through
    all 17 distinct gating phases in order and returns to bhavanga.
    Assert javana fires the same CittaType all 7 times.
    Assert moment-8 gate can early-exit on zero-surprise input — and ONLY at moment 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm import config
from drhm.citta.types import CittaFunction, CittaType, get
from drhm.citta.vedana import VedanāType
from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent
from drhm.snn.bhavanga import Bhavanga, BhavangaState, PerturbationEvent
from drhm.snn.citta_vithi import (
    CittaVithi,
    CittaVithiResult,
    DeterminerInput,
    DeterminerOutput,
    DeterminerFn,
    MomentRecord,
    _stub_determiner,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_bhavanga() -> Bhavanga:
    return Bhavanga(rng=np.random.default_rng(0))


def make_vithi(bhavanga: Bhavanga | None = None, determiner: DeterminerFn | None = None) -> CittaVithi:
    b = bhavanga or make_bhavanga()
    return CittaVithi(b, determiner=determiner)


def fake_trigger() -> PerturbationEvent:
    return PerturbationEvent(timestamp=1.0, input_spikes=50, population_rate=0.8)


def vision_events(n: int = 5, payload: float = 1.0) -> list[SpikeEvent]:
    return [SpikeEvent.now(Modality.VISION, i, payload=payload) for i in range(n)]


def arrest_bhavanga(b: Bhavanga) -> PerturbationEvent:
    """Inject strong events until bhavanga arrests; return the trigger."""
    fired: list[PerturbationEvent] = []
    b.on_arrest = fired.append
    events = [SpikeEvent.now(Modality.VISION, i, payload=1.0) for i in range(50)]
    for _ in range(20):
        if b.inject(events) == BhavangaState.ARRESTED:
            break
    assert fired, "bhavanga did not arrest"
    return fired[0]


# ---------------------------------------------------------------------------
# ★ Sequence Integrity Test (canonical M3 acceptance)
# ---------------------------------------------------------------------------

class TestSequenceIntegrity:
    """CLAUDE.md §8 M3 acceptance gate."""

    def test_ati_mahanta_produces_17_moments_in_order(self) -> None:
        """Full ATI_MAHANTA cycle must have exactly 17 moments numbered 1–17."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        trigger = fake_trigger()
        result = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, trigger)

        assert result.moment_numbers == list(range(1, 18)), (
            f"Expected [1..17], got {result.moment_numbers}"
        )

    def test_mahanta_produces_15_moments_in_order(self) -> None:
        """MAHANTA cycle: moments 1–15 (no tadarammana)."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.moment_numbers == list(range(1, 16))

    def test_paritta_early_exits_at_moment_8(self) -> None:
        """PARITTA cycle: moments 1–8 only; early_exit=True."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
        assert result.moment_numbers == list(range(1, 9))
        assert result.early_exit is True

    def test_javana_fires_same_type_all_7_times(self) -> None:
        """Moments 9–15 must all carry the identical CittaType (CLAUDE.md §6 rule 4)."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())

        javana_moments = result.javana_moments
        assert len(javana_moments) == config.JAVANA_COUNT  # 7
        javana_types = {m.citta_type.id for m in javana_moments}
        assert len(javana_types) == 1, (
            f"Javana should be ONE type repeated 7 times; got {javana_types}"
        )

    def test_only_moment_8_can_early_exit(self) -> None:
        """Early-exit may only occur after moment 8, never earlier or later."""
        b = make_bhavanga()
        vithi = make_vithi(b)

        # PARITTA exits after 8
        result = vithi.process(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
        assert result.early_exit and result.moments[-1].number == 8

        # MAHANTA and ATI_MAHANTA never early-exit
        for grade in (ArammanaGrade.MAHANTA, ArammanaGrade.ATI_MAHANTA):
            b.reset()  # bhavanga already reset by previous cycle; belt-and-braces
            r = vithi.process(vision_events(), grade, fake_trigger())
            assert not r.early_exit

    def test_moments_are_strictly_sequential(self) -> None:
        """Each moment number must be exactly one more than the previous."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        nums = result.moment_numbers
        for i in range(1, len(nums)):
            assert nums[i] == nums[i - 1] + 1, (
                f"Gap between moments {nums[i-1]} and {nums[i]}"
            )

    def test_bhavanga_returns_to_flowing_after_cycle(self) -> None:
        """After process() completes, bhavanga must be FLOWING again."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        assert b.state == BhavangaState.FLOWING

    def test_bhavanga_returns_to_flowing_after_early_exit(self) -> None:
        """Even on early-exit at moment 8, bhavanga must be reset to FLOWING."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        vithi.process(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
        assert b.state == BhavangaState.FLOWING

    def test_end_to_end_with_real_bhavanga_arrest(self) -> None:
        """Full path: inject spikes → bhavanga arrests → vithi fires → 17 moments."""
        b = make_bhavanga()
        vithi = make_vithi(b)
        trigger = arrest_bhavanga(b)

        # bhavanga just arrested; now run the vithi
        events = vision_events(50, payload=1.0)
        result = vithi.process(events, ArammanaGrade.ATI_MAHANTA, trigger)

        assert result.moment_numbers == list(range(1, 18))
        assert not result.early_exit
        assert result.javana_type is not None
        assert b.state == BhavangaState.FLOWING


# ---------------------------------------------------------------------------
# Javana correctness
# ---------------------------------------------------------------------------

class TestJavana:

    def test_javana_type_has_javana_function(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.javana_type is not None
        assert result.javana_type.function == CittaFunction.JAVANA

    def test_javana_count_matches_config(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert len(result.javana_moments) == config.JAVANA_COUNT

    def test_javana_none_on_early_exit(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
        assert result.javana_type is None
        assert result.javana_moments == []

    def test_custom_determiner_is_called(self) -> None:
        """The determiner callable is invoked at moment 8."""
        called: list[DeterminerInput] = []

        def my_determiner(inp: DeterminerInput) -> DeterminerOutput:
            called.append(inp)
            return DeterminerOutput(
                javana_type=get("maha_kusala_upekkha_panna_asankharika"),
                vedana=VedanāType.UPEKKHA,
            )

        b = make_bhavanga()
        vithi = make_vithi(b, determiner=my_determiner)
        vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert len(called) == 1
        assert called[0].grade == ArammanaGrade.MAHANTA

    def test_custom_determiner_type_propagates_to_all_javana_moments(self) -> None:
        """Whatever type the determiner returns must appear in all 7 javana moments."""
        target = get("lobha_somanassa_ditthi_asankharika")

        def my_determiner(inp: DeterminerInput) -> DeterminerOutput:
            return DeterminerOutput(javana_type=target, vedana=VedanāType.SOMANASSA)

        b = make_bhavanga()
        vithi = make_vithi(b, determiner=my_determiner)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())

        for m in result.javana_moments:
            assert m.citta_type is target


# ---------------------------------------------------------------------------
# Registration (tadarammana)
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_tadarammana_fires_for_ati_mahanta(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        assert result.registered is True
        registering = [m for m in result.moments if m.phase == "MOMENT_REGISTERING"]
        assert len(registering) == 2

    def test_tadarammana_does_not_fire_for_mahanta(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.registered is False
        registering = [m for m in result.moments if m.phase == "MOMENT_REGISTERING"]
        assert registering == []

    def test_tadarammana_moments_have_registering_function(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        for m in result.moments:
            if m.phase == "MOMENT_REGISTERING":
                assert m.citta_type.function == CittaFunction.REGISTERING

    def test_tadarammana_same_type_both_repetitions(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        reg = [m for m in result.moments if m.phase == "MOMENT_REGISTERING"]
        assert reg[0].citta_type is reg[1].citta_type


# ---------------------------------------------------------------------------
# Sense-door selection (moment 5)
# ---------------------------------------------------------------------------

class TestSenseDoorSelection:

    def _m5_for(self, modality: Modality, payload: float = 1.0) -> MomentRecord:
        b = make_bhavanga()
        vithi = make_vithi(b)
        events = [SpikeEvent.now(modality, 0, payload=payload)]
        result = vithi.process(events, ArammanaGrade.MAHANTA, fake_trigger())
        return result.moments[4]  # index 4 = moment number 5

    def test_vision_maps_to_eye_vinnana(self) -> None:
        m5 = self._m5_for(Modality.VISION)
        assert "eye" in m5.citta_type.id

    def test_audition_maps_to_ear_vinnana(self) -> None:
        m5 = self._m5_for(Modality.AUDITION)
        assert "ear" in m5.citta_type.id

    def test_touch_maps_to_body_vinnana(self) -> None:
        m5 = self._m5_for(Modality.TOUCH)
        assert "body" in m5.citta_type.id

    def test_positive_payload_is_kusala_vipaka(self) -> None:
        m5 = self._m5_for(Modality.VISION, payload=1.0)
        assert "kusala_vipaka" in m5.citta_type.id

    def test_negative_payload_is_akusala_vipaka(self) -> None:
        m5 = self._m5_for(Modality.VISION, payload=-1.0)
        assert "akusala_vipaka" in m5.citta_type.id

    def test_moment_5_has_sense_function(self) -> None:
        m5 = self._m5_for(Modality.VISION)
        assert m5.citta_type.function == CittaFunction.SENSE


# ---------------------------------------------------------------------------
# Phase-label correctness
# ---------------------------------------------------------------------------

class TestPhaseLabels:

    def test_moment_1_is_bhavanga_sota(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.moments[0].phase == "MOMENT_BHAVANGA_SOTA"

    def test_moment_4_is_adverting(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.moments[3].phase == "MOMENT_ADVERTING"

    def test_moment_8_is_determining(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.moments[7].phase == "MOMENT_DETERMINING"

    def test_moments_9_to_15_are_javana(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        for m in result.moments[8:15]:
            assert m.phase == "MOMENT_JAVANA"

    def test_moment_4_citta_has_five_door_adverting_function(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.moments[3].citta_type.function == CittaFunction.FIVE_DOOR_ADVERTING

    def test_moment_8_citta_has_determining_function(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)
        result = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert result.moments[7].citta_type.function == CittaFunction.DETERMINING


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_produce_same_moment_sequence(self) -> None:
        """Identical inputs must produce identical traces (FSM is deterministic)."""
        events = vision_events(10, payload=0.5)
        trigger = fake_trigger()
        grade = ArammanaGrade.ATI_MAHANTA

        results = []
        for _ in range(3):
            b = Bhavanga(rng=np.random.default_rng(0))
            vithi = CittaVithi(b)
            results.append(vithi.process(events, grade, trigger))

        for r in results[1:]:
            assert r.moment_numbers == results[0].moment_numbers
            assert r.javana_type == results[0].javana_type

    def test_grade_ati_mahanta_always_produces_17(self) -> None:
        for _ in range(5):
            b = make_bhavanga()
            vithi = make_vithi(b)
            result = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
            assert len(result.moments) == 17

    def test_grade_paritta_always_produces_8(self) -> None:
        for _ in range(5):
            b = make_bhavanga()
            vithi = make_vithi(b)
            result = vithi.process(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
            assert len(result.moments) == 8


# ---------------------------------------------------------------------------
# Stub determiner
# ---------------------------------------------------------------------------

class TestStubDeterminer:

    def test_paritta_returns_early_exit(self) -> None:
        inp = DeterminerInput(grade=ArammanaGrade.PARITTA, events=[])
        out = _stub_determiner(inp)
        assert out.javana_type is None

    def test_mahanta_returns_wholesome_type(self) -> None:
        inp = DeterminerInput(grade=ArammanaGrade.MAHANTA, events=[])
        out = _stub_determiner(inp)
        assert out.javana_type is not None
        assert out.javana_type.function == CittaFunction.JAVANA

    def test_ati_mahanta_returns_somanassa_type(self) -> None:
        inp = DeterminerInput(grade=ArammanaGrade.ATI_MAHANTA, events=[])
        out = _stub_determiner(inp)
        assert out.javana_type is not None
        assert VedanāType.SOMANASSA in out.javana_type.permitted_vedana

    def test_determiner_output_has_matching_vedana(self) -> None:
        """vedana in DeterminerOutput must match a permitted_vedana of javana_type."""
        for grade in (ArammanaGrade.MAHANTA, ArammanaGrade.ATI_MAHANTA):
            inp = DeterminerInput(grade=grade, events=[])
            out = _stub_determiner(inp)
            if out.javana_type is not None:
                assert out.vedana in out.javana_type.permitted_vedana


# ---------------------------------------------------------------------------
# Repeated cycle (vithi can run multiple times on same CittaVithi instance)
# ---------------------------------------------------------------------------

class TestMultipleCycles:

    def test_two_consecutive_cycles_both_complete(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)

        r1 = vithi.process(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        # bhavanga was reset; run again
        r2 = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())

        assert r1.moment_numbers == list(range(1, 16))
        assert r2.moment_numbers == list(range(1, 18))
        assert b.state == BhavangaState.FLOWING

    def test_early_exit_then_full_cycle(self) -> None:
        b = make_bhavanga()
        vithi = make_vithi(b)

        r1 = vithi.process(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
        r2 = vithi.process(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())

        assert r1.early_exit
        assert not r2.early_exit
        assert len(r2.moments) == 17
