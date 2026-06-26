"""Tests for the ManoDvaraVithi waking cascade (M3.5 acceptance).

Acceptance criteria (CLAUDE.md §8 M3.5):
  - After a sense-door vithi, ≥ MANO_DVARA_CASCADE_MIN (3) mind-door chains fire
    with brief bhavanga between each.
  - The cascade stops when cetanā momentum drops below threshold.
  - The same FSM runs from a custom (replay-style) determiner — the mechanism the
    sleep/dreaming path (M6) will reuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm import config
from drhm.citta.types import CittaFunction, get
from drhm.citta.vedana import VedanāType
from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent
from drhm.snn.bhavanga import Bhavanga, PerturbationEvent
from drhm.snn.citta_vithi import CittaVithi, CognitionResult
from drhm.snn.mano_dvara_vithi import (
    CascadeResult,
    ManoDeterminerInput,
    ManoDeterminerOutput,
    ManoDvaraVithi,
    ManoDvaraVithiResult,
    cascade_depth,
    initial_momentum,
    run_cascade,
    vedana_to_scalar,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_trigger() -> PerturbationEvent:
    return PerturbationEvent(timestamp=1.0, input_spikes=50, population_rate=0.8)


def vision_events(n: int = 5, payload: float = 1.0) -> list[SpikeEvent]:
    return [SpikeEvent.now(Modality.VISION, i, payload=payload) for i in range(n)]


def make_vithi() -> CittaVithi:
    return CittaVithi(Bhavanga(rng=np.random.default_rng(0)))


# ---------------------------------------------------------------------------
# Momentum helpers
# ---------------------------------------------------------------------------

class TestMomentumHelpers:

    def test_initial_momentum_scales_with_vedana(self) -> None:
        m_strong = initial_momentum(0.9, ArammanaGrade.ATI_MAHANTA)
        m_weak = initial_momentum(0.1, ArammanaGrade.ATI_MAHANTA)
        assert m_strong > m_weak

    def test_initial_momentum_scales_with_grade(self) -> None:
        m_ati = initial_momentum(0.8, ArammanaGrade.ATI_MAHANTA)
        m_mah = initial_momentum(0.8, ArammanaGrade.MAHANTA)
        assert m_ati > m_mah

    def test_initial_momentum_uses_absolute_valence(self) -> None:
        """Painful (negative) objects drive as much rumination as pleasant ones."""
        assert initial_momentum(-0.8, ArammanaGrade.MAHANTA) == initial_momentum(0.8, ArammanaGrade.MAHANTA)

    def test_neutral_object_has_zero_momentum(self) -> None:
        assert initial_momentum(0.0, ArammanaGrade.ATI_MAHANTA) == 0.0

    def test_cascade_depth_minimum_enforced(self) -> None:
        """Zero momentum still fires the minimum number of chains."""
        assert cascade_depth(0.0) == config.MANO_DVARA_CASCADE_MIN

    def test_cascade_depth_increases_with_momentum(self) -> None:
        assert cascade_depth(1.0) > cascade_depth(0.3)

    def test_cascade_depth_high_momentum_exceeds_minimum(self) -> None:
        assert cascade_depth(1.0) > config.MANO_DVARA_CASCADE_MIN

    def test_vedana_to_scalar_signs(self) -> None:
        assert vedana_to_scalar(VedanāType.SOMANASSA) > 0
        assert vedana_to_scalar(VedanāType.DOMANASSA) < 0
        assert vedana_to_scalar(VedanāType.UPEKKHA) == 0.0
        assert vedana_to_scalar(VedanāType.SUKHA) > 0
        assert vedana_to_scalar(VedanāType.DUKKHA) < 0


# ---------------------------------------------------------------------------
# One ManoDvaraVithi chain structure
# ---------------------------------------------------------------------------

class TestSingleChain:

    def _chain(self, grade: ArammanaGrade = ArammanaGrade.MAHANTA, vedana_scalar: float = 0.8) -> ManoDvaraVithiResult:
        return ManoDvaraVithi().process(
            chain_index=0, momentum=0.8, vedana_scalar=vedana_scalar, grade=grade
        )

    def test_non_registering_chain_has_10_moments(self) -> None:
        """Bv-Ba-M-J×7 = 10 moments when no tadarammana."""
        result = self._chain(ArammanaGrade.MAHANTA)
        assert len(result.moments) == 10

    def test_registering_chain_has_12_moments(self) -> None:
        """Bv-Ba-M-J×7-D×2 = 12 moments for ATI_MAHANTA."""
        result = self._chain(ArammanaGrade.ATI_MAHANTA)
        assert len(result.moments) == 12

    def test_chain_starts_with_bhavanga(self) -> None:
        result = self._chain()
        assert result.starts_with_bhavanga
        assert result.moments[0].phase == "MOMENT_BHAVANGA_CALANA"
        assert result.moments[1].phase == "MOMENT_BHAVANGA_UPACCHEDA"

    def test_third_moment_is_mind_door_adverting(self) -> None:
        result = self._chain()
        assert result.moments[2].phase == "MOMENT_MIND_DOOR_ADVERTING"
        assert result.moments[2].citta_type.function == CittaFunction.MIND_DOOR_ADVERTING

    def test_javana_fires_seven_times(self) -> None:
        result = self._chain()
        assert len(result.javana_moments) == config.JAVANA_COUNT

    def test_javana_is_single_type(self) -> None:
        """All 7 javana moments are the identical CittaType."""
        result = self._chain()
        types = {m.citta_type.id for m in result.javana_moments}
        assert len(types) == 1

    def test_javana_type_has_javana_function(self) -> None:
        result = self._chain()
        assert result.javana_type.function == CittaFunction.JAVANA

    def test_registration_only_for_ati_mahanta(self) -> None:
        assert self._chain(ArammanaGrade.MAHANTA).registered is False
        assert self._chain(ArammanaGrade.ATI_MAHANTA).registered is True

    def test_tadarammana_moments_have_registering_function(self) -> None:
        result = self._chain(ArammanaGrade.ATI_MAHANTA)
        reg = [m for m in result.moments if m.phase == "MOMENT_REGISTERING"]
        assert len(reg) == 2
        for m in reg:
            assert m.citta_type.function == CittaFunction.REGISTERING

    def test_momentum_decays_after_chain(self) -> None:
        result = ManoDvaraVithi().process(
            chain_index=0, momentum=1.0, vedana_scalar=0.8, grade=ArammanaGrade.MAHANTA
        )
        assert result.momentum_out == pytest.approx(1.0 * config.CETANA_MOMENTUM_DECAY)

    def test_moment_numbers_sequential(self) -> None:
        result = self._chain(ArammanaGrade.ATI_MAHANTA)
        nums = [m.number for m in result.moments]
        assert nums == list(range(1, 13))


# ---------------------------------------------------------------------------
# Stub determiner valence routing
# ---------------------------------------------------------------------------

class TestStubDeterminer:

    def test_pleasant_object_apperceived_wholesomely(self) -> None:
        result = ManoDvaraVithi().process(0, 0.8, vedana_scalar=0.9, grade=ArammanaGrade.MAHANTA)
        assert result.vedana == VedanāType.SOMANASSA
        assert "maha_kusala" in result.javana_type.id

    def test_painful_object_drives_dosa_rumination(self) -> None:
        """A painful mind-door object fires aversive (dosa-rooted) javana."""
        result = ManoDvaraVithi().process(0, 0.8, vedana_scalar=-0.9, grade=ArammanaGrade.MAHANTA)
        assert "dosa" in result.javana_type.id

    def test_neutral_object_equanimous(self) -> None:
        result = ManoDvaraVithi().process(0, 0.0, vedana_scalar=0.0, grade=ArammanaGrade.MAHANTA)
        assert result.vedana == VedanāType.UPEKKHA


# ---------------------------------------------------------------------------
# ★ Cascade acceptance: ≥ MIN chains, momentum-driven, bhavanga between
# ---------------------------------------------------------------------------

class TestCascade:

    def test_neutral_object_fires_minimum_chains(self) -> None:
        """Neutral vedanā → exactly MANO_DVARA_CASCADE_MIN chains."""
        cascade = run_cascade(vedana_scalar=0.0, grade=ArammanaGrade.MAHANTA)
        assert cascade.chain_count == config.MANO_DVARA_CASCADE_MIN

    def test_cascade_always_fires_at_least_minimum(self) -> None:
        for scalar in (0.0, 0.2, 0.5, 0.8):
            for grade in (ArammanaGrade.PARITTA, ArammanaGrade.MAHANTA, ArammanaGrade.ATI_MAHANTA):
                cascade = run_cascade(vedana_scalar=scalar, grade=grade)
                assert cascade.chain_count >= config.MANO_DVARA_CASCADE_MIN

    def test_charged_object_ruminates_more(self) -> None:
        """Strong valence on a vivid object fires more than the minimum chains."""
        cascade = run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA)
        assert cascade.chain_count > config.MANO_DVARA_CASCADE_MIN

    def test_chains_are_indexed_in_order(self) -> None:
        cascade = run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA)
        indices = [c.chain_index for c in cascade.chains]
        assert indices == list(range(cascade.chain_count))

    def test_momentum_monotonically_decreases(self) -> None:
        """Each chain's outgoing momentum is strictly less than the previous."""
        cascade = run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA)
        momenta = cascade.momenta
        for i in range(1, len(momenta)):
            assert momenta[i] < momenta[i - 1]

    def test_every_chain_starts_with_bhavanga(self) -> None:
        """Brief bhavanga between each chain (each chain opens with Bv-Ba)."""
        cascade = run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA)
        for chain in cascade.chains:
            assert chain.starts_with_bhavanga

    def test_cascade_stops_below_threshold(self) -> None:
        """The last chain's outgoing momentum falls below the stop threshold."""
        cascade = run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA)
        # cascade_depth runs while momentum > threshold; after the last counted
        # chain, momentum must have decayed under threshold (unless min floor padded).
        if cascade.chain_count > config.MANO_DVARA_CASCADE_MIN:
            assert cascade.momenta[-1] < config.CETANA_MOMENTUM_THRESHOLD / config.CETANA_MOMENTUM_DECAY

    def test_each_chain_has_seven_javana(self) -> None:
        cascade = run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA)
        for chain in cascade.chains:
            assert len(chain.javana_moments) == config.JAVANA_COUNT

    def test_total_javana_cittas(self) -> None:
        cascade = run_cascade(vedana_scalar=0.0, grade=ArammanaGrade.MAHANTA)
        assert cascade.total_javana_cittas == config.MANO_DVARA_CASCADE_MIN * config.JAVANA_COUNT

    def test_cascade_deterministic(self) -> None:
        c1 = run_cascade(vedana_scalar=0.7, grade=ArammanaGrade.MAHANTA)
        c2 = run_cascade(vedana_scalar=0.7, grade=ArammanaGrade.MAHANTA)
        assert c1.chain_count == c2.chain_count
        assert [c.javana_type.id for c in c1.chains] == [c.javana_type.id for c in c2.chains]


# ---------------------------------------------------------------------------
# Custom determiner (the M6 dreaming/replay reuse path)
# ---------------------------------------------------------------------------

class TestCustomDeterminer:

    def test_custom_determiner_drives_chain(self) -> None:
        """ManoDvaraVithi runs from an arbitrary determiner — the replay hook."""
        target = get("maha_kiriya_upekkha_panna_asankharika")

        def replay_determiner(inp: ManoDeterminerInput) -> ManoDeterminerOutput:
            return ManoDeterminerOutput(javana_type=target, vedana=VedanāType.UPEKKHA)

        cascade = run_cascade(
            vedana_scalar=0.5, grade=ArammanaGrade.MAHANTA, determiner=replay_determiner
        )
        for chain in cascade.chains:
            assert chain.javana_type is target

    def test_determiner_receives_chain_index(self) -> None:
        """Replay/AI determiner can vary by abstraction level (chain index)."""
        seen: list[int] = []

        def recording_determiner(inp: ManoDeterminerInput) -> ManoDeterminerOutput:
            seen.append(inp.chain_index)
            return ManoDeterminerOutput(
                javana_type=get("maha_kusala_upekkha_nopanna_asankharika"),
                vedana=VedanāType.UPEKKHA,
            )

        run_cascade(vedana_scalar=0.0, grade=ArammanaGrade.MAHANTA, determiner=recording_determiner)
        assert seen == list(range(config.MANO_DVARA_CASCADE_MIN))

    def test_determiner_receives_decaying_momentum(self) -> None:
        momenta: list[float] = []

        def recording_determiner(inp: ManoDeterminerInput) -> ManoDeterminerOutput:
            momenta.append(inp.momentum)
            return ManoDeterminerOutput(
                javana_type=get("maha_kusala_upekkha_nopanna_asankharika"),
                vedana=VedanāType.UPEKKHA,
            )

        run_cascade(vedana_scalar=0.9, grade=ArammanaGrade.ATI_MAHANTA, determiner=recording_determiner)
        # Momentum entering each chain decreases
        for i in range(1, len(momenta)):
            assert momenta[i] < momenta[i - 1]


# ---------------------------------------------------------------------------
# Integration: CittaVithi.process_and_cascade
# ---------------------------------------------------------------------------

class TestSenseDoorCascadeIntegration:

    def test_full_vithi_triggers_cascade(self) -> None:
        """A full sense-door vithi is followed by a mind-door cascade."""
        vithi = make_vithi()
        result = vithi.process_and_cascade(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        assert isinstance(result, CognitionResult)
        assert result.cascade is not None
        assert result.cascade.chain_count >= config.MANO_DVARA_CASCADE_MIN

    def test_early_exit_vithi_has_no_cascade(self) -> None:
        """A PARITTA object dropped at moment 8 produces no rumination."""
        vithi = make_vithi()
        result = vithi.process_and_cascade(vision_events(), ArammanaGrade.PARITTA, fake_trigger())
        assert result.sense.early_exit
        assert result.cascade is None

    def test_ati_mahanta_seeds_longer_cascade_than_mahanta(self) -> None:
        """Stub determiner gives ATI_MAHANTA a somanassa javana (high momentum),
        MAHANTA an upekkhā javana (zero momentum, minimum chains)."""
        vithi = make_vithi()
        r_mah = vithi.process_and_cascade(vision_events(), ArammanaGrade.MAHANTA, fake_trigger())
        r_ati = vithi.process_and_cascade(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        assert r_ati.cascade.chain_count >= r_mah.cascade.chain_count
        assert r_mah.cascade.chain_count == config.MANO_DVARA_CASCADE_MIN

    def test_sense_result_preserved_in_cognition(self) -> None:
        vithi = make_vithi()
        result = vithi.process_and_cascade(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        assert result.sense.moment_numbers == list(range(1, 18))
        assert result.sense.javana_vedana is not None

    def test_cascade_uses_sense_javana_vedana(self) -> None:
        """ATI_MAHANTA → somanassa sense-door javana → pleasant mind-door chains."""
        vithi = make_vithi()
        result = vithi.process_and_cascade(vision_events(), ArammanaGrade.ATI_MAHANTA, fake_trigger())
        # somanassa scalar > 0 → stub mano determiner picks wholesome somanassa
        first_chain = result.cascade.chains[0]
        assert first_chain.vedana == VedanāType.SOMANASSA

    def test_custom_mano_determiner_passes_through(self) -> None:
        target = get("maha_kiriya_somanassa_panna_asankharika")

        def md(inp: ManoDeterminerInput) -> ManoDeterminerOutput:
            return ManoDeterminerOutput(javana_type=target, vedana=VedanāType.SOMANASSA)

        vithi = make_vithi()
        result = vithi.process_and_cascade(
            vision_events(), ArammanaGrade.MAHANTA, fake_trigger(), mano_determiner=md
        )
        for chain in result.cascade.chains:
            assert chain.javana_type is target


# ---------------------------------------------------------------------------
# CascadeResult container
# ---------------------------------------------------------------------------

class TestCascadeResultContainer:

    def test_empty_construction(self) -> None:
        cr = CascadeResult(chains=[])
        assert cr.chain_count == 0
        assert cr.total_javana_cittas == 0
        assert cr.momenta == []
