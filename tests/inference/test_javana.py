"""★ Active Inference / EFE acceptance tests — M5 gate (CLAUDE.md §8 M5).

Acceptance criteria:
  1. Novel stimuli → high surprise → gate opens → pañña javana type selected
     (epistemic foraging observable).
  2. Familiar stimuli → low surprise → EFE gate fires → early exit (None type).
  3. ATI_MAHANTA objects bypass the gate even when familiar.
  4. PARITTA always early-exits regardless of surprise.
  5. Pleasant percept → somanassa type; neutral → upekkhā type.
  6. Painful percept → upekkhā-pañña preferred over dosa (equanimous response).
  7. Anusaya bias shifts type selection correctly.
  8. EFE is deterministic given the same model state and inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from drhm import config
from drhm.citta.cetasikas import Cetasika
from drhm.citta.types import get
from drhm.citta.vedana import VedanāType
from drhm.inference.active_inference import GenerativeModel
from drhm.inference.javana import JavanaDeterminer, Policy, efe_score, _build_default_policies
from drhm.sensory.events import ArammanaGrade
from drhm.snn.citta_vithi import DeterminerInput, DeterminerOutput

D_SMALL = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_model(D: int = D_SMALL, eta: float = 0.3) -> GenerativeModel:
    return GenerativeModel(D=D, eta=eta)


def fresh_det(model: GenerativeModel | None = None, **kwargs) -> JavanaDeterminer:
    m = model or fresh_model()
    return JavanaDeterminer(model=m, **kwargs)


def det_input(
    grade: ArammanaGrade = ArammanaGrade.MAHANTA,
    vedana: float = 0.0,
    x_t: np.ndarray | None = None,
) -> DeterminerInput:
    return DeterminerInput(grade=grade, events=[], vedana_scalar=vedana, x_t=x_t)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(config.SEED)


def random_hv(rng: np.random.Generator, D: int = D_SMALL) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=D)


# ── ★ Acceptance 1: novel → gate open → pañña type ───────────────────────────

class TestAcceptanceNovelEpistemicForaging:

    def test_acceptance_novel_pleasant_selects_panna_type(self, rng: np.random.Generator) -> None:
        """Novel pleasant percept → high surprise → pañña type (epistemic foraging)."""
        det = fresh_det()
        x_t = random_hv(rng)
        inp = det_input(grade=ArammanaGrade.MAHANTA, vedana=0.8, x_t=x_t)
        out = det(inp)
        assert out.javana_type is not None, "Gate must OPEN for novel stimulus"
        assert Cetasika.PANNA in out.javana_type.cetasika_profile, (
            f"Expected pañña type for novel+pleasant, got {out.javana_type.id}"
        )

    def test_acceptance_novel_neutral_selects_panna_type(self, rng: np.random.Generator) -> None:
        """Novel neutral percept → pañña type preferred via epistemic foraging."""
        det = fresh_det()
        x_t = random_hv(rng)
        inp = det_input(grade=ArammanaGrade.MAHANTA, vedana=0.0, x_t=x_t)
        out = det(inp)
        assert out.javana_type is not None
        assert Cetasika.PANNA in out.javana_type.cetasika_profile

    def test_acceptance_novel_painful_selects_upekkha_panna_not_dosa(
        self, rng: np.random.Generator
    ) -> None:
        """Novel painful percept → upekkhā-pañña preferred over dosa (equanimous wisdom).

        EFE for dosa = (−0.8 − (−0.8))² − 0           = 0
        EFE for upekkhā-pañña = (−0.8 − 0)²   − surprise = 0.64 − surprise

        When surprise ≈ 1.0 (novel):  upekkhā-pañña EFE = 0.64 − 1.0 = −0.36 < 0 (wins)
        """
        det = fresh_det(epistemic_weight=1.0)
        x_t = random_hv(rng)
        inp = det_input(grade=ArammanaGrade.MAHANTA, vedana=-0.8, x_t=x_t)
        out = det(inp)
        assert out.javana_type is not None
        # upekkhā-pañña must win over dosa
        assert out.javana_type.id == "maha_kusala_upekkha_panna_asankharika", (
            f"Expected upekkhā-pañña for painful+novel, got {out.javana_type.id}"
        )


# ── ★ Acceptance 2: familiar → gate closes ────────────────────────────────────

class TestAcceptanceFamiliarGate:

    def test_acceptance_familiar_mahanta_gets_early_exit(self, rng: np.random.Generator) -> None:
        """After habituation, MAHANTA stimulus gets early-exit at moment 8."""
        m = fresh_model(eta=0.5)
        det = fresh_det(model=m, surprise_gate=config.EFE_SURPRISE_GATE)
        x_t = random_hv(rng)
        # Habituate the model
        for _ in range(30):
            m.observe(x_t)
        assert m.surprise(x_t) < config.EFE_SURPRISE_GATE, "Model must be habituated"
        inp = det_input(grade=ArammanaGrade.MAHANTA, vedana=0.5, x_t=x_t)
        out = det(inp)
        assert out.javana_type is None, "Familiar MAHANTA must early-exit"

    def test_familiar_paritta_always_exits(self, rng: np.random.Generator) -> None:
        """PARITTA grade always early-exits regardless of surprise level."""
        det = fresh_det()
        x_t = random_hv(rng)
        inp = det_input(grade=ArammanaGrade.PARITTA, vedana=0.8, x_t=x_t)
        out = det(inp)
        assert out.javana_type is None

    def test_familiar_paritta_even_without_x_t(self) -> None:
        det = fresh_det()
        inp = det_input(grade=ArammanaGrade.PARITTA, vedana=0.9)
        assert det(inp).javana_type is None


# ── ★ Acceptance 3: ATI_MAHANTA bypasses gate ────────────────────────────────

class TestAcceptanceAtiMahanta:

    def test_acceptance_ati_mahanta_fires_even_when_familiar(self, rng: np.random.Generator) -> None:
        """ATI_MAHANTA objects must proceed to full processing even when familiar.

        CLAUDE.md: 'ATI_MAHANTA objects are exempt — they always proceed to registration.'
        """
        m = fresh_model(eta=0.9)
        det = fresh_det(model=m, surprise_gate=0.9)  # very high gate
        x_t = random_hv(rng)
        for _ in range(50):
            m.observe(x_t)
        assert m.surprise(x_t) < 0.9, "Model must be well below gate"
        inp = det_input(grade=ArammanaGrade.ATI_MAHANTA, vedana=0.7, x_t=x_t)
        out = det(inp)
        assert out.javana_type is not None, "ATI_MAHANTA must bypass the gate"


# ── ★ Acceptance 4: vedanā alignment ─────────────────────────────────────────

class TestAcceptanceVedanaAlignment:

    def test_acceptance_pleasant_selects_somanassa_type(self, rng: np.random.Generator) -> None:
        """Pleasant percept (vedanā_scalar = +0.9) → somanassa javana type."""
        det = fresh_det()
        x_t = random_hv(rng)
        out = det(det_input(ArammanaGrade.MAHANTA, vedana=0.9, x_t=x_t))
        assert out.javana_type is not None
        assert VedanāType.SOMANASSA in out.javana_type.permitted_vedana

    def test_acceptance_neutral_selects_upekkha_type(self, rng: np.random.Generator) -> None:
        """Neutral percept (vedanā_scalar ≈ 0) → upekkhā javana type."""
        det = fresh_det()
        x_t = random_hv(rng)
        out = det(det_input(ArammanaGrade.MAHANTA, vedana=0.0, x_t=x_t))
        assert out.javana_type is not None
        assert VedanāType.UPEKKHA in out.javana_type.permitted_vedana

    def test_output_vedana_matches_type(self, rng: np.random.Generator) -> None:
        """DeterminerOutput.vedana must be consistent with the selected type."""
        det = fresh_det()
        x_t = random_hv(rng)
        out = det(det_input(ArammanaGrade.MAHANTA, vedana=0.8, x_t=x_t))
        if out.javana_type is not None:
            assert out.vedana in out.javana_type.permitted_vedana


# ── ★ Acceptance 5: deterministic given seed ──────────────────────────────────

class TestAcceptanceDeterminism:

    def test_same_input_same_output(self, rng: np.random.Generator) -> None:
        """Identical model state + inputs must produce identical DeterminerOutput."""
        x_t = random_hv(rng)
        m1 = fresh_model(eta=0.3)
        m2 = fresh_model(eta=0.3)
        det1 = fresh_det(model=m1)
        det2 = fresh_det(model=m2)
        inp = det_input(ArammanaGrade.MAHANTA, vedana=0.5, x_t=x_t)
        out1 = det1(inp)
        out2 = det2(inp)
        assert out1.javana_type == out2.javana_type
        assert out1.vedana == out2.vedana


# ── ★ Acceptance 6: anusaya bias shifts selection ────────────────────────────

class TestAcceptanceAnusaya:

    def test_negative_bias_favours_target_type(self, rng: np.random.Generator) -> None:
        """Negative anusaya bias on a type makes it more likely to be chosen."""
        x_t = random_hv(rng)
        target_id = "maha_kusala_upekkha_nopanna_asankharika"
        det_biased = fresh_det(
            anusaya_bias={target_id: -5.0},   # strong habitual pull toward this type
        )
        inp = det_input(ArammanaGrade.MAHANTA, vedana=0.0, x_t=x_t)
        out = det_biased(inp)
        assert out.javana_type is not None
        assert out.javana_type.id == target_id, (
            f"Anusaya bias should have selected {target_id}, got {out.javana_type.id}"
        )

    def test_positive_bias_suppresses_target_type(self, rng: np.random.Generator) -> None:
        """Positive anusaya bias on the most-preferred type forces another to win."""
        x_t = random_hv(rng)
        # Block pañña somanassa with a large positive penalty
        det_biased = fresh_det(
            anusaya_bias={"maha_kusala_somanassa_panna_asankharika": 10.0},
        )
        inp = det_input(ArammanaGrade.MAHANTA, vedana=0.8, x_t=x_t)
        out = det_biased(inp)
        if out.javana_type is not None:
            assert out.javana_type.id != "maha_kusala_somanassa_panna_asankharika"


# ── EFE scoring unit tests ────────────────────────────────────────────────────

class TestEfeScore:

    def test_pragmatic_zero_at_preferred_valence(self) -> None:
        """EFE has no pragmatic cost when vedanā matches the policy's preference."""
        p = Policy(get("maha_kusala_somanassa_panna_asankharika"), +0.8, True)
        score = efe_score(p, vedana_scalar=0.8, surprise=0.0, epistemic_weight=1.0)
        assert score == pytest.approx(0.0)

    def test_epistemic_bonus_reduces_score(self) -> None:
        """Pañña types score lower (better) than nopañña for the same pragmatics."""
        panna = Policy(get("maha_kusala_upekkha_panna_asankharika"),   0.0, True)
        nopanna = Policy(get("maha_kusala_upekkha_nopanna_asankharika"), 0.0, False)
        surprise = 0.8
        s_panna   = efe_score(panna,   0.0, surprise, 1.0)
        s_nopanna = efe_score(nopanna, 0.0, surprise, 1.0)
        assert s_panna < s_nopanna

    def test_mismatched_valence_increases_pragmatic_cost(self) -> None:
        p = Policy(get("maha_kusala_somanassa_panna_asankharika"), +0.8, True)
        neutral_score  = efe_score(p, 0.8, 0.0, 1.0)
        mismatch_score = efe_score(p, -0.5, 0.0, 1.0)
        assert mismatch_score > neutral_score

    def test_anusaya_bias_additive(self) -> None:
        p = Policy(get("maha_kusala_upekkha_panna_asankharika"), 0.0, True)
        s0 = efe_score(p, 0.0, 0.5, 1.0, anusaya_bias=0.0)
        s1 = efe_score(p, 0.0, 0.5, 1.0, anusaya_bias=-2.0)
        assert s1 == pytest.approx(s0 - 2.0)


# ── JavanaDeterminer API ──────────────────────────────────────────────────────

class TestJavanaDeterminerApi:

    def test_observe_delegates_to_model(self, rng: np.random.Generator) -> None:
        m = fresh_model()
        det = fresh_det(model=m)
        x_t = random_hv(rng)
        assert m.n_obs == 0
        det.observe(x_t)
        assert m.n_obs == 1

    def test_model_property_accessible(self) -> None:
        m = fresh_model()
        det = fresh_det(model=m)
        assert det.model is m

    def test_fallback_to_vedana_proxy(self) -> None:
        """When x_t is None, the determiner uses a 1-D vedanā proxy."""
        m = GenerativeModel(D=1, eta=0.5)
        det = JavanaDeterminer(model=m)
        inp = DeterminerInput(grade=ArammanaGrade.MAHANTA, events=[], vedana_scalar=0.8)
        out = det(inp)
        # No crash; either fires or early-exits depending on prior
        assert isinstance(out, DeterminerOutput)

    def test_custom_policies_respected(self, rng: np.random.Generator) -> None:
        """Custom policy list limits the selection to only those policies."""
        only = Policy(get("maha_kusala_upekkha_nopanna_asankharika"), 0.0, False)
        det = fresh_det(policies=[only])
        x_t = random_hv(rng)
        out = det(det_input(ArammanaGrade.MAHANTA, vedana=0.0, x_t=x_t))
        if out.javana_type is not None:
            assert out.javana_type.id == "maha_kusala_upekkha_nopanna_asankharika"

    def test_default_policies_count(self) -> None:
        assert len(_build_default_policies()) == 6

    def test_all_default_policies_are_javana(self) -> None:
        for p in _build_default_policies():
            assert p.citta_type.function.value == "javana"
