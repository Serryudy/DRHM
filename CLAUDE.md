# CLAUDE.md

Guidance for any agent (or human) developing **`digital-replica-of-a-human-mind`** — an
Abhidhamma-grounded neuromorphic cognitive agent. Read this before writing code so that
contributions stay architecturally consistent. The authoritative design rationale lives in
`docs/`; this file is the engineering contract derived from it.

---

## 1. What we are building (one paragraph)

A continuously-running, event-driven cognitive agent whose architecture is isomorphic to the
Theravada **Abhidhamma** model of mind. Sensory hardware (camera = eyes, microphone = ears,
touchscreen = skin, speakers = mouth) feeds an asynchronous spiking substrate. The substrate
rests in a low-energy **bhavanga** attractor and is perturbed — never polled — by sensory
spikes. Each perturbation drives a **17-moment cognitive cycle (citta-vithi)** implemented as a
deterministic state machine over a Spiking Neural Network (SNN). Percepts are grounded as
geometric **Conceptual Spaces**, encoded as **Vector-Symbolic / hyperdimensional** vectors,
routed by **Conceptors**, and acted upon by an **Active Inference** engine that minimizes
expected free energy during the *javana* (impulsion) phase. The agent learns continuously and
locally (STDP + astrocyte-gated metaplasticity) while awake, and consolidates memory offline
during **sleep/dreaming** (slow-wave synaptic downscaling + *manodvaravithi* replay) to avoid
catastrophic forgetting.

The non-negotiable invariant of this project: **no polling loops, no global-clock frame
processing, no `while True: read_frame()` heuristics.** Cognition is event-driven and
energy-proportional. If a change is observed, the change itself must force the network out of
equilibrium.

---

## 2. Source docs (read in this order)

1. `docs/Architectural Blueprint for an Abhidhamma Grounded Neuromorphic Agent...md` — the master
   spec. Sections 2–6 define the sensory substrate, the 17-moment table, semantic grounding,
   plasticity, and sleep.
2. `docs/The Architecture of Transcendence...md` — §7 gives the **6-phase processing synthesis**
   table (Baseline → Apprehension → Assimilation → Determination → Impulsion → Consolidation)
   and §8 the three energy-efficiency mechanisms. Use this table as the module contract.
3. `docs/The Architecture of "X"...md` — the theory of the pre-linguistic unit "X" (Conceptual
   Spaces → VSA → Conceptors → Active Inference). Reference when implementing semantics.
4. `docs/implementation phases.md` — the phased roadmap and the four canonical test protocols.

When the docs and this file disagree, the docs win on *intent*; update this file to match.

---

## 3. Canonical vocabulary (use these names in code)

Map Pali/phenomenological terms to code identifiers **consistently**. Class/module names use the
Pali term; a short English gloss goes in the docstring.

| Pali / concept        | Code identifier                | Meaning                                              |
|-----------------------|--------------------------------|------------------------------------------------------|
| bhavanga              | `Bhavanga` / `bhavanga`        | Resting attractor / life-continuum (idle state)      |
| citta-vithi           | `CittaVithi`                   | The 17-moment cognitive cycle (the state machine)    |
| citta                 | `Citta`                        | A single thought-moment / FSM state                  |
| cetasika              | `Cetasika`                     | Mental factor modulating a citta                     |
| pancadvaravajjana     | `MOMENT_ADVERTING`             | Moment 4: five-door adverting / attention routing    |
| pancavinnana          | `MOMENT_SENSE`                 | Moment 5: raw sense consciousness / feature encoding |
| sampaticchana         | `MOMENT_RECEIVING`             | Moment 6: receiving / spatiotemporal aggregation     |
| santirana             | `MOMENT_INVESTIGATING`         | Moment 7: investigating / pattern matching           |
| votthapana            | `MOMENT_DETERMINING`           | Moment 8: determining / **salience gate**            |
| javana                | `MOMENT_JAVANA` (×7)           | Moments 9–15: impulsion / Active Inference loop      |
| tadarammana           | `MOMENT_REGISTERING` (×2)      | Moments 16–17: registration / plasticity commit      |
| thina / middha        | `sloth` / `torpor`             | Sleep-pressure factors (trigger sleep mode)          |
| manodvaravithi        | `ManoDvaraVithi`               | Mind-door process = dreaming / offline replay        |

Do **not** invent alternative spellings (e.g. `votthapanna`, `javanna`). Grep for the term before
adding it.

---

## 4. Target module layout

Build under a top-level package `drhm/` (digital replica of a human mind). Create directories as
phases land — do not stub everything up front.

```
drhm/
  sensory/            # Phase 1 — async, event-driven I/O. NO polling.
    events.py         #   SpikeEvent dataclass; async event bus (asyncio.Queue based)
    vision.py         #   camera -> DVS-style change events (log-intensity threshold)
    audition.py       #   mic -> silicon-cochlea bandpass -> spike events
    touch.py          #   touchscreen -> capacitive spike events
    vocalization.py   #   motor spikes -> audio out (babbling / articulatory synth)
    digital/          # Phase 1.5 — digital embodiment (the computer as body)
      screen.py       #     screen capture -> DvsEncoder (mss / X11 DAMAGE)
      loopback.py     #     system-audio loopback -> CochleaEncoder (PipeWire monitor)
      peripherals.py  #     mouse/keyboard -> Modality.PROPRIOCEPTION (keys hashed)
      _x11.py         #     active-window geometry for focus adverting
  attention/          # Phase 1.5 — adverting front-end (citta-vithi moments 1-4 + gate 8)
    salience.py       #   SalienceGate: predictive novelty -> 4 arammana grades (moghavara drop)
    focus.py          #   FocusRouter: single-object spatial adverting (pancadvaravajjana)
    frontend.py       #   AttentionFrontend: focus -> grade -> drop the futile
  recording/          # Phase 1.5 — session capture for later (M2-M6) consolidation
    session.py        #   SessionRecorder: graded spike stream -> local JSONL (opt-in, pausable)
  snn/                # Phase 2 — spiking substrate
    neurons.py        #   LIF (and optional Izhikevich) neuron models
    bhavanga.py       #   resting attractor; perturbation detection (moments 1-3)
    citta_vithi.py    #   the 17-moment deterministic state machine
  semantics/          # Phase 3 — the unit "X"
    conceptual_space.py  # quality dimensions, convex regions, prototypes (Voronoi)
    vsa.py               # hypervectors: bind (⊗), bundle (⊕), permute (Π). D=10_000
    conceptors.py        # C = R(R + a^-2 I)^-1 ; Boolean AND/OR/NOT on conceptors
  inference/          # Phase 3/4 — volition
    active_inference.py  # generative world model, free energy, expected free energy (EFE)
    javana.py            # the 7-step impulsion loop driven by EFE + conceptors
  learning/           # Phase 4 — plasticity
    stdp.py              # local spike-timing-dependent plasticity
    agmp.py              # astrocyte-gated metaplasticity via ISI coefficient-of-variation
  sleep/              # Phase 4 — consolidation
    slow_wave.py         # synaptic homeostasis / global downscaling (deep bhavanga)
    replay.py            # manodvaravithi: noise-injected episodic replay (SWR), interleaving
  memory/
    episodic.py          # fast volatile buffer (hippocampus analogue)
    neocortical.py       # slow distributed store (consolidation target)
  agent.py            # wires the subsystems into one running organism
  config.py           # all magic numbers (thresholds, D, apertures, time constants)
tests/                # mirrors drhm/ ; see §8 for the four acceptance protocols
```

`config.py` holds every tunable constant (firing threshold, membrane time constant τ,
hypervector dimensionality `D`, conceptor aperture α, STDP windows, sleep-pressure threshold).
**No magic numbers inline** — import from config so experiments are reproducible.

---

## 5. Tech stack & dependencies

Python 3.12, virtualenv already at `.venv/`. Currently installed: `numpy`, `opencv-python`,
`sounddevice`, `cffi`. Add libraries deliberately, pinning versions, and record why.

| Concern              | Preferred library                | Notes                                                    |
|----------------------|----------------------------------|----------------------------------------------------------|
| Async sensory I/O    | stdlib `asyncio`                 | The backbone. Everything sensory is a coroutine.         |
| Camera / audio       | `opencv-python`, `sounddevice`   | Already present. Use as *event sources*, not frame loops.|
| SNN substrate        | `Nengo` **or** `BindsNET`        | Pick one in Phase 2; document the choice. Don't mix.     |
| VSA / hypervectors   | `torchhd`                        | Pulls in `torch`. Or a thin numpy impl if torch is too heavy for edge. |
| Conceptors / linalg  | `numpy` (+ `scipy` if needed)    | Conceptors are closed-form matrix ops.                   |
| Tests                | `pytest`, `pytest-asyncio`       | Add in Phase 1.                                          |

Decision rule for new deps: if numpy/scipy/stdlib can do it cleanly, do not add a dependency. The
project values a small, auditable, edge-deployable footprint (§8 of the Transcendence doc:
energy proportionality is a goal, not an afterthought).

Always install via `.venv/bin/pip` and append to `requirements.txt` with a pinned version.

---

## 6. Coding conventions & architectural rules

These are enforced by review. Violating an invariant is a bug regardless of test status.

1. **Event-driven, never polled.** Sensory modules MUST emit events from async callbacks /
   hardware interrupts / blocking reads in executors — never a busy `while True` that compares
   frames. Idle CPU must approach zero (see Perturbation Test). Any `cv2.absdiff`-style
   change-detection-by-polling is explicitly forbidden in production paths.
2. **One event type.** All sensory output is a `SpikeEvent` (timestamp, modality, channel,
   payload). Downstream code consumes the unified spike stream; modality-specific structure stays
   in the sensory layer.
3. **The 17 moments are sequential and total.** A perturbed `CittaVithi` MUST pass through its
   gating sequence in order and return to `Bhavanga`. The salience gate at moment 8
   (`MOMENT_DETERMINING`) is the ONLY sanctioned early-exit: zero surprise → drop straight back to
   bhavanga (energy proportionality). Never skip moments silently elsewhere.
4. **Javana is a fixed 7-step recurrent loop**, not a deep feedforward stack. Depth comes from
   recurrence + conceptor algebra, not added layers.
5. **Learning is local and gradient-free in the running agent.** STDP/AGMP only. No
   backpropagation, no global loss, no offline epoch-based gradient training in the live cognitive
   path. (Pre-training auxiliary perception nets offline is acceptable only if clearly quarantined
   and documented.)
6. **Plasticity commits at registration.** Synaptic changes from a cycle are buffered as
   eligibility traces during javana and only consolidated at `MOMENT_REGISTERING` (16–17).
7. **Sleep is mandatory, not optional.** Continuous waking STDP must be balanced by slow-wave
   downscaling + replay or the system will drift and forget. The sleep subsystem is a
   first-class part of correctness, not a feature.
8. **Determinism where it matters.** The FSM and conceptor algebra must be deterministic and
   unit-testable; seed all RNGs (`numpy.random.default_rng(seed)` from config) so hypervector
   generation and replay noise are reproducible.
9. **Style.** Type hints everywhere. `dataclasses` for state. Docstrings name the Abhidhamma term
   and cite the doc section. Keep modules small and single-responsibility. Format with `ruff`
   (add in Phase 1); no unformatted code merges.
10. **Numerics.** Hypervectors default to bipolar {-1,+1} at `D=10_000` (config). Conceptors are
    PSD matrices with singular values in [0,1]; assert this in tests. Use cosine similarity for
    hypervector comparison.

---

## 7. Development workflow

- **Branch/PR per phase-milestone.** Each milestone in §8 is a reviewable unit with its named
  acceptance test green.
- **Test-first for the FSM and algebra.** The citta-vithi sequence, VSA ops, and conceptor
  Boolean logic have exact expected behavior — write the test before the implementation.
- **Run the suite before declaring done:** `.venv/bin/python -m pytest -q`. Report real results;
  if a protocol test is skipped or slow (e.g. the 24h Perturbation Test), say so explicitly and
  provide the short-form variant that ran.
- **Keep `config.py` and `requirements.txt` authoritative.** Any tuned constant or new dep lands
  there in the same change.
- **Document choices in code.** When you pick Nengo vs BindsNET, or torchhd vs numpy-VSA, leave a
  short ADR-style note at the top of the relevant module.
- When the architecture evolves, update this file and the relevant `docs/` section in the same PR.

---

## 8. Full development & test plan

Six milestones grouped into the four roadmap phases. Each milestone lists **objective**, **build**,
and a **named acceptance test**. The four bold protocols are the canonical gates from
`docs/implementation phases.md`; the others are supporting unit/integration tests.

### Phase 1 — Asynchronous sensory substrate (`drhm/sensory/`)

**M1 · Event-driven I/O**
- *Objective:* zero-polling environment; physical change triggers computation.
- *Build:* `SpikeEvent`; an `asyncio`-based event bus; `vision.py` (log-intensity change →
  events), `audition.py` (bandpass channels → threshold spikes), `touch.py`, `vocalization.py`.
  Camera/mic are wrapped as event sources via executor threads / callbacks — **no frame-diff
  polling loop in the hot path.**
- ★ **Perturbation Test (acceptance):** run idle in a dark/quiet room; CPU utilization must stay
  near zero (assert mean CPU below a small threshold over the window). Inject a sudden
  light/sound; assert an event is emitted within a bounded latency and nothing is dropped.
  Provide a fast CI variant (seconds) and document the long-form (24h) protocol.
- *Supporting tests:* event ordering/timestamp monotonicity; backpressure (queue never
  busy-spins); each modality emits the correct `SpikeEvent` schema.
- *Status:* **DONE.** `drhm/sensory/` (`events`, `vision`, `audition`, `touch`, `vocalization`).

### Phase 1.5 — Digital embodiment & the attention front-end (§8.5) — DONE

The agent's body becomes the computer: screen pixels are its eyes, system audio its ears,
mouse/keyboard a new proprioceptive sense. This reuses the M1 encoders (no new sensing logic,
no `cv2.absdiff` polling — CLAUDE.md §6.1) and adds the **adverting front-end**, which is the
front of the citta-vīthi (moments 1–4 + the moment-8 grade gate) grounded in deep research on
the Abhidhamma account of attention (sources: abhidhamma.com, verified).

- *Build (`drhm/sensory/digital/`):* `screen.py` (`ScreenVisionSource`: mss / X11-DAMAGE →
  `DvsEncoder`), `loopback.py` (`LoopbackAuditionSource`: PipeWire monitor → `CochleaEncoder`),
  `peripherals.py` (`PeripheralSource`: pynput → `Modality.PROPRIOCEPTION`, keystrokes **hashed**
  for privacy). Live backends import lazily; `.feed*()` drives them headless.
- *Build (`drhm/attention/`):* `salience.py` (`SalienceGate` — per-region predictive novelty →
  the four `ārammaṇa` grades; `ATI_PARITTA` = `moghavāra`, dropped); `focus.py` (`FocusRouter` —
  single-object spatial adverting via the focused window); `frontend.py` (`AttentionFrontend` —
  focus → grade → drop the futile).
- *Build (`drhm/recording/`):* `SessionRecorder` logs the graded multimodal spike stream to a
  local JSONL session (opt-in, pausable) so real usage becomes a dataset M2–M6 will consolidate.
- *Keystone test:* a **repetitive** region (ticking clock) habituates to `ATI_PARITTA` and is
  dropped; a **changing** region (course video) keeps passing. Plus focus-gating, recorder
  round-trip, and digital-source `.feed()` tests.
- *Doctrine → architecture (the ārammaṇa-grade → process-depth mapping):*

  | `ArammanaGrade` | Abhidhamma | Attention action | Future FSM depth (M3) |
  |-----------------|-----------|------------------|------------------------|
  | `ATI_PARITTA` (very slight) | `moghavāra` — "not known at all" | **dropped** | no process |
  | `PARITTA` (slight) | stops at *votthapana* | published, tagged | early-exit at moment 8 |
  | `MAHANTA` (great) | runs to *javana* | published | full process |
  | `ATI_MAHANTA` (very great) | runs to *tadārammaṇa* | published | full + registration/learn |

  Novelty (= surprise) is the crude precursor of the moment-8 free-energy gate; the EMA
  expectation becomes the Active-Inference generative model in M4.

### Phase 2 — Spiking substrate & the cognitive state machine (`drhm/snn/`)

**M2 · SNN core (bhavanga + LIF)**
- *Objective:* a resting attractor that is perturbed, not polled.
- *Build:* `neurons.py` LIF dynamics (leak, integrate, threshold, reset, refractory);
  `bhavanga.py` low-frequency recurrent resting state; moments 1–3 = spike-driven membrane
  accumulation crossing threshold (`bhavanga-upaccheda`).
- *Acceptance:* **Idle-stability test** — with no input, membrane potentials decay to baseline and
  the network stays in the attractor; injected spikes raise potentials and, above threshold, exit
  bhavanga. No exit without sufficient input.

**M3 · The 17-moment Citta-Vithi FSM (`citta_vithi.py`)**
- *Objective:* the 17 discrete thought-moments as a deterministic automaton over the SNN.
- *Build:* states for moments 1–17 (`MOMENT_*`), strict transitions, the moment-8 salience gate,
  the 7× javana sub-loop, the 2× registration tail, then return to `Bhavanga`.
- ★ **Sequence Integrity Test (acceptance):** inject an artificial spike train; trace the
  cascade; assert it passes through **all 17 distinct gating phases in order** and returns to
  bhavanga. Separately assert the moment-8 gate can early-exit on zero-surprise input
  (energy-proportionality path) — and ONLY at moment 8.
- *Supporting tests:* transitions are pure/deterministic given (state, input); no skipped or
  reordered moments under fuzzed input.

### Phase 3 — Semantic binding & active inference (`drhm/semantics/`, `drhm/inference/`)

**M4 · The unit "X": Conceptual Spaces + VSA + Conceptors**
- *Objective:* turn raw SNN features into grounded, composable concepts.
- *Build:* `conceptual_space.py` (quality dimensions, convex regions, prototypes/Voronoi);
  `vsa.py` (`bind`, `bundle`, `permute`, similarity at `D=10_000`); `conceptors.py`
  (`C = R(R + α⁻²I)⁻¹`, plus `NOT = I−C`, `AND`, `OR`).
- ★ **Conceptor Logic Test (acceptance):** feed two distinct stimuli (a tone + a shape); assert
  VSA `bind` produces a vector **orthogonal** (≈0 cosine) to both operands while `bundle` stays
  **similar** to its constituents; assert `permute` makes "A then B" ≠ "B then A". Assert
  conceptor Boolean ops behave (intersection/union/complement) and singular values ∈ [0,1].
- *Supporting tests:* convexity check (a point between two same-category instances is in-category);
  prototype recall; hypervector noise tolerance (random bit flips don't change nearest concept).

**M5 · Volitional engine: Active Inference in Javana (`drhm/inference/`)**
- *Objective:* moment-8 gating + the 7-step javana loop driven by free energy.
- *Build:* `active_inference.py` (generative world model; variational free energy = surprise;
  Expected Free Energy scoring policies = epistemic + pragmatic terms); `javana.py` wires EFE +
  conceptor dynamics into the recurrent 7-moment impulsion loop and emits motor/vocal spikes.
- *Acceptance (extends Conceptor Logic Test):* novel stimulus combinations register as **high
  free energy / surprising**; familiar combinations as low. The moment-8 gate opens only on high
  EFE. Actions chosen reduce expected free energy in a toy environment (epistemic foraging
  observable).
- *Supporting tests:* EFE balances exploration vs exploitation; deterministic given seed.

### Phase 4 — Continuous ontogeny & sleep consolidation (`drhm/learning/`, `drhm/sleep/`)

**M6 · Plasticity + sleep/dreaming**
- *Objective:* lifelong local learning without catastrophic forgetting.
- *Build:* `stdp.py` (LTP/LTD by spike timing); `agmp.py` (astrocytes monitor ISI
  coefficient-of-variation; low-CV/stable synapses get a reduced learning rate, high-CV stay
  plastic); `slow_wave.py` (global synaptic downscaling preserving relative weights — deep
  bhavanga); `replay.py` (`manodvaravithi`: inject noise into `memory/episodic.py`, replay
  hypervector/spike sequences time-compressed, interleave with old memories, consolidate into
  `memory/neocortical.py` via STDP). Sleep is entered when `sloth`/`torpor` pressure crosses the
  config threshold.
- ★ **Catastrophic Forgetting Test (acceptance):** train Task A (recognize pattern A) to stable
  recall; train heavily on Task B; **without sleep** assert Task A degrades; run the
  sleep/dream cycle; **after sleep** assert BOTH Task A and Task B are retained above threshold.
- *Supporting tests:* STDP strengthens pre-before-post / weakens post-before-pre; AGMP shields
  low-CV synapses (their effective learning rate drops); slow-wave downscaling preserves weight
  *ordering* while reducing total synaptic mass; replay interleaving touches old + new memories.

### Phase 5 — Integration (`drhm/agent.py`)

**M7 · The whole organism**
- *Objective:* one process: sense → perturb bhavanga → run citta-vithi → bind/infer/act → learn →
  sleep → repeat, indefinitely, event-driven.
- *Acceptance — End-to-end "day-in-the-life" test:* run the agent against a scripted sensory
  timeline (quiet → stimulus bursts → quiet → rising sleep pressure → sleep → wake). Assert:
  idle CPU stays low between stimuli (Perturbation invariant holds in the full system); each
  stimulus produces a complete logged citta-vithi; concepts accumulate; a sleep cycle fires and
  consolidates; post-wake recall of earlier-day stimuli is retained. This is the standing
  regression test for the integrated system.

### Test matrix summary

| Protocol (★ = canonical gate)   | Phase | Verifies                                              |
|---------------------------------|-------|-------------------------------------------------------|
| ★ Perturbation Test             | 1     | Zero-polling idle; event-driven spike on stimulus     |
| Idle-stability test             | 2     | Bhavanga attractor; threshold-gated exit              |
| ★ Sequence Integrity Test       | 2     | All 17 moments in order; only moment-8 early-exit      |
| ★ Conceptor Logic Test          | 3     | bind/bundle/permute + conceptor Boolean algebra       |
| Active-inference / EFE tests    | 3     | Surprise gating; epistemic foraging                   |
| ★ Catastrophic Forgetting Test  | 4     | Sleep/replay retains old + new tasks                  |
| Day-in-the-life integration     | 5     | All invariants hold in the running organism           |

---

## 9. Definition of done (per change)

- The relevant named acceptance test(s) pass via `.venv/bin/python -m pytest`.
- No polling loop introduced in any sensory/cognitive hot path (§6.1).
- New constants in `config.py`; new deps pinned in `requirements.txt`.
- Abhidhamma terms used per §3; docstrings cite the doc section.
- This file / `docs/` updated if the architecture changed.

---

## 10. Quick reference: the 6-phase processing pipeline (Transcendence §7)

```
I.   Baseline      moments 1-3   → async event detection interrupts bhavanga
II.  Apprehension  moments 4-5   → SNN feature extraction → Conceptual Space / image schema
III. Assimilation  moments 6-7   → VSA bind/bundle into composite hypervector "X"
IV.  Determination moment 8      → free-energy salience gate (early-exit if zero surprise)
V.   Impulsion     moments 9-15  → 7-step Conceptor + Active-Inference javana loop
VI.  Consolidation moments 16-17 → STDP / ISI-CV metaplasticity commit; back to bhavanga
```

Energy proportionality comes from three compounding mechanisms (Transcendence §8): event-driven
sparsity (idle = dormant), salience gating at moment 8 (skip the expensive javana loop on
predictable input), and recurrent javana instead of deep feedforward stacks. Honor all three.
