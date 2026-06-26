"""Central configuration: every tunable constant lives here (CLAUDE.md §4, §6.9).

No magic numbers inline anywhere else in the codebase. Import from here so that
experiments stay reproducible. All RNGs are seeded from ``SEED``.
"""

from __future__ import annotations

# --- Global -----------------------------------------------------------------
SEED: int = 1729  # seed for every numpy.random.default_rng in the project

# --- Event bus (drhm/sensory/events.py) -------------------------------------
# 0 == unbounded per-subscriber queue (never drops). A positive value bounds
# the queue; overflow is counted as dropped events (the Perturbation Test
# asserts zero drops).
EVENT_QUEUE_MAXSIZE: int = 0

# --- Vision: DVS (Dynamic Vision Sensor) emulation (drhm/sensory/vision.py) --
# A real DVS emits per-pixel ON/OFF spikes on log-intensity change. With a
# commodity webcam we emulate that behaviour; this is a *sensor adapter*, not a
# cv2.absdiff attention trick in the cognitive path (CLAUDE.md §6.1).
DVS_RESOLUTION: tuple[int, int] = (32, 32)  # (height, width) the encoder works at
DVS_CONTRAST_THRESHOLD: float = 0.20  # |Δ log-intensity| to emit a spike
CAMERA_INDEX: int = 0
CAMERA_FPS: float = 30.0  # capture cadence of the executor-thread adapter only

# --- Audition: silicon-cochlea emulation (drhm/sensory/audition.py) ----------
SAMPLE_RATE: int = 16_000  # input audio sample rate (Hz)
COCHLEA_CHANNELS: int = 16  # log-spaced bandpass channels
COCHLEA_FMIN: float = 80.0  # Hz, lowest band edge
COCHLEA_FMAX: float = 8_000.0  # Hz, highest band edge (<= SAMPLE_RATE / 2)
COCHLEA_BLOCK: int = 512  # samples per analysis block
# A channel spikes when its share of total block energy exceeds this fraction.
COCHLEA_ENERGY_FRACTION: float = 0.10
COCHLEA_SILENCE_FLOOR: float = 1e-8  # below this total energy the block is silence

# --- Touch: capacitive spiking tactile array (drhm/sensory/touch.py) ---------
TOUCH_GRID: tuple[int, int] = (8, 8)  # (rows, cols) of tactile cells
TOUCH_PRESSURE_THRESHOLD: float = 0.05  # normalized pressure to register contact
TOUCH_MAX_SPIKES: int = 8  # rate-coding cap per contact sample

# --- Vocalization: spike-driven articulatory synth (drhm/sensory/vocalization.py)
VOCAL_SAMPLE_RATE: int = 16_000
VOCAL_BURST_MS: float = 20.0  # duration of the tone burst excited by one motor spike
# Maps a motor channel index -> carrier frequency (Hz) for babbling.
VOCAL_BASE_FREQ: float = 120.0
VOCAL_FREQ_STEP: float = 80.0

# --- Digital embodiment: screen/audio/peripheral capture (drhm/sensory/digital/)
SCREEN_MONITOR: int = 1  # mss monitor index (1 == primary)
SCREEN_CAPTURE_FPS: float = 20.0  # executor capture cadence of the mss fallback
# Optional static region-of-interest override in *screen* pixels (x, y, w, h).
# None == follow the focused window / whole screen (the faithful default).
SCREEN_STATIC_ROI: tuple[int, int, int, int] | None = None
LOOPBACK_INCLUDE_DEFAULT_SPEAKER: bool = True  # capture the default output's monitor

# --- Attention front-end: adverting + salience grading (drhm/attention/) -----
# Habituation EMA weight on history (per occurrence of a region). Higher ==
# slower habituation. ~0.6 suppresses a repeated stimulus after ~4 occurrences.
ATTENTION_EMA_DECAY: float = 0.6
# Cuts on normalized predictive novelty in [0, 1] -> ArammanaGrade.
# novelty < c0           -> ATI_PARITTA (moghavāra: dropped)
# c0 <= novelty < c1     -> PARITTA      (stops at moment 8)
# c1 <= novelty < c2     -> MAHANTA      (full process)
# novelty >= c2          -> ATI_MAHANTA  (full + registration)
ATTENTION_GRADE_CUTS: tuple[float, float, float] = (0.15, 0.40, 0.75)
ATTENTION_REGION_GRID: tuple[int, int] = (8, 8)  # coarse grid over the visual field

# --- Proprioception: mouse/keyboard as a sense (drhm/sensory/digital/peripherals.py)
PROPRIOCEPTION_CHANNELS: int = 256  # hash space for keys/buttons
# Privacy: log a stable per-key hash (a channel id), never the literal character.
PROPRIOCEPTION_HASH_KEYS: bool = True

# --- Session recording (drhm/recording/) -------------------------------------
RECORDING_DIR: str = "~/.drhm/sessions"  # expanduser at use; stays local
RECORDING_FORMAT: str = "jsonl"

# --- Citta-vithi & ManoDvaraVithi (drhm/snn/, drhm/citta/) -------------------
# Javana always repeats exactly 7 times in normal waking cognition (Abhidhammattha
# Sangaha §4). In jhana this extends; that exception is handled in drhm/snn/citta_vithi.py.
JAVANA_COUNT: int = 7
# Minimum mind-door (manodvara) chains that fire after every sense-door vithi.
# The Abhidhamma describes at least 3 cascading processes (whole → form → name).
MANO_DVARA_CASCADE_MIN: int = 3
# Cetana momentum decays by this factor after each manodvaravithi chain. Cascade
# continues while momentum > CETANA_MOMENTUM_THRESHOLD.
CETANA_MOMENTUM_DECAY: float = 0.7
CETANA_MOMENTUM_THRESHOLD: float = 0.1

# --- Vedanā thresholds (drhm/citta/vedana.py) ---------------------------------
# A continuous valence scalar from Conceptual Space is categorised into the three
# vedanā tones. Values are normalised to [-1, +1]; pleasant/painful cut at ±0.3.
VEDANA_PLEASANT_THRESHOLD: float = 0.3   # ≥ this → somanassa / sukha
VEDANA_PAINFUL_THRESHOLD: float = -0.3  # ≤ this → domanassa / dukkha; between → upekkha

# --- LIF neuron parameters (drhm/snn/neurons.py) ------------------------------
# Leaky Integrate-and-Fire in discrete time (Euler, dt = 1 ms).
# Voltage: mV. Current: nA. Resistance: MΩ. R_m * I_nA = mV (dimensionally consistent).
LIF_TAU_M: float = 20.0        # membrane time constant (ms)
LIF_V_REST: float = -65.0      # resting membrane potential (mV)
LIF_V_THRESHOLD: float = -55.0 # spike threshold (mV; 10 mV gap → fires in ~6 ms at full drive)
LIF_V_RESET: float = -65.0     # reset potential after a spike (mV)
LIF_R_MEMBRANE: float = 10.0   # membrane resistance (MΩ)
LIF_T_REFRACTORY: float = 2.0  # absolute refractory period (ms)
LIF_DT: float = 1.0            # simulation time step (ms)

# --- Active Inference / EFE (drhm/inference/) ---------------------------------
# The generative model is an EMA over observed X_t hypervectors.
# Surprise = 1 − cosine(X_t, μ) ∈ [0, 2]; 1.0 when no prior exists.
GENERATIVE_MODEL_ETA: float = 0.1    # EMA step size: how fast beliefs update
# MAHANTA-grade stimuli whose surprise falls below this threshold get an
# early-exit at moment 8 (energy proportionality — familiar → no javana).
# ATI_MAHANTA objects are exempt: they always proceed to registration.
EFE_SURPRISE_GATE: float = 0.2
# Scaling of epistemic (information-gain) value in EFE.  Higher → the agent
# explores more; lower → it follows the pragmatic (valence) term more closely.
EFE_EPISTEMIC_WEIGHT: float = 1.0

# --- Semantics: VSA hypervectors + Conceptors (drhm/semantics/) ---------------
# All concepts are encoded as bipolar {-1, +1} hypervectors of dimension VSA_D.
# Dimensionality is 10 000 by default; smaller values are fine in tests.
VSA_D: int = 10_000
# Default aperture α for conceptors.  Small α → narrow (strict); large α → broad (relaxed).
# α = 10 gives moderate coverage; tune per domain.
CONCEPTOR_ALPHA: float = 10.0

# --- Bhavanga attractor (drhm/snn/bhavanga.py) --------------------------------
# LIF population modelling the resting life-continuum. Sensory SpikeEvents inject
# current; when the population firing rate in the calana window crosses the
# threshold, bhavanga-upaccheda (moment 3) fires and the citta-vithi begins.
BHAVANGA_N_NEURONS: int = 256       # size of the resting attractor population
# Per-neuron drive: I_neuron = (sum_payload × GAIN) / N   [nA]
BHAVANGA_INPUT_GAIN: float = 20.0
BHAVANGA_NOISE_STD: float = 0.5    # background noise std per neuron (nA)
# Fraction of neurons that must fire in the calana window to trigger arrest.
BHAVANGA_PERTURB_THRESHOLD: float = 0.15
# If no arrest within this many steps of first input, vibration times out and
# bhavanga returns to FLOWING (insufficient perturbation decays away).
BHAVANGA_CALANA_WINDOW: int = 10
