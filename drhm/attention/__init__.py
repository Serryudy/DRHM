"""The attention front-end — adverting + salience grading (CLAUDE.md §8.5).

This package realizes the front of the citta-vīthi as established by research
into the Abhidhamma account of attention:

  * ``SalienceGate`` (salience.py) implements the four ``ārammaṇa`` grades via
    per-region predictive novelty (the seed of the moment-8 free-energy gate).
    Repetitive / predictable stimuli decay to ATI_PARITTA and are dropped
    (``moghavāra`` — "the object is not known at all").
  * ``FocusRouter`` (focus.py) implements single-object spatial adverting
    (``pañcadvārāvajjana``): only one source (the focused window) is processed,
    honoring "one cognitive process, one object, no parallel consciousness."
  * ``AttentionFrontend`` composes the two: focus -> grade -> drop the futile.
"""

from drhm.attention.focus import FocusRouter
from drhm.attention.frontend import AttentionFrontend
from drhm.attention.salience import SalienceGate, SalienceVerdict

__all__ = ["SalienceGate", "SalienceVerdict", "FocusRouter", "AttentionFrontend"]
