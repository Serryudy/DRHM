"""Digital embodiment — the computer itself as the agent's body (CLAUDE.md §8.5).

Instead of a webcam/microphone in a room, these sources pipe the machine's own
streams into the M1 encoders:

  * ``ScreenVisionSource``    — desktop screen-capture -> existing ``DvsEncoder``.
  * ``LoopbackAuditionSource``— system audio loopback -> existing ``CochleaEncoder``.
  * ``PeripheralSource``      — mouse/keyboard -> ``Modality.PROPRIOCEPTION``.

Every source reuses M1 sensing logic and stays event-driven: there is no
``cv2.absdiff`` polling loop (CLAUDE.md §6.1). Live capture imports its backend
(mss / soundcard / pynput / Xlib) lazily, so the package works headless and the
encoders/sources are unit-testable via ``.feed*()``.
"""

from drhm.sensory.digital.loopback import LoopbackAuditionSource
from drhm.sensory.digital.peripherals import PeripheralSource
from drhm.sensory.digital.screen import ScreenVisionSource

__all__ = ["ScreenVisionSource", "LoopbackAuditionSource", "PeripheralSource"]
