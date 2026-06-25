"""Phase 1 — the asynchronous, event-driven sensory substrate (CLAUDE.md §8 M1).

Everything here is event-driven: sources push ``SpikeEvent``s onto an
``EventBus``; consumers ``await`` them. There is no polling loop and no
global-clock frame processing in any hot path (CLAUDE.md §6.1).
"""

from drhm.sensory.audition import CochleaEncoder
from drhm.sensory.events import (
    ArammanaGrade,
    EventBus,
    Modality,
    SpikeEvent,
    Subscription,
)
from drhm.sensory.touch import TouchEncoder
from drhm.sensory.vision import DvsEncoder
from drhm.sensory.vocalization import render_motor_spikes

__all__ = [
    "EventBus",
    "Modality",
    "ArammanaGrade",
    "SpikeEvent",
    "Subscription",
    "DvsEncoder",
    "CochleaEncoder",
    "TouchEncoder",
    "render_motor_spikes",
]
