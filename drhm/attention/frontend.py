"""The adverting front-end: focus -> grade -> drop the futile (CLAUDE.md §8.5).

``AttentionFrontend`` composes the two gates established by the Abhidhamma
research into the front of the citta-vīthi:

  1. ``FocusRouter`` — single-object spatial adverting (``pañcadvārāvajjana``):
     keep only spikes from the attended region.
  2. ``SalienceGate`` — per-region predictive-novelty grading (the four
     ``ārammaṇa`` grades). Each surviving spike is stamped with its grade;
     ATI_PARITTA (``moghavāra``) spikes are dropped — "not known at all."

A "region" is a cell of a coarse grid over the visual field for vision, and the
(modality, channel) pair for non-spatial senses. The frontend processes one
batch of spikes (one cognitive cycle) at a time and returns the graded
survivors, ready to publish onto the bus.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from drhm import config
from drhm.attention.focus import FocusRouter
from drhm.attention.salience import SalienceGate
from drhm.sensory.events import ArammanaGrade, Modality, SpikeEvent


class AttentionFrontend:
    """Compose spatial adverting and salience grading over a spike batch."""

    def __init__(
        self,
        resolution: tuple[int, int] | None = None,
        region_grid: tuple[int, int] | None = None,
        focus: FocusRouter | None = None,
        salience: SalienceGate | None = None,
    ) -> None:
        self.resolution = resolution or config.DVS_RESOLUTION
        self.region_grid = region_grid or config.ATTENTION_REGION_GRID
        self.focus = focus or FocusRouter(self.resolution)
        self.salience = salience or SalienceGate()

    def _vision_region(self, channel: int) -> tuple[str, int, int]:
        """Map a vision pixel channel to its coarse region key (row, col)."""
        gh, gw = self.resolution
        rg_h, rg_w = self.region_grid
        row, col = divmod(channel, gw)
        return ("vis", row * rg_h // gh, col * rg_w // gw)

    def _region_key(self, event: SpikeEvent) -> object:
        if event.modality == Modality.VISION:
            return self._vision_region(event.channel)
        return (event.modality, event.channel)

    def process(self, events: list[SpikeEvent]) -> list[SpikeEvent]:
        """Return the graded, non-futile spikes for one cognitive cycle."""
        if not events:
            return []

        focused = self.focus.route(events)
        if not focused:
            return []

        # Build one activity feature vector per region from this batch.
        buckets: dict[object, list[SpikeEvent]] = defaultdict(list)
        for event in focused:
            buckets[self._region_key(event)].append(event)

        survivors: list[SpikeEvent] = []
        for region_key, region_events in buckets.items():
            feature = self._feature(region_key, region_events)
            verdict = self.salience.assess(region_key, feature)
            if verdict is None or verdict.is_futile:
                continue  # moghavāra: object not processed
            grade = verdict.grade
            survivors.extend(_with_grade(e, grade) for e in region_events)
        return survivors

    def _feature(self, region_key: object, region_events: list[SpikeEvent]):
        """Activity pattern for a region: per-local-channel signed magnitude."""
        if isinstance(region_key, tuple) and region_key and region_key[0] == "vis":
            # Pattern over the pixels of this coarse region within the frame grid.
            gh, gw = self.resolution
            rg_h, rg_w = self.region_grid
            cell_h = max(1, gh // rg_h)
            cell_w = max(1, gw // rg_w)
            vec = np.zeros(cell_h * cell_w, dtype=np.float64)
            _, rr, cc = region_key
            for e in region_events:
                row, col = divmod(e.channel, gw)
                lr, lc = row - rr * cell_h, col - cc * cell_w
                if 0 <= lr < cell_h and 0 <= lc < cell_w:
                    vec[lr * cell_w + lc] += e.payload
            return vec
        # Non-spatial sense: a one-element feature carrying summed magnitude.
        return np.array([sum(abs(e.payload) for e in region_events)], dtype=np.float64)


def _with_grade(event: SpikeEvent, grade: ArammanaGrade) -> SpikeEvent:
    if event.grade == grade:
        return event
    return SpikeEvent(
        event.timestamp,
        event.modality,
        event.channel,
        event.payload,
        event.source,
        grade,
    )
