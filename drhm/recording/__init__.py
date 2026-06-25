"""Session recording — capture today's usage for tomorrow's brain (CLAUDE.md §8.5).

The learning substrate (SNN, memory, consolidation) is milestones M2–M6 and does
not exist yet. So while you work, the ``SessionRecorder`` logs the timestamped,
graded, multimodal spike stream to a local file. When M2–M6 land, those sessions
replay into the network for consolidation — nothing is lost before the brain
exists.

Privacy: logs stay local (no network), recording is opt-in per session, can be
paused live, and keystrokes are already hashed upstream (see peripherals).
"""

from drhm.recording.session import SessionRecorder, replay

__all__ = ["SessionRecorder", "replay"]
