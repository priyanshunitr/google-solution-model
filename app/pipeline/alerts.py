"""
Alert debouncer. Requires a raw signal to be True for `sustain` consecutive
frames before firing a 'start' event; after an 'end' event a cooldown
prevents start/end flapping.
"""
from __future__ import annotations

from typing import Optional


class AlertDebouncer:
    def __init__(self, sustain: int, cooldown: int) -> None:
        self.sustain = sustain
        self.cooldown = cooldown
        self.true_streak = 0
        self.active = False
        self.cooldown_left = 0
        self.start_frame: Optional[int] = None

    def step(self, raw_signal: bool, frame_idx: int) -> dict:
        event: Optional[str] = None
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            raw_signal = False

        if raw_signal:
            self.true_streak += 1
            if not self.active and self.true_streak >= self.sustain:
                self.active = True
                self.start_frame = frame_idx - self.sustain + 1
                event = "start"
        else:
            if self.active:
                event = "end"
                self.active = False
                self.cooldown_left = self.cooldown
            self.true_streak = 0

        return {"active": self.active, "event": event, "start_frame": self.start_frame}
