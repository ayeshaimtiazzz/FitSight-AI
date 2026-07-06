"""
Shared state machine + smoothing logic used by every exercise counter.
Each specific counter (squat, pushup, curl) subclasses this and only
needs to implement `_extract_angle()`.
"""
from collections import deque
from enum import Enum


class RepState(Enum):
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class BaseRepCounter:
    """
    Generic UP/DOWN state machine with angle smoothing.

    A rep is counted on the transition DOWN -> UP.
    Smoothing uses a rolling average of the last `buffer_size` angle readings
    to avoid landmark jitter causing false state flips.
    """

    def __init__(self, up_threshold: float, down_threshold: float, buffer_size: int = 5):
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.state = RepState.UNKNOWN
        self.rep_count = 0
        self.last_smoothed_angle = None

    def _smooth(self, raw_angle: float) -> float:
        self.buffer.append(raw_angle)
        return sum(self.buffer) / len(self.buffer)

    def _extract_angle(self, landmark_result) -> float:
        """Subclasses override this to pull the relevant angle for their exercise."""
        raise NotImplementedError

    def update(self, landmark_result) -> dict:
        """
        Call once per frame. Returns a dict with the current state, angle,
        and whether a rep was just completed on this frame.
        """
        if not landmark_result.detected:
            return {
                "state": self.state.value,
                "angle": self.last_smoothed_angle,
                "rep_completed": False,
                "rep_count": self.rep_count,
            }

        raw_angle = self._extract_angle(landmark_result)
        smoothed = self._smooth(raw_angle)
        self.last_smoothed_angle = smoothed

        rep_completed = False
        previous_state = self.state

        if smoothed > self.up_threshold:
            new_state = RepState.UP
        elif smoothed < self.down_threshold:
            new_state = RepState.DOWN
        else:
            new_state = self.state  # stay in current state within the dead zone

        # Rep counts on the DOWN -> UP transition only
        if previous_state == RepState.DOWN and new_state == RepState.UP:
            self.rep_count += 1
            rep_completed = True

        self.state = new_state

        return {
            "state": self.state.value,
            "angle": round(smoothed, 1),
            "rep_completed": rep_completed,
            "rep_count": self.rep_count,
        }

    def reset(self):
        self.buffer.clear()
        self.state = RepState.UNKNOWN
        self.rep_count = 0
        self.last_smoothed_angle = None