"""
Smooths boolean form-error signals across frames so a single noisy frame
doesn't trigger a false alert. Uses a majority vote over a rolling window.
"""
from collections import deque


class FormErrorBuffer:
    """
    Tracks whether a specific form error (e.g. knee valgus) has been
    persistently detected over the last `window_size` frames.

    An error is only "confirmed" once at least `confirm_ratio` of the
    frames in the window flagged it as true. This mirrors the angle
    smoothing buffer from Phase 1, but for booleans instead of floats.
    """

    def __init__(self, window_size: int = 10, confirm_ratio: float = 0.7):
        self.window_size = window_size
        self.confirm_ratio = confirm_ratio
        self.buffer = deque(maxlen=window_size)

    def update(self, raw_flag: bool) -> bool:
        """
        Call once per frame with the raw (unsmoothed) detector result.
        Returns True only if the error has been sustained long enough
        to count as a confirmed, real error rather than noise.
        """
        self.buffer.append(raw_flag)

        # Require the window to be reasonably full before confirming anything —
        # otherwise a single bad frame at startup could look like 100% of a tiny buffer.
        if len(self.buffer) < max(3, self.window_size // 2):
            return False

        true_ratio = sum(self.buffer) / len(self.buffer)
        return true_ratio >= self.confirm_ratio

    def reset(self):
        self.buffer.clear()