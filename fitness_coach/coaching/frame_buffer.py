"""
Maintains a rolling buffer of recent frames + their tracked angle, so that
when a rep completes we can retroactively find the "peak contraction" frame
(e.g. lowest knee angle in a squat) instead of just using the last frame.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class BufferedFrame:
    frame: np.ndarray   # the raw BGR OpenCV frame at this moment
    angle: float        # the tracked joint angle at this moment


class FrameBuffer:
    """
    Stores the last `maxlen` (frame, angle) pairs. After a rep completes,
    call get_peak_frame() to retrieve whichever buffered frame had the most
    extreme angle — this is the visually "deepest" moment of the rep.
    """

    def __init__(self, maxlen: int = 30):
        self.buffer: deque[BufferedFrame] = deque(maxlen=maxlen)

    def add(self, frame: np.ndarray, angle: float):
        # Store a copy — the caller's frame gets overwritten next loop iteration
        self.buffer.append(BufferedFrame(frame=frame.copy(), angle=angle))

    def get_peak_frame(self, mode: str = "min") -> np.ndarray:
        """
        mode="min": returns the frame with the smallest angle (e.g. bottom of a squat,
                    top of a curl where elbow angle is smallest).
        mode="max": returns the frame with the largest angle (rarely needed, but
                    kept for symmetry/future exercises).

        Returns None if the buffer is empty (shouldn't happen in practice since
        we only call this after at least one rep has completed).
        """
        if not self.buffer:
            return None

        if mode == "min":
            best = min(self.buffer, key=lambda bf: bf.angle)
        else:
            best = max(self.buffer, key=lambda bf: bf.angle)

        return best.frame

    def clear(self):
        self.buffer.clear()