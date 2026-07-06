import numpy as np

from fitness_coach.coaching.frame_buffer import FrameBuffer


def _fake_frame(color_value: int) -> np.ndarray:
    return np.full((10, 10, 3), color_value, dtype=np.uint8)


def test_frame_buffer_returns_none_when_empty():
    buf = FrameBuffer(maxlen=5)
    assert buf.get_peak_frame(mode="min") is None


def test_frame_buffer_finds_min_angle_frame():
    buf = FrameBuffer(maxlen=5)
    buf.add(_fake_frame(10), angle=160)
    buf.add(_fake_frame(20), angle=90)   # this should be the "peak" (deepest squat)
    buf.add(_fake_frame(30), angle=170)

    peak = buf.get_peak_frame(mode="min")
    assert peak[0, 0, 0] == 20  # confirms we got the frame tagged with angle=90


def test_frame_buffer_respects_maxlen():
    buf = FrameBuffer(maxlen=3)
    for i in range(5):
        buf.add(_fake_frame(i), angle=float(i))
    assert len(buf.buffer) == 3


def test_frame_buffer_clear():
    buf = FrameBuffer(maxlen=5)
    buf.add(_fake_frame(1), angle=100)
    buf.clear()
    assert buf.get_peak_frame(mode="min") is None