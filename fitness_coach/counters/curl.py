from fitness_coach.counters.base import BaseRepCounter
from fitness_coach.geometry.angles import AngleCalculator


class CurlCounter(BaseRepCounter):
    """
    Tracks wrist-elbow-shoulder angle for a single arm.
    UP threshold: 160 (arm extended)
    DOWN threshold: 40 (fully curled)

    Left and right arms are independent instances — run one CurlCounter
    per side and sum their rep_count for a combined total.
    """

    def __init__(self, side: str = "LEFT"):
        super().__init__(up_threshold=160, down_threshold=40, buffer_size=5)
        self.side = side.upper()

    def _extract_angle(self, landmark_result) -> float:
        shoulder = landmark_result.get_xy(f"{self.side}_SHOULDER")
        elbow = landmark_result.get_xy(f"{self.side}_ELBOW")
        wrist = landmark_result.get_xy(f"{self.side}_WRIST")
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)