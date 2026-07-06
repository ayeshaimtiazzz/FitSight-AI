from fitness_coach.counters.base import BaseRepCounter
from fitness_coach.geometry.angles import AngleCalculator


class SquatCounter(BaseRepCounter):
    """
    Tracks the hip-knee-ankle angle.
    UP threshold: 160 degrees (standing)
    DOWN threshold: 90 degrees (at/below parallel)
    """

    def __init__(self, side: str = "LEFT"):
        super().__init__(up_threshold=160, down_threshold=90, buffer_size=5)
        self.side = side.upper()

    def _extract_angle(self, landmark_result) -> float:
        hip = landmark_result.get_xy(f"{self.side}_HIP")
        knee = landmark_result.get_xy(f"{self.side}_KNEE")
        ankle = landmark_result.get_xy(f"{self.side}_ANKLE")
        return AngleCalculator.calculate_angle(hip, knee, ankle)