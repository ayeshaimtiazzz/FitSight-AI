from fitness_coach.counters.base import BaseRepCounter
from fitness_coach.geometry.angles import AngleCalculator


class CurlCounter(BaseRepCounter):
    """
    Tracks wrist-elbow-shoulder angle for a single arm, using 3D landmarks
    to avoid false rep counts caused by moving the arm toward/away from the
    camera (which distorts a flat 2D angle even without real elbow flexion).

    UP threshold: 160 (arm extended)
    DOWN threshold: 40 (fully curled)
    """

    def __init__(self, side: str = "LEFT"):
        super().__init__(up_threshold=160, down_threshold=40, buffer_size=5)
        self.side = side.upper()

    def _extract_angle(self, landmark_result) -> float:
        shoulder = landmark_result.get_xyz(f"{self.side}_SHOULDER")
        elbow = landmark_result.get_xyz(f"{self.side}_ELBOW")
        wrist = landmark_result.get_xyz(f"{self.side}_WRIST")
        return AngleCalculator.calculate_angle_3d(shoulder, elbow, wrist)