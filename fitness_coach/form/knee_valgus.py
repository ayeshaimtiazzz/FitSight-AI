"""
Detects knee valgus (knees caving inward) during the down phase of a squat.

Method: project the knee's x-coordinate onto the straight line connecting
hip and ankle. If the knee deviates more than a threshold amount inward
(medially) from that line, the knee is caving.
"""
from fitness_coach.geometry.angles import AngleCalculator


class KneeValgusDetector:
    # Normalized coordinate units (0-1 range, same scale as MediaPipe landmarks).
    # 0.05 was chosen as a starting point per the project spec — tune this
    # after testing against your own squat depth and camera distance.
    DEVIATION_THRESHOLD = 0.05

    def __init__(self, side: str = "LEFT"):
        self.side = side.upper()

    def check(self, landmark_result, current_state: str) -> bool:
        """
        Returns True (raw, unsmoothed) if knee valgus is detected this frame.
        Only checks during the DOWN state — valgus during standing is meaningless.
        """
        if current_state != "DOWN":
            return False

        hip = landmark_result.get_xy(f"{self.side}_HIP")
        knee = landmark_result.get_xy(f"{self.side}_KNEE")
        ankle = landmark_result.get_xy(f"{self.side}_ANKLE")

        deviation = AngleCalculator.vertical_deviation(knee, ref_top=hip, ref_bottom=ankle)
        return deviation > self.DEVIATION_THRESHOLD