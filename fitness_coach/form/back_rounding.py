"""
Detects back rounding (loss of neutral spine) using two combined signals:
1. The hip-shoulder-neck angle — should stay open (close to straight) when
   the spine is neutral.
2. Head position relative to hips — if the head drops well in front of/below
   where it should be, the chest has likely collapsed too.
"""
from fitness_coach.geometry.angles import AngleCalculator


class BackRoundingDetector:
    SPINE_ANGLE_THRESHOLD = 150  # degrees; below this = rounding
    HEAD_DROP_THRESHOLD = 0.08   # normalized y-distance the nose can dip below the hip-shoulder midline

    def __init__(self, side: str = "LEFT"):
        self.side = side.upper()

    def check(self, landmark_result, current_state: str = None) -> bool:
        """
        Returns True (raw, unsmoothed) if back rounding is detected this frame.
        Unlike knee valgus, this is checked in every state — a rounded back
        matters whether standing or squatting.
        """
        hip = landmark_result.get_xy(f"{self.side}_HIP")
        shoulder = landmark_result.get_xy(f"{self.side}_SHOULDER")
        # We treat the shoulder itself as a stand-in "neck base" reference point,
        # and use NOSE as the head position proxy per the project spec.
        nose = landmark_result.get_xy("NOSE")

        # Signal 1: hip-shoulder-nose angle. A neutral spine keeps this fairly open.
        spine_angle = AngleCalculator.calculate_angle(hip, shoulder, nose)
        angle_signal = spine_angle < self.SPINE_ANGLE_THRESHOLD

        # Signal 2: is the head drooping notably below the shoulder line?
        # (y increases downward in image coordinates, so a larger nose.y than
        # shoulder.y by more than the threshold means the head has dropped.)
        head_drop_signal = nose[1] > (shoulder[1] + self.HEAD_DROP_THRESHOLD)

        # Combine both signals — require both to reduce false positives from
        # someone simply looking down briefly without actually rounding their back.
        return angle_signal and head_drop_signal