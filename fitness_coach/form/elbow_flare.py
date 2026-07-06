"""
Detects excessive elbow flare during push-ups — elbows winging out too far
from the torso, which increases shoulder strain.

Method: measure the angle at the shoulder formed between the torso line
(shoulder-hip) and the upper arm (shoulder-elbow). A large angle here means
the elbow is flared far out to the side rather than tucked closer to the body.
"""
from fitness_coach.geometry.angles import AngleCalculator


class ElbowFlareDetector:
    # Angle at the shoulder between torso and upper arm. Above this = flared.
    FLARE_ANGLE_THRESHOLD = 65

    def __init__(self, side: str = "LEFT"):
        self.side = side.upper()

    def check(self, landmark_result, current_state: str) -> bool:
        """
        Returns True (raw, unsmoothed) if elbow flare is detected this frame.
        Most relevant during the DOWN phase of a push-up, where flare tends
        to be worst.
        """
        if current_state != "DOWN":
            return False

        shoulder = landmark_result.get_xy(f"{self.side}_SHOULDER")
        hip = landmark_result.get_xy(f"{self.side}_HIP")
        elbow = landmark_result.get_xy(f"{self.side}_ELBOW")

        flare_angle = AngleCalculator.calculate_angle(hip, shoulder, elbow)
        return flare_angle > self.FLARE_ANGLE_THRESHOLD