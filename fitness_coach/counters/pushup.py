from fitness_coach.counters.base import BaseRepCounter
from fitness_coach.geometry.angles import AngleCalculator


class PushupCounter(BaseRepCounter):
    """
    Tracks elbow angle (shoulder-elbow-wrist) for the rep state machine,
    AND body alignment (shoulder-hip-ankle) as a rep validity gate.

    A rep only counts if elbow angle dips below 90 AND body alignment
    stays above 165 (straight body) throughout the down phase. If alignment
    drops below 165 during the rep, it's flagged as a form error and NOT counted.
    """

    ALIGNMENT_THRESHOLD = 165

    def __init__(self, side: str = "LEFT"):
        super().__init__(up_threshold=160, down_threshold=90, buffer_size=5)
        self.side = side.upper()
        self._alignment_broke_this_rep = False

    def _extract_angle(self, landmark_result) -> float:
        shoulder = landmark_result.get_xy(f"{self.side}_SHOULDER")
        elbow = landmark_result.get_xy(f"{self.side}_ELBOW")
        wrist = landmark_result.get_xy(f"{self.side}_WRIST")
        return AngleCalculator.calculate_angle(shoulder, elbow, wrist)

    def _check_alignment(self, landmark_result) -> float:
        shoulder = landmark_result.get_xy(f"{self.side}_SHOULDER")
        hip = landmark_result.get_xy(f"{self.side}_HIP")
        ankle = landmark_result.get_xy(f"{self.side}_ANKLE")
        return AngleCalculator.calculate_angle(shoulder, hip, ankle)

    def update(self, landmark_result) -> dict:
        result = super().update(landmark_result)

        if not landmark_result.detected:
            result["form_error"] = False
            return result

        alignment_angle = self._check_alignment(landmark_result)
        alignment_broken = alignment_angle < self.ALIGNMENT_THRESHOLD

        if alignment_broken:
            self._alignment_broke_this_rep = True

        # If a rep was just completed but alignment broke at some point during it,
        # undo the count — this was a form-invalid rep.
        if result["rep_completed"] and self._alignment_broke_this_rep:
            self.rep_count -= 1
            result["rep_count"] = self.rep_count
            result["rep_completed"] = False
            result["form_error"] = True
        else:
            result["form_error"] = alignment_broken

        # Reset the flag once we're back in the UP state (rep cycle complete)
        if result["state"] == "UP":
            self._alignment_broke_this_rep = False

        result["alignment_angle"] = round(alignment_angle, 1)
        return result