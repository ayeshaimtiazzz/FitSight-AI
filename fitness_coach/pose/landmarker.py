"""
Thin wrapper around MediaPipe Pose that converts raw MediaPipe output into
a clean, typed dataclass keyed by readable landmark names instead of
numeric indices.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Only the landmarks we actually use across all exercises. Full list has 33;
# we keep all of them but this shows the ones that matter for reference.
LandmarkPoint = Tuple[float, float, float, float]  # x, y, z, visibility


@dataclass
class LandmarkResult:
    landmarks: Dict[str, LandmarkPoint] = field(default_factory=dict)
    detected: bool = False
    raw_results: Optional[object] = None  # kept for drawing utilities

    def get_xy(self, name: str) -> Tuple[float, float]:
        """Convenience accessor returning just (x, y) for angle calculations."""
        lm = self.landmarks.get(name)
        if lm is None:
            raise KeyError(f"Landmark '{name}' not found in this frame's result.")
        return (lm[0], lm[1])

    def visibility(self, name: str) -> float:
        lm = self.landmarks.get(name)
        return lm[3] if lm else 0.0


class PoseLandmarker:
    """Wraps mp.solutions.pose.Pose and produces LandmarkResult per frame."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.6):
        self._pose = mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> LandmarkResult:
        """
        Takes an OpenCV BGR frame, runs MediaPipe Pose, and returns a
        LandmarkResult with all 33 landmarks normalized to frame dimensions
        (x, y in [0,1], z relative depth, visibility in [0,1]).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return LandmarkResult(detected=False, raw_results=results)

        landmarks: Dict[str, LandmarkPoint] = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = mp_pose.PoseLandmark(idx).name  # e.g. "LEFT_KNEE"
            landmarks[name] = (lm.x, lm.y, lm.z, lm.visibility)

        return LandmarkResult(landmarks=landmarks, detected=True, raw_results=results)

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()