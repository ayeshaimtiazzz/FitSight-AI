"""
OpenCV rendering: draws the MediaPipe skeleton, rep count, state text,
and a colored form score bar onto the frame.
"""
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

COLOR_GOOD = (0, 200, 0)      # green (BGR)
COLOR_WARNING = (0, 200, 255) # yellow
COLOR_ERROR = (0, 0, 255)     # red
COLOR_TEXT_BG = (20, 20, 20)


def draw_skeleton(frame: np.ndarray, raw_results) -> np.ndarray:
    if raw_results and raw_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            raw_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
        )
    return frame


def draw_hud(frame: np.ndarray, exercise_name: str, counter_output: dict, good_reps: int, total_reps: int) -> np.ndarray:
    """Draws rep count, current state, and angle reading in the top-left corner."""
    h, w = frame.shape[:2]

    # Semi-transparent HUD background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 130), COLOR_TEXT_BG, -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Exercise: {exercise_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {counter_output.get('rep_count', 0)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    state = counter_output.get("state", "UNKNOWN")
    state_color = COLOR_GOOD if state == "UP" else COLOR_WARNING if state == "DOWN" else (150, 150, 150)
    cv2.putText(frame, f"State: {state}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    angle = counter_output.get("angle")
    angle_text = f"Angle: {angle}" if angle is not None else "Angle: --"
    cv2.putText(frame, angle_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Form score bar (top-right)
    form_pct = (good_reps / total_reps * 100) if total_reps > 0 else 100
    bar_color = COLOR_GOOD if form_pct >= 80 else COLOR_WARNING if form_pct >= 50 else COLOR_ERROR
    bar_x, bar_y, bar_w, bar_h = w - 220, 10, 200, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    filled_w = int(bar_w * form_pct / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"Form: {form_pct:.0f}%", (bar_x + 5, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_status_box(frame: np.ndarray, ok: bool) -> np.ndarray:
    """Draws a colored border around the frame: green if form is fine, red if a form error is flagged."""
    color = COLOR_GOOD if ok else COLOR_ERROR
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    return frame