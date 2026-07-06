"""
OpenCV rendering: draws the MediaPipe skeleton, rep count, state text,
a colored form score bar, and now confirmed form-error labels with a
3-tier color coding (green/yellow/red) onto the frame.
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

    form_pct = (good_reps / total_reps * 100) if total_reps > 0 else 100
    bar_color = COLOR_GOOD if form_pct >= 80 else COLOR_WARNING if form_pct >= 50 else COLOR_ERROR
    bar_x, bar_y, bar_w, bar_h = w - 220, 10, 200, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    filled_w = int(bar_w * form_pct / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"Form: {form_pct:.0f}%", (bar_x + 5, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_form_alerts(frame: np.ndarray, confirmed_errors: dict, raw_errors: dict) -> np.ndarray:
    """
    Draws a label for each form check underneath the HUD, in the bottom-left corner.

    confirmed_errors: dict of {error_name: bool} — True means the FormErrorBuffer
                       has confirmed this as sustained, real error (RED).
    raw_errors: dict of {error_name: bool} — True means the raw detector fired
                 this frame but hasn't been confirmed yet (YELLOW warning).
    Neither True = GREEN (good form for that check).
    """
    h, w = frame.shape[:2]
    y_start = h - 20 - (25 * len(confirmed_errors))

    for i, (name, confirmed) in enumerate(confirmed_errors.items()):
        raw = raw_errors.get(name, False)

        if confirmed:
            color = COLOR_ERROR
            status = "ERROR"
        elif raw:
            color = COLOR_WARNING
            status = "WARNING"
        else:
            color = COLOR_GOOD
            status = "OK"

        y = y_start + i * 25
        label = name.replace("_", " ").title()
        cv2.putText(frame, f"{label}: {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


def draw_status_box(frame: np.ndarray, severity: str = "good") -> np.ndarray:
    """
    Draws a colored border around the frame.
    severity: "good" (green), "warning" (yellow), or "error" (red).
    """
    color_map = {"good": COLOR_GOOD, "warning": COLOR_WARNING, "error": COLOR_ERROR}
    color = color_map.get(severity, COLOR_GOOD)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
    return frame