"""
Entrypoint: runs the webcam loop, wires together PoseLandmarker, the selected
exercise counter, form-error detectors, VLM coaching feedback, TTS, and the
OpenCV renderer.

Usage:
    poetry run python -m fitness_coach.main --exercise squat
    poetry run python -m fitness_coach.main --exercise pushup
    poetry run python -m fitness_coach.main --exercise curl
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2

from fitness_coach.coaching.coaching_agent import CoachingAgent
from fitness_coach.coaching.frame_buffer import FrameBuffer
from fitness_coach.coaching.speech_engine import SpeechEngine
from fitness_coach.counters.curl import CurlCounter
from fitness_coach.counters.pushup import PushupCounter
from fitness_coach.counters.squat import SquatCounter
from fitness_coach.form.back_rounding import BackRoundingDetector
from fitness_coach.form.elbow_flare import ElbowFlareDetector
from fitness_coach.form.error_buffer import FormErrorBuffer
from fitness_coach.form.knee_valgus import KneeValgusDetector
from fitness_coach.pipeline.renderer import (
    draw_form_alerts,
    draw_hud,
    draw_skeleton,
    draw_status_box,
)
from fitness_coach.pose.landmarker import PoseLandmarker

COUNTER_MAP = {
    "squat": SquatCounter,
    "pushup": PushupCounter,
    "curl": CurlCounter,
}

FORM_CHECKS_MAP = {
    "squat": {
        "knee_valgus": KneeValgusDetector(),
        "back_rounding": BackRoundingDetector(),
    },
    "pushup": {
        "elbow_flare": ElbowFlareDetector(),
        "back_rounding": BackRoundingDetector(),
    },
    "curl": {},
}

COACHING_INTERVAL = 10  # generate feedback every N reps


def main():
    parser = argparse.ArgumentParser(description="FitSight AI - Phase 3 VLM Coaching")
    parser.add_argument("--exercise", choices=COUNTER_MAP.keys(), default="squat")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--no-voice", action="store_true", help="Disable TTS voice feedback")
    args = parser.parse_args()

    counter = COUNTER_MAP[args.exercise]()
    form_checks = FORM_CHECKS_MAP[args.exercise]
    error_buffers = {name: FormErrorBuffer(window_size=10, confirm_ratio=0.7) for name in form_checks}

    frame_buffer = FrameBuffer(maxlen=30)
    coaching_agent = CoachingAgent()
    speech_engine = None if args.no_voice else SpeechEngine(cooldown_seconds=30.0)
    executor = ThreadPoolExecutor(max_workers=2)

    recent_form_errors: list[str] = []
    latest_feedback_text = ""

    good_reps = 0
    total_reps = 0

    def _on_feedback_ready(future, exercise_name, rep_count):
        nonlocal latest_feedback_text
        try:
            feedback = future.result()
        except Exception as e:
            feedback = f"(Coaching error: {e})"
        latest_feedback_text = feedback
        print(f"\n[Coaching @ rep {rep_count}] {feedback}\n")
        if speech_engine:
            speech_engine.speak(feedback)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try a different --camera value.")

    print(f"Starting {args.exercise} counter with VLM coaching every {COACHING_INTERVAL} reps. Press 'q' to quit.")

    with PoseLandmarker(model_complexity=1, min_detection_confidence=0.6) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            landmark_result = landmarker.process(frame)

            frame = draw_skeleton(frame, landmark_result.raw_results)

            if landmark_result.detected:
                output = counter.update(landmark_result)
                current_state = output.get("state", "UNKNOWN")
                current_angle = output.get("angle")

                if current_angle is not None:
                    frame_buffer.add(frame, current_angle)

                raw_errors = {}
                confirmed_errors = {}
                for name, detector in form_checks.items():
                    raw = detector.check(landmark_result, current_state)
                    confirmed = error_buffers[name].update(raw)
                    raw_errors[name] = raw
                    confirmed_errors[name] = confirmed
                    if confirmed and name not in recent_form_errors:
                        recent_form_errors.append(name)

                any_confirmed = any(confirmed_errors.values())
                any_raw = any(raw_errors.values())
                counter_flagged_error = output.get("form_error", False)

                if output.get("rep_completed"):
                    total_reps += 1
                    if not any_confirmed and not counter_flagged_error:
                        good_reps += 1

                    # Every COACHING_INTERVAL reps, kick off a background VLM call
                    if counter.rep_count > 0 and counter.rep_count % COACHING_INTERVAL == 0:
                        peak_frame = frame_buffer.get_peak_frame(mode="min")
                        if peak_frame is not None:
                            future = executor.submit(
                                coaching_agent.generate_feedback,
                                peak_frame,
                                args.exercise,
                                counter.rep_count,
                                recent_form_errors[-3:],  # last 3 distinct errors seen
                            )
                            future.add_done_callback(
                                lambda f, rc=counter.rep_count: _on_feedback_ready(f, args.exercise, rc)
                            )
                            recent_form_errors.clear()

                severity = "error" if (any_confirmed or counter_flagged_error) else "warning" if any_raw else "good"

                frame = draw_status_box(frame, severity=severity)
                frame = draw_hud(frame, args.exercise, output, good_reps, max(total_reps, 1))
                if form_checks:
                    frame = draw_form_alerts(frame, confirmed_errors, raw_errors)

                # Show the latest coaching feedback as wrapped text at the bottom of the frame
                if latest_feedback_text:
                    _draw_wrapped_text(frame, latest_feedback_text)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("FitSight AI - Phase 3", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    if speech_engine:
        speech_engine.stop()
    print(f"Session ended. Total reps: {counter.rep_count}, Good form reps: {good_reps}")


def _draw_wrapped_text(frame, text, font_scale=0.5, line_height=20, max_lines=6):
    """
    Word-wraps the coaching note to fit the frame's actual pixel width
    (measured via cv2.getTextSize, not a guessed character count), and
    displays up to `max_lines` lines so the full message isn't cut off.
    If the message is too long even at max_lines, shrinks the font instead
    of truncating the text.
    """
    import textwrap
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10
    usable_width = w - (2 * padding)

    avg_char_width = cv2.getTextSize("A", font, font_scale, 1)[0][0]
    wrap_chars = max(20, usable_width // avg_char_width)
    lines = textwrap.wrap(text, width=wrap_chars)

    if len(lines) > max_lines:
        font_scale = max(0.35, font_scale * (max_lines / len(lines)))
        avg_char_width = cv2.getTextSize("A", font, font_scale, 1)[0][0]
        wrap_chars = max(20, usable_width // avg_char_width)
        lines = textwrap.wrap(text, width=wrap_chars)

    lines = lines[:max_lines]

    box_height = padding * 2 + line_height * len(lines)
    y_start = h - box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    for i, line in enumerate(lines):
        y = y_start + padding + (i + 1) * line_height - 6
        cv2.putText(frame, line, (padding, y), font, font_scale, (255, 255, 0), 1, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    main()