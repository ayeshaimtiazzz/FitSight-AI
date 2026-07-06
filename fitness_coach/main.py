"""
Entrypoint: runs the webcam loop, wires together PoseLandmarker,
the selected exercise counter, form-error detectors, and the OpenCV renderer.

Usage:
    poetry run python -m fitness_coach.main --exercise squat
    poetry run python -m fitness_coach.main --exercise pushup
    poetry run python -m fitness_coach.main --exercise curl
"""
import argparse

import cv2

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

# Which form checks apply to which exercise. Curl has none defined yet in the spec.
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


def main():
    parser = argparse.ArgumentParser(description="FitSight AI - Phase 2 Form Error Detection")
    parser.add_argument("--exercise", choices=COUNTER_MAP.keys(), default="squat")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    counter = COUNTER_MAP[args.exercise]()
    form_checks = FORM_CHECKS_MAP[args.exercise]
    error_buffers = {name: FormErrorBuffer(window_size=10, confirm_ratio=0.7) for name in form_checks}

    good_reps = 0
    total_reps = 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try a different --camera value.")

    print(f"Starting {args.exercise} counter with form checks: {list(form_checks.keys())}. Press 'q' to quit.")

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

                # Run every form check, smooth each through its own buffer
                raw_errors = {}
                confirmed_errors = {}
                for name, detector in form_checks.items():
                    raw = detector.check(landmark_result, current_state)
                    confirmed = error_buffers[name].update(raw)
                    raw_errors[name] = raw
                    confirmed_errors[name] = confirmed

                any_confirmed = any(confirmed_errors.values())
                any_raw = any(raw_errors.values())

                # Rep counting: a rep only counts as "good form" if no form error
                # was confirmed during it. We treat the pushup counter's own
                # internal alignment check as an additional signal (already
                # baked into its output.form_error).
                counter_flagged_error = output.get("form_error", False)

                if output.get("rep_completed"):
                    total_reps += 1
                    if not any_confirmed and not counter_flagged_error:
                        good_reps += 1

                severity = "error" if (any_confirmed or counter_flagged_error) else "warning" if any_raw else "good"

                frame = draw_status_box(frame, severity=severity)
                frame = draw_hud(frame, args.exercise, output, good_reps, max(total_reps, 1))
                if form_checks:
                    frame = draw_form_alerts(frame, confirmed_errors, raw_errors)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("FitSight AI - Phase 2", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total reps: {counter.rep_count}, Good form reps: {good_reps}")


if __name__ == "__main__":
    main()