"""
Entrypoint: runs the webcam loop, wires together PoseLandmarker,
the selected exercise counter, and the OpenCV renderer.

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
from fitness_coach.pipeline.renderer import draw_hud, draw_skeleton, draw_status_box
from fitness_coach.pose.landmarker import PoseLandmarker

COUNTER_MAP = {
    "squat": SquatCounter,
    "pushup": PushupCounter,
    "curl": CurlCounter,
}


def main():
    parser = argparse.ArgumentParser(description="FitSight AI - Phase 1 Pose Pipeline")
    parser.add_argument("--exercise", choices=COUNTER_MAP.keys(), default="squat")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    counter = COUNTER_MAP[args.exercise]()
    good_reps = 0
    total_reps = 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try a different --camera value.")

    print(f"Starting {args.exercise} counter. Press 'q' to quit.")

    with PoseLandmarker(model_complexity=1, min_detection_confidence=0.6) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural selfie-view interaction
            landmark_result = landmarker.process(frame)

            frame = draw_skeleton(frame, landmark_result.raw_results)

            if landmark_result.detected:
                output = counter.update(landmark_result)

                if output.get("rep_completed"):
                    total_reps += 1
                    if not output.get("form_error", False):
                        good_reps += 1

                form_ok = not output.get("form_error", False)
                frame = draw_status_box(frame, ok=form_ok)
                frame = draw_hud(frame, args.exercise, output, good_reps, max(total_reps, 1))
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("FitSight AI - Phase 1", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total reps: {counter.rep_count}, Good form reps: {good_reps}")


if __name__ == "__main__":
    main()