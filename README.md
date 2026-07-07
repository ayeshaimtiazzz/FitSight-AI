# FitSight AI 🏋️‍♀️

A real-time AI fitness coaching application that uses computer vision to track exercise form, count reps, detect form errors, and deliver personalized AI coaching feedback — all running on CPU with no GPU required.

## What it does

FitSight AI watches you exercise through a webcam and provides live feedback across four layers:

1. **Pose tracking & rep counting** — MediaPipe Pose extracts 33 body landmarks per frame in real time. A geometry engine computes joint angles, and a state machine counts reps for squats, push-ups, and bicep curls.
2. **Form error detection** — Rule-based detectors flag knee valgus (knees caving inward), back rounding, and elbow flare, with a persistence buffer that only raises an alert once an error is sustained for multiple frames (avoiding false positives from single noisy frames).
3. **AI coaching feedback** — Every 10 reps, the system captures the peak-contraction frame (e.g. lowest point of a squat) and sends it to Google's Gemini Vision API, which returns a short, specific coaching note. Feedback is read aloud via offline text-to-speech.
4. **Session storage & analytics** — Every workout is logged to a local SQLite database. A separate Streamlit dashboard visualizes trends across sessions: reps over time, form score trends, and joint angle distributions.

## Tech stack

| Component | Technology |
|---|---|
| Pose estimation | MediaPipe Pose (BlazePose, CPU-optimized) |
| Video pipeline | OpenCV |
| AI coaching | Google Gemini Vision API (`google-genai` SDK) |
| Text-to-speech | pyttsx3 (offline) |
| Storage | SQLite + SQLAlchemy |
| Analytics dashboard | Streamlit + Plotly |
| PDF reports | ReportLab |
| Dependency management | Poetry |

## Project structure

fitness_coach/
├── pose/               # MediaPipe wrapper — extracts landmarks per frame
├── geometry/           # Pure angle-calculation math (2D and 3D), no framework dependencies
├── counters/           # Rep-counting state machines: squat, push-up, curl
├── form/                # Form error detectors: knee valgus, back rounding, elbow flare
├── coaching/            # Gemini Vision API integration, frame buffer, TTS
├── storage/             # SQLAlchemy models + database engine
├── analytics/           # Plotly chart builders
├── reports/              # PDF report generation
├── dashboard.py          # Streamlit analytics app
└── main.py               # Live webcam entrypoint
tests/                    # pytest unit tests (angle math, form detectors, frame buffer)
data/
└── fitsight.db            # SQLite database (created automatically)

## Setup

**Requirements:** Python 3.11 or 3.12 (MediaPipe does not yet support 3.13).

```bash
# Clone the repo
git clone https://github.com/ayeshaimtiazzz/FitSight-AI.git
cd FitSight-AI

# Install dependencies via Poetry
poetry install
```

### Environment variables

Create a `.env` file in the project root with your Gemini API key (get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

GEMINI_API_KEY=your_key_here

## Usage

### Run a live workout session

```bash
poetry run python -m fitness_coach.main --exercise squat
poetry run python -m fitness_coach.main --exercise pushup --side RIGHT
poetry run python -m fitness_coach.main --exercise curl
```

**Flags:**
- `--exercise` — `squat`, `pushup`, or `curl`
- `--side` — `LEFT` or `RIGHT` (which arm/leg to track; useful for push-ups/curls depending on your camera angle)
- `--camera` — webcam index (default `0`)
- `--no-voice` — disable spoken TTS feedback

Press `q` to end the session. Your reps, form score, and coaching notes are automatically saved to the local database.

### View analytics

```bash
poetry run streamlit run fitness_coach/dashboard.py
```

Opens a browser dashboard showing session history, rep trends, form score trends, and joint angle distributions across all logged workouts.

### Run tests

```bash
poetry run pytest tests/ -v
```

## Key engineering decisions

- **3D angle calculations for curls** — Early testing showed that flat 2D angle math falsely counted reps when the arm moved toward/away from the camera (depth) without real elbow flexion. Switching to MediaPipe's native 3D landmarks (x, y, z) resolved this by correctly distinguishing depth movement from actual joint bending.
- **Persistence-based form error buffering** — Form checks (knee valgus, back rounding, elbow flare) use a rolling majority-vote buffer rather than raising alerts on single-frame detections, which prevents jittery false positives from momentary landmark noise.
- **Background-threaded VLM calls** — Gemini API calls run on a `ThreadPoolExecutor` so the live video feed never freezes waiting on a network response.
- **Streamlit kept separate from the live video loop** — Rather than fighting Streamlit's full-script-rerender execution model to embed live webcam video, the live workout tool (OpenCV `cv2.imshow`) and the analytics dashboard (Streamlit) are intentionally separate apps that share the same SQLite database.




