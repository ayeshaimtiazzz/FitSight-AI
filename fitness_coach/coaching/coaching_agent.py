"""
Wraps the Gemini API (new google-genai SDK) to generate short, specific
coaching feedback from a single keyframe image plus exercise context.
"""
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import cv2
import numpy as np
import io

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Make sure you have a .env file in the project "
        "root with a line like: GEMINI_API_KEY=your_key_here"
    )


class CoachingAgent:
    """
    Generates a short coaching note from a keyframe using Gemini's vision model.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=_API_KEY)
        self.model_name = model_name

    def _cv2_to_png_bytes(self, frame_bgr: np.ndarray) -> bytes:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()

    def generate_feedback(
        self,
        frame_bgr: np.ndarray,
        exercise: str,
        rep_count: int,
        recent_form_errors: list[str],
    ) -> str:
        """
        Sends the keyframe + context to Gemini and returns a short coaching note.
        Blocking call — the caller (main.py) runs this in a background thread
        so the video loop doesn't freeze.
        """
        image_bytes = self._cv2_to_png_bytes(frame_bgr)
        errors_text = ", ".join(recent_form_errors) if recent_form_errors else "none detected"

        prompt = (
            f"You are an expert personal trainer. The user just completed {rep_count} "
            f"{exercise} reps. Detected form issues in recent reps: {errors_text}. "
            "Analyse this frame and give: "
            "(1) one specific thing they did well, "
            "(2) the most important form correction needed, "
            "(3) one drill to practice this week. "
            "Be encouraging, specific, and under 60 words."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            return response.text.strip()
        except Exception as e:
            return f"(Coaching feedback unavailable this round: {e})"
        