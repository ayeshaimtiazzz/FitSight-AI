"""
Offline text-to-speech for reading coaching feedback aloud, with a cooldown
so overlapping messages don't talk over each other.
"""
import queue
import threading
import time

import pyttsx3


class SpeechEngine:
    """
    Runs pyttsx3 on its own dedicated thread (pyttsx3 is not thread-safe to
    call from multiple threads directly), consuming messages from a queue.
    Enforces a minimum cooldown between messages so feedback doesn't overlap
    or fire too rapidly.
    """

    def __init__(self, cooldown_seconds: float = 30.0, rate: int = 165):
        self.cooldown_seconds = cooldown_seconds
        self._queue: queue.Queue[str] = queue.Queue()
        self._last_spoken_at = 0.0
        self._stop_event = threading.Event()
        self._rate = rate

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)

        while not self._stop_event.is_set():
            try:
                message = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Enforce cooldown: if not enough time has passed since the last
            # message finished, wait out the remainder before speaking again.
            elapsed = time.time() - self._last_spoken_at
            if elapsed < self.cooldown_seconds:
                time.sleep(self.cooldown_seconds - elapsed)

            engine.say(message)
            engine.runAndWait()
            self._last_spoken_at = time.time()

    def speak(self, message: str):
        """Queues a message to be spoken. Non-blocking."""
        self._queue.put(message)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)