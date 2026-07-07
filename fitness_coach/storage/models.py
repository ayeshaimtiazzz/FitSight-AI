"""
SQLAlchemy ORM models for session storage.

sessions          — one row per workout session
rep_events        — one row per completed rep within a session
coaching_feedback — one row per Gemini coaching note generated during a session
"""
from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class WorkoutSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    exercise = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float, default=0.0)
    total_reps = Column(Integer, default=0)
    good_form_reps = Column(Integer, default=0)
    avg_form_score = Column(Float, default=100.0)

    rep_events = relationship("RepEvent", back_populates="session", cascade="all, delete-orphan")
    coaching_feedback = relationship("CoachingFeedback", back_populates="session", cascade="all, delete-orphan")


class RepEvent(Base):
    __tablename__ = "rep_events"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    joint_angle = Column(Float)
    form_errors_json = Column(Text)  # JSON-encoded list of error names, e.g. '["knee_valgus"]'
    is_good_form = Column(Integer, default=1)  # 1/0 instead of Boolean for simplest SQLite compatibility

    session = relationship("WorkoutSession", back_populates="rep_events")


class CoachingFeedback(Base):
    __tablename__ = "coaching_feedback"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    rep_number = Column(Integer)
    feedback_text = Column(Text)
    frame_path = Column(String, nullable=True)  # optional saved keyframe thumbnail path

    session = relationship("WorkoutSession", back_populates="coaching_feedback")