"""
Streamlit analytics dashboard — reads accumulated session history from SQLite
and displays it via Plotly charts. This is a separate app from the live
webcam workout tool (main.py); run it after you've logged some sessions.

Usage:
    poetry run streamlit run fitness_coach/dashboard.py
"""
import pandas as pd
import streamlit as st

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fitness_coach.analytics.charts import (
    form_score_trend_chart,
    joint_angle_distribution_chart,
    personal_records_table,
    rep_timeline_chart,
)
from fitness_coach.storage.db import get_db_session, init_db
from fitness_coach.storage.models import RepEvent, WorkoutSession

st.set_page_config(page_title="FitSight AI — Analytics", layout="wide")

init_db()


@st.cache_data(ttl=5)
def load_sessions_df():
    db = get_db_session()
    try:
        sessions = db.query(WorkoutSession).all()
        return pd.DataFrame([{
            "id": s.id,
            "exercise": s.exercise,
            "date": s.date,
            "duration_seconds": s.duration_seconds,
            "total_reps": s.total_reps,
            "good_form_reps": s.good_form_reps,
            "avg_form_score": s.avg_form_score,
        } for s in sessions])
    finally:
        db.close()


@st.cache_data(ttl=5)
def load_rep_events_df():
    db = get_db_session()
    try:
        events = db.query(RepEvent).all()
        return pd.DataFrame([{
            "session_id": e.session_id,
            "joint_angle": e.joint_angle,
            "is_good_form": e.is_good_form,
        } for e in events])
    finally:
        db.close()


st.title("🏋️ FitSight AI — Analytics Dashboard")

sessions_df = load_sessions_df()
rep_events_df = load_rep_events_df()

if sessions_df.empty:
    st.info("No sessions recorded yet. Run `poetry run python -m fitness_coach.main --exercise squat`, "
             "complete a workout, and refresh this page.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", len(sessions_df))
    col2.metric("Total Reps Logged", int(sessions_df["total_reps"].sum()))
    col3.metric("Avg Form Score", f"{sessions_df['avg_form_score'].mean():.1f}%")

    st.plotly_chart(rep_timeline_chart(sessions_df), use_container_width=True)
    st.plotly_chart(form_score_trend_chart(sessions_df), use_container_width=True)

    if not rep_events_df.empty:
        st.plotly_chart(joint_angle_distribution_chart(rep_events_df), use_container_width=True)
    else:
        st.info("No rep-level angle data yet.")

    st.subheader("Personal Records")
    st.dataframe(personal_records_table(sessions_df), use_container_width=True)

    st.subheader("Session History")
    st.dataframe(
        sessions_df.sort_values("date", ascending=False)[
            ["date", "exercise", "total_reps", "good_form_reps", "avg_form_score"]
        ],
        use_container_width=True,
    )