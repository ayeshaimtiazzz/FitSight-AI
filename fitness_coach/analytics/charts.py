"""
Plotly chart builders, shared between the Streamlit dashboard and the PDF
report generator so both stay visually consistent.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def rep_timeline_chart(sessions_df: pd.DataFrame) -> go.Figure:
    """Bar chart: total reps per session over time."""
    fig = px.bar(
        sessions_df, x="date", y="total_reps", color="exercise",
        title="Reps per Session Over Time",
        labels={"date": "Session Date", "total_reps": "Total Reps"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def form_score_trend_chart(sessions_df: pd.DataFrame) -> go.Figure:
    """Line chart with trend: average form score per session."""
    fig = px.line(
        sessions_df, x="date", y="avg_form_score", color="exercise",
        markers=True,
        title="Form Score Trend Over Time",
        labels={"date": "Session Date", "avg_form_score": "Avg Form Score (%)"},
    )
    fig.update_layout(template="plotly_white", yaxis_range=[0, 100])
    return fig


def joint_angle_distribution_chart(rep_events_df: pd.DataFrame) -> go.Figure:
    """Violin plot: knee angle distribution, split by good_form vs bad_form."""
    df = rep_events_df.copy()
    df["form_label"] = df["is_good_form"].map({1: "Good Form", 0: "Bad Form"})
    fig = px.violin(
        df, y="joint_angle", x="form_label", color="form_label",
        box=True, points="all",
        title="Joint Angle Distribution: Good vs Bad Form Reps",
        labels={"joint_angle": "Joint Angle (degrees)", "form_label": ""},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def personal_records_table(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a small summary table of best session metrics per exercise."""
    if sessions_df.empty:
        return pd.DataFrame(columns=["exercise", "best_reps", "best_form_score", "total_sessions"])

    grouped = sessions_df.groupby("exercise").agg(
        best_reps=("total_reps", "max"),
        best_form_score=("avg_form_score", "max"),
        total_sessions=("id", "count"),
    ).reset_index()
    return grouped