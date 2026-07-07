"""
Generates a 2-page ReportLab PDF workout summary:
Page 1: session stats + a Plotly chart exported as PNG
Page 2: coaching feedback notes from the session
"""
import io
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet


def generate_session_pdf(session, rep_events, coaching_notes, chart_figure, output_path: str):
    """
    session: a WorkoutSession ORM object
    rep_events: list of RepEvent ORM objects for this session
    coaching_notes: list of CoachingFeedback ORM objects for this session
    chart_figure: a Plotly Figure to embed (e.g. form_score_trend_chart output)
    output_path: where to write the PDF file
    """
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    # --- Page 1: Summary ---
    story.append(Paragraph(f"FitSight AI — Workout Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Exercise: {session.exercise.title()}", styles["Normal"]))
    story.append(Paragraph(f"Date: {session.date.strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Duration: {session.duration_seconds:.0f} seconds", styles["Normal"]))
    story.append(Paragraph(f"Total Reps: {session.total_reps}", styles["Normal"]))
    story.append(Paragraph(f"Good Form Reps: {session.good_form_reps}", styles["Normal"]))
    story.append(Paragraph(f"Average Form Score: {session.avg_form_score:.1f}%", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Export the Plotly chart as a PNG image in-memory, embed it in the PDF.
    # Requires the 'kaleido' package for Plotly's static image export.
    try:
        img_bytes = chart_figure.to_image(format="png", width=600, height=350)
        img_buffer = io.BytesIO(img_bytes)
        story.append(RLImage(img_buffer, width=5.5 * inch, height=3.2 * inch))
    except Exception as e:
        story.append(Paragraph(f"(Chart could not be rendered: {e})", styles["Normal"]))

    story.append(PageBreak())

    # --- Page 2: Coaching notes ---
    story.append(Paragraph("Coaching Feedback", styles["Title"]))
    story.append(Spacer(1, 12))
    if not coaching_notes:
        story.append(Paragraph("No coaching feedback was generated this session.", styles["Normal"]))
    else:
        for note in coaching_notes:
            story.append(Paragraph(f"<b>Rep {note.rep_number}:</b> {note.feedback_text}", styles["Normal"]))
            story.append(Spacer(1, 10))

    doc.build(story)
    return output_path