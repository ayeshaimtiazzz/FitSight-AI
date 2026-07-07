"""
Database engine and session factory. Uses a local SQLite file at data/fitsight.db,
created automatically on first run.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fitness_coach.storage.models import Base

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "fitsight.db")
DB_URL = f"sqlite:///{DB_PATH}"

os.makedirs(DB_DIR, exist_ok=True)

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Creates all tables if they don't already exist. Safe to call every run."""
    Base.metadata.create_all(engine)


def get_db_session():
    """Returns a new SQLAlchemy session. Caller is responsible for closing it."""
    return SessionLocal()