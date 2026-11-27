from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlmodel import Field, Session, SQLModel, create_engine

from app.config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Course(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    professor: str = ""
    description: str = ""
    created_at: datetime = Field(default_factory=_now)


class Lecture(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    course_id: int = Field(foreign_key="course.id", index=True)
    title: str = ""
    number: int = 0
    source_url: str
    video_id: str = ""  # YouTube id, used by the frontend player
    duration_sec: float = 0
    # queued | downloading | transcribing | indexing | indexed | failed
    status: str = "queued"
    error: str = ""
    summary: str = ""  # JSON: {summary, key_points: [{text, start_sec}]}
    # From the visual pipeline: screen_only | facecam_screen | facecam_only | board_close | board_far | unknown
    layout_type: str = ""
    visual_quality_score: float | None = None  # mean readability (0..1) of the frames with content
    slides_name: str = ""  # original file name of uploaded slides (PDF/PPTX), "" if none
    notes_name: str = ""  # original file name of uploaded notes (PDF/MD/TXT), "" if none
    created_at: datetime = Field(default_factory=_now)


_connect_args = {"check_same_thread": False} if settings.db_url.startswith("sqlite") else {}
engine = create_engine(settings.db_url, connect_args=_connect_args)


def init_db() -> None:
    settings.data_path.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)
    # create_all never alters existing tables, so add columns introduced after the first release.
    added = {
        "summary": "TEXT NOT NULL DEFAULT ''",
        "layout_type": "TEXT NOT NULL DEFAULT ''",
        "visual_quality_score": "FLOAT",
        "slides_name": "TEXT NOT NULL DEFAULT ''",
        "notes_name": "TEXT NOT NULL DEFAULT ''",
    }
    with engine.begin() as conn:
        existing = {c["name"] for c in inspect(conn).get_columns("lecture")}
        for name, ddl in added.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE lecture ADD COLUMN {name} {ddl}"))


def get_session():
    with Session(engine) as session:
        yield session
