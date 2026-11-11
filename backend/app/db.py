from datetime import datetime, timezone

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
    created_at: datetime = Field(default_factory=_now)


_connect_args = {"check_same_thread": False} if settings.db_url.startswith("sqlite") else {}
engine = create_engine(settings.db_url, connect_args=_connect_args)


def init_db() -> None:
    settings.data_path.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
