import logging
import threading

from sqlmodel import Session, select

from app.config import settings
from app.db import Lecture, engine
from app.ingestion.acquire import download_audio
from app.ingestion.chunk import chunk_transcript
from app.ingestion.classify import classify_chunks
from app.ingestion.transcribe import transcribe
from app.retrieval import store

log = logging.getLogger(__name__)
_lock = threading.Lock()


def resume_unfinished() -> None:
    """Re-run lectures left mid-ingest by a server restart. Cached steps make this cheap."""
    with Session(engine) as session:
        if not store.collection_exists():
            # Index schema changed (or was deleted): rebuild from cached transcripts.
            for lecture in session.exec(select(Lecture).where(Lecture.status == "indexed")):
                lecture.status = "queued"
                session.add(lecture)
            session.commit()
        ids = session.exec(
            select(Lecture.id).where(Lecture.status.not_in(["indexed", "failed"])).order_by(Lecture.id)
        ).all()
    for lecture_id in ids:
        run_ingest(lecture_id)


def _set(session: Session, lecture: Lecture, **fields) -> None:
    for k, v in fields.items():
        setattr(lecture, k, v)
    session.add(lecture)
    session.commit()
    session.refresh(lecture)


def run_ingest(lecture_id: int) -> None:
    """Download -> transcribe -> chunk -> embed -> index. Each step caches to disk, so a retry resumes."""
    # One lecture at a time: transcription saturates the CPU, and the lecture stays "queued" while waiting.
    with _lock, Session(engine) as session:
        lecture = session.get(Lecture, lecture_id)
        if lecture is None or lecture.status == "indexed":
            return
        work_dir = settings.data_path / "courses" / str(lecture.course_id) / str(lecture.id)
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            _set(session, lecture, status="downloading", error="")
            meta = download_audio(lecture.source_url, work_dir)
            _set(
                session,
                lecture,
                title=lecture.title or meta["title"],
                duration_sec=meta["duration_sec"],
                video_id=meta["video_id"],
                status="transcribing",
            )
            segs = transcribe(meta["audio_path"], work_dir / "transcript.json")

            _set(session, lecture, status="indexing")
            chunks = chunk_transcript(segs)
            if settings.classify_segments:
                chunks = classify_chunks(chunks, work_dir / "labels.json")
            n = store.index_lecture(lecture, chunks)
            if n == 0:
                raise RuntimeError("No speech found in this video")
            _set(session, lecture, status="indexed")
        except Exception as e:  # surface the reason in the UI and allow retry
            log.exception("Ingest failed for lecture %s", lecture_id)
            session.rollback()
            _set(session, lecture, status="failed", error=str(e)[:500])
