import json
import logging
import threading

from sqlmodel import Session, select

from app.config import settings
from app.db import Lecture, engine
from app.generation import derive
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


def attach_slides(chunks: list[dict], work_dir) -> list[dict]:
    """Align uploaded slides (if any) to the chunks; without slides, chunks pass through unchanged."""
    from app.ingestion import slides

    deck = slides.load(work_dir)
    return slides.align(chunks, deck, store.embed_dense) if deck else chunks


def reindex_notes(lecture: Lecture, work_dir) -> None:
    """Index the lecture's uploaded notes (if any) from notes.json. index_lecture keeps notes
    points, but after a schema change the collection is rebuilt from scratch, so re-add them here."""
    from app.ingestion import notes

    try:  # the transcript is indexed already; missing notes must not fail the lecture
        passages = notes.load(work_dir)
        if passages:
            store.index_notes(lecture, passages)
    except Exception:
        log.warning("Re-indexing notes failed for lecture %s", lecture.id, exc_info=True)


def relink_slides(lecture_id: int) -> None:
    """Re-align an indexed lecture's segments to newly uploaded slides and re-index them.
    Works from the index itself, so no transcript or video is needed."""
    with _lock, Session(engine) as session:
        lecture = session.get(Lecture, lecture_id)
        if lecture is None or lecture.status not in ("indexed", "indexing"):
            return
        work_dir = settings.data_path / "courses" / str(lecture.course_id) / str(lecture.id)
        keys = ("start_sec", "end_sec", "text", "label", "topic", "visual_text")
        chunks = [{k: s.get(k, "") for k in keys} for s in store.lecture_segments(lecture_id)]
        try:
            _set(session, lecture, status="indexing")
            store.index_lecture(lecture, attach_slides(chunks, work_dir))
            _set(session, lecture, status="indexed", error="")
        except Exception as e:
            log.exception("Slide alignment failed for lecture %s", lecture_id)
            session.rollback()
            # The old segments are still in the index; only the slide links are missing.
            _set(session, lecture, status="indexed", error=f"Slide alignment failed: {str(e)[:300]}")


def run_ingest(lecture_id: int) -> None:
    """Download -> transcribe -> chunk -> classify -> read the screen -> align slides -> embed -> index. Each step caches to disk, so a retry resumes."""
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
            if settings.visual_pipeline:
                try:  # optional: without it the lecture is still searchable by its transcript
                    from app.ingestion import visual

                    chunks, layout, quality = visual.run(lecture.source_url, work_dir, chunks)
                    _set(session, lecture, layout_type=layout, visual_quality_score=quality)
                except Exception:
                    log.warning("Visual pipeline failed for lecture %s; indexing the transcript only",
                                lecture_id, exc_info=True)
            chunks = attach_slides(chunks, work_dir)
            n = store.index_lecture(lecture, chunks)
            if n == 0:
                raise RuntimeError("No speech found in this video")
            reindex_notes(lecture, work_dir)
            _set(session, lecture, status="indexed")
            try:  # nice-to-have; the summary endpoint retries on demand if this fails
                _set(session, lecture, summary=json.dumps(derive.summarize(chunks), ensure_ascii=False))
            except Exception:
                log.warning("Summary generation failed for lecture %s", lecture_id, exc_info=True)
        except Exception as e:  # surface the reason in the UI and allow retry
            log.exception("Ingest failed for lecture %s", lecture_id)
            session.rollback()
            _set(session, lecture, status="failed", error=str(e)[:500])
