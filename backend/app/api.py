import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, func, select

from app.db import Course, Lecture, get_session
from app.generation import derive
from app.generation.answer import answer_stream
from app.ingestion.acquire import list_playlist
from app.ingestion.pipeline import run_ingest
from app.retrieval import store
from app.retrieval.search import retrieve

router = APIRouter()


class CourseIn(BaseModel):
    name: str
    professor: str = ""
    description: str = ""


class LectureIn(BaseModel):
    url: str
    title: str = ""
    number: int | None = None


class PlaylistIn(BaseModel):
    url: str
    limit: int | None = None


class AskIn(BaseModel):
    question: str
    include_off_topic: bool = False


class QuizIn(BaseModel):
    lecture_ids: list[int]
    n: int = 5


class SearchIn(BaseModel):
    question: str
    mode: str | None = None  # dense | bm25 | hybrid
    rerank: bool | None = None
    k: int | None = None
    include_off_topic: bool = False


def _get(session: Session, model, id_: int):
    obj = session.get(model, id_)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {id_} not found")
    return obj


@router.post("/courses")
def create_course(body: CourseIn, session: Session = Depends(get_session)) -> Course:
    course = Course(**body.model_dump())
    session.add(course)
    session.commit()
    session.refresh(course)
    return course


@router.get("/courses")
def list_courses(session: Session = Depends(get_session)) -> list[Course]:
    return list(session.exec(select(Course).order_by(Course.id)))


@router.get("/courses/{course_id}")
def get_course(course_id: int, session: Session = Depends(get_session)) -> dict:
    course = _get(session, Course, course_id)
    lectures = session.exec(select(Lecture).where(Lecture.course_id == course_id).order_by(Lecture.number))
    return {**course.model_dump(), "lectures": list(lectures)}


@router.post("/courses/{course_id}/lectures")
def add_lecture(
    course_id: int, body: LectureIn, tasks: BackgroundTasks, session: Session = Depends(get_session)
) -> Lecture:
    _get(session, Course, course_id)
    number = body.number
    if number is None:
        current = session.exec(select(func.max(Lecture.number)).where(Lecture.course_id == course_id)).one()
        number = (current or 0) + 1
    lecture = Lecture(course_id=course_id, source_url=body.url, title=body.title, number=number)
    session.add(lecture)
    session.commit()
    session.refresh(lecture)
    tasks.add_task(run_ingest, lecture.id)
    return lecture


@router.post("/courses/{course_id}/playlist")
def add_playlist(
    course_id: int, body: PlaylistIn, tasks: BackgroundTasks, session: Session = Depends(get_session)
) -> list[Lecture]:
    """Add every video of a playlist (optionally only the first `limit`) as lectures. Skips ones already added."""
    _get(session, Course, course_id)
    try:
        entries = list_playlist(body.url, body.limit)
    except Exception as e:
        raise HTTPException(400, f"Could not read playlist: {str(e)[:300]}")
    existing = session.exec(select(Lecture).where(Lecture.course_id == course_id)).all()
    known = {l.source_url for l in existing}
    number = max((l.number for l in existing), default=0)
    created = []
    for e in entries:
        if e["url"] in known:
            continue
        number += 1
        lecture = Lecture(
            course_id=course_id,
            source_url=e["url"],
            title=e["title"],
            number=number,
            video_id=e["video_id"],
            duration_sec=e["duration_sec"],
        )
        session.add(lecture)
        created.append(lecture)
    session.commit()
    for lecture in created:
        session.refresh(lecture)
        tasks.add_task(run_ingest, lecture.id)
    return created


@router.get("/lectures/{lecture_id}")
def get_lecture(lecture_id: int, session: Session = Depends(get_session)) -> Lecture:
    return _get(session, Lecture, lecture_id)


@router.post("/lectures/{lecture_id}/retry")
def retry_lecture(lecture_id: int, tasks: BackgroundTasks, session: Session = Depends(get_session)) -> Lecture:
    """Re-run ingestion for a failed lecture, or re-index a finished one (cached steps are reused)."""
    lecture = _get(session, Lecture, lecture_id)
    if lecture.status not in ("failed", "indexed"):
        raise HTTPException(409, f"Lecture is already being processed ({lecture.status})")
    lecture.status = "queued"
    session.add(lecture)
    session.commit()
    session.refresh(lecture)
    tasks.add_task(run_ingest, lecture.id)
    return lecture


@router.get("/lectures/{lecture_id}/transcript")
def get_transcript(lecture_id: int, session: Session = Depends(get_session)) -> list[dict]:
    _get(session, Lecture, lecture_id)
    return store.lecture_segments(lecture_id)


@router.get("/lectures/{lecture_id}/timeline")
def get_timeline(lecture_id: int, session: Session = Depends(get_session)) -> dict:
    lecture = _get(session, Lecture, lecture_id)
    return derive.build_timeline(store.lecture_segments(lecture_id), lecture.duration_sec)


@router.get("/lectures/{lecture_id}/summary")
def get_summary(lecture_id: int, session: Session = Depends(get_session)) -> dict:
    """Cached summary; generated on first request for lectures indexed before summaries existed."""
    lecture = _get(session, Lecture, lecture_id)
    if lecture.summary:
        return json.loads(lecture.summary)
    if lecture.status != "indexed":
        raise HTTPException(409, f"Lecture is not ready ({lecture.status})")
    try:
        summary = derive.summarize(store.lecture_segments(lecture_id))
    except Exception as e:
        raise HTTPException(502, f"Could not generate summary: {str(e)[:300]}")
    lecture.summary = json.dumps(summary, ensure_ascii=False)
    session.add(lecture)
    session.commit()
    return summary


@router.post("/courses/{course_id}/quiz")
def quiz(course_id: int, body: QuizIn, session: Session = Depends(get_session)) -> list[dict]:
    """Multiple-choice questions from the teaching segments of the chosen lectures."""
    _get(session, Course, course_id)
    segments = []
    for lecture_id in body.lecture_ids:
        lecture = _get(session, Lecture, lecture_id)
        if lecture.course_id != course_id:
            raise HTTPException(422, f"Lecture {lecture_id} is not in this course")
        segments += store.lecture_segments(lecture_id)
    if not segments:
        raise HTTPException(409, "The chosen lectures have no indexed content yet")
    try:
        return derive.make_quiz(segments, max(1, min(body.n, 15)))
    except Exception as e:
        raise HTTPException(502, f"Could not generate quiz: {str(e)[:300]}")


@router.post("/courses/{course_id}/search")
def search(course_id: int, body: SearchIn, session: Session = Depends(get_session)) -> list[dict]:
    """Retrieval only, with per-request overrides. Used by the eval script for ablations."""
    _get(session, Course, course_id)
    try:
        return retrieve(
            course_id,
            body.question,
            mode=body.mode,
            rerank=body.rerank,
            k=body.k,
            include_off_topic=body.include_off_topic,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/courses/{course_id}/exam-hints")
def exam_hints(course_id: int, session: Session = Depends(get_session)) -> list[dict]:
    _get(session, Course, course_id)
    return store.course_segments_by_label(course_id, "exam_hint")


@router.get("/courses/{course_id}/announcements")
def announcements(course_id: int, session: Session = Depends(get_session)) -> list[dict]:
    _get(session, Course, course_id)
    return store.course_segments_by_label(course_id, "admin")


@router.post("/courses/{course_id}/ask")
def ask(course_id: int, body: AskIn, session: Session = Depends(get_session)) -> StreamingResponse:
    _get(session, Course, course_id)
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Question is empty")
    return StreamingResponse(
        answer_stream(course_id, question, body.include_off_topic),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
