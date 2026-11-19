import json
import re
from collections.abc import Iterator

from app.config import settings
from app.generation.llm import stream_completion
from app.retrieval.search import retrieve

NOT_COVERED = "This doesn't seem to be covered in your lectures."

SYSTEM_PROMPT = f"""You answer a student's question using ONLY the lecture excerpts provided.

Rules:
- Use only information in the excerpts. Never use outside knowledge.
- After each claim, cite the excerpt(s) it came from using their tags, e.g. [S2] or [S1][S4].
- If the excerpts do not contain the answer, reply with exactly: {NOT_COVERED}
- Be concise: a short paragraph or a few short lines. Plain text only, no Markdown.
- Excerpts are automatic transcripts and may contain small transcription errors."""

OVERVIEW_PROMPT = f"""You answer a student's question about what their lectures cover, using ONLY the lecture
overviews provided (a summary per lecture plus timestamped key points).

Rules:
- Use only the overviews. Never use outside knowledge.
- After each claim, cite the overview line(s) it came from using their tags, e.g. [S2] or [S1][S4].
- If the question names a lecture, describe that one. If it says "this lecture" without saying which,
  give a short overview of each lecture, one or two lines each.
- If the overviews do not answer the question, reply with exactly: {NOT_COVERED}
- Be concise. Plain text only, no Markdown."""

_CITE_RE = re.compile(r"\[S(\d+)\]")

# Questions about the lectures themselves ("what is this lecture about?") rather than their content.
# Such questions match no passage, so they are only tried after the normal search would refuse.
_CONTAINER_RE = re.compile(r"\b(lectures?|course|class(es)?|videos?|playlist|series|l\d+)\b", re.I)
_OVERVIEW_RE = re.compile(
    r"\b(about|cover(s|ed)?|contain(s|ed)?|summar\w*|overview|topics?|taught|teach(es)?|discuss(es|ed)?|"
    r"go(es)? over|went over|in (it|this|that|the|these|lecture)|what('s| is| was) in)\b",
    re.I,
)
_LECTURE_NO_RE = re.compile(r"\b(?:lecture|class|video|l)\s*#?\s*(\d+)\b", re.I)


def overview_lecture(question: str) -> tuple[bool, int | None]:
    """(is an overview question, lecture number it names or None)."""
    if not (_CONTAINER_RE.search(question) and _OVERVIEW_RE.search(question)):
        return False, None
    m = _LECTURE_NO_RE.search(question)
    return True, int(m.group(1)) if m else None


def overview_hits(course_id: int, number: int | None = None) -> list[dict]:
    """One pseudo-hit per lecture summary and per key point, from the summaries cached at ingest."""
    from sqlmodel import Session, select

    from app.db import Lecture, engine

    with Session(engine) as session:
        query = select(Lecture).where(Lecture.course_id == course_id, Lecture.status == "indexed")
        if number is not None:
            query = query.where(Lecture.number == number)
        lectures = session.exec(query.order_by(Lecture.number)).all()
    hits = []
    for lec in lectures:
        try:
            summary = json.loads(lec.summary) if lec.summary else {}
        except json.JSONDecodeError:
            summary = {}
        base = {
            "lecture_id": lec.id,
            "lecture_number": lec.number,
            "lecture_title": lec.title,
            "video_id": lec.video_id,
        }
        if summary.get("summary"):
            hits.append({**base, "id": f"overview-{lec.id}", "start_sec": 0,
                         "end_sec": lec.duration_sec, "text": "Summary: " + summary["summary"]})
        for j, p in enumerate(summary.get("key_points", [])):
            hits.append({**base, "id": f"overview-{lec.id}-{j}", "start_sec": p["start_sec"],
                         "end_sec": p["start_sec"], "text": "Key point: " + p["text"]})
    return hits


def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = [
        f"[S{i}] Lecture {h['lecture_number']}: {h['lecture_title']} "
        f"({fmt_time(h['start_sec'])}-{fmt_time(h['end_sec'])})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    ]
    return "Lecture excerpts:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def cited_indices(answer: str, n_hits: int) -> list[int]:
    """1-based excerpt numbers cited in the answer, in first-appearance order, valid ones only."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= n_hits and i not in seen:
            seen.append(i)
    return seen


def _citation(tag: int, h: dict) -> dict:
    return {
        "tag": f"S{tag}",
        "segment_id": h["id"],
        "lecture_id": h["lecture_id"],
        "lecture_number": h["lecture_number"],
        "lecture_title": h["lecture_title"],
        "video_id": h.get("video_id", ""),
        "start_sec": h["start_sec"],
        "end_sec": h["end_sec"],
        "snippet": h["text"][:240],
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def answer_stream(course_id: int, question: str, include_off_topic: bool = False) -> Iterator[str]:
    """SSE stream: `sources`, then `token` events, then one `final` (or `error`) event."""
    try:
        hits = retrieve(course_id, question, include_off_topic=include_off_topic)
        # Confidence gate: nothing relevant enough -> refuse without spending an LLM call.
        scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
        best = max(scores) if scores else None
        refuse = not hits or (best is not None and best < settings.rerank_threshold)
        system = SYSTEM_PROMPT
        if refuse:
            # "What is this lecture about?" matches no passage; answer it from the lecture summaries.
            is_overview, number = overview_lecture(question)
            overview = overview_hits(course_id, number) if is_overview else []
            if overview:
                hits, system, refuse = overview, OVERVIEW_PROMPT, False
        if refuse:
            closest = [_citation(i, h) for i, h in enumerate(hits[:3], 1)]
            yield _sse("final", {"answer": NOT_COVERED, "citations": closest, "supported": False})
            return
        # Sent first so the UI can turn [S#] tags into timestamp links while text streams.
        yield _sse("sources", {"sources": [_citation(i, h) for i, h in enumerate(hits, 1)]})
        parts: list[str] = []
        for delta in stream_completion(system, build_prompt(question, hits)):
            parts.append(delta)
            yield _sse("token", {"text": delta})
        answer = "".join(parts).strip()
        cited = cited_indices(answer, len(hits))
        supported = bool(cited) and NOT_COVERED not in answer
        # When unsupported, still show the closest moments so the student can check.
        shown = cited if supported else list(range(1, min(3, len(hits)) + 1))
        # Different excerpts can start at the same moment (e.g. a summary and a key point); show one chip.
        unique, moments = [], set()
        for i in shown:
            moment = (hits[i - 1]["lecture_id"], hits[i - 1]["start_sec"])
            if moment not in moments:
                moments.add(moment)
                unique.append(i)
        shown = unique
        yield _sse(
            "final",
            {
                "answer": answer,
                "citations": [_citation(i, hits[i - 1]) for i in shown],
                "supported": supported,
            },
        )
    except Exception as e:
        yield _sse("error", {"message": str(e)[:500]})
