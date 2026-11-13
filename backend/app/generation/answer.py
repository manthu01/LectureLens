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

_CITE_RE = re.compile(r"\[S(\d+)\]")


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
        if not hits or (best is not None and best < settings.rerank_threshold):
            closest = [_citation(i, h) for i, h in enumerate(hits[:3], 1)]
            yield _sse("final", {"answer": NOT_COVERED, "citations": closest, "supported": False})
            return
        # Sent first so the UI can turn [S#] tags into timestamp links while text streams.
        yield _sse("sources", {"sources": [_citation(i, h) for i, h in enumerate(hits, 1)]})
        parts: list[str] = []
        for delta in stream_completion(SYSTEM_PROMPT, build_prompt(question, hits)):
            parts.append(delta)
            yield _sse("token", {"text": delta})
        answer = "".join(parts).strip()
        cited = cited_indices(answer, len(hits))
        supported = bool(cited) and NOT_COVERED not in answer
        # When unsupported, still show the closest moments so the student can check.
        shown = cited if supported else list(range(1, min(3, len(hits)) + 1))
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
