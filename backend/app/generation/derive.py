"""Content derived from a lecture's segments: topic timeline, summary, practice quiz."""

from app.generation.llm import complete_json

LABEL_TITLES = {"off_topic": "Off topic", "admin": "Course info", "exam_hint": "Exam hint", "teaching": "Teaching"}
MAX_SEGMENTS_IN_PROMPT = 60


def build_timeline(segments: list[dict], duration_sec: float = 0) -> dict:
    """Merge consecutive segments on the same topic into chapters and total up teaching time.

    Segments overlap slightly, so each one is counted from its start to the next one's start.
    """
    segs = sorted(segments, key=lambda s: s["start_sec"])
    chapters: list[dict] = []
    for i, s in enumerate(segs):
        end = segs[i + 1]["start_sec"] if i + 1 < len(segs) else s["end_sec"]
        label = s.get("label", "teaching")
        title = s.get("topic") or LABEL_TITLES.get(label, label)
        last = chapters[-1] if chapters else None
        if last and last["label"] == label and last["title"] == title:
            last["end_sec"] = end
        else:
            chapters.append({"title": title, "label": label, "start_sec": s["start_sec"], "end_sec": end})
    teaching = sum(c["end_sec"] - c["start_sec"] for c in chapters if c["label"] in ("teaching", "exam_hint"))
    total = max(duration_sec, chapters[-1]["end_sec"] if chapters else 0)
    return {"chapters": chapters, "teaching_sec": round(teaching, 1), "total_sec": round(total, 1)}


def _teaching(segments: list[dict]) -> list[dict]:
    """Teaching segments in order, evenly thinned if there are too many for one prompt."""
    segs = sorted(
        (s for s in segments if s.get("label", "teaching") in ("teaching", "exam_hint")),
        key=lambda s: (s.get("lecture_number", 0), s["start_sec"]),
    )
    if len(segs) > MAX_SEGMENTS_IN_PROMPT:
        step = len(segs) / MAX_SEGMENTS_IN_PROMPT
        segs = [segs[int(i * step)] for i in range(MAX_SEGMENTS_IN_PROMPT)]
    return segs


def _numbered(segs: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(segs))


SUMMARY_SYSTEM = """You summarise one lecture from numbered transcript segments (automatic transcripts,
possibly with small errors). Use only what the segments say.
Return:
- summary: 3-5 plain sentences on what the lecture teaches.
- key_points: 4-8 of the most important takeaways, each one sentence, each with i = the number of the
  segment where it is explained."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "i": {"type": "integer"}},
                "required": ["text", "i"],
            },
        },
    },
    "required": ["summary", "key_points"],
}


def summarize(segments: list[dict]) -> dict:
    """Returns {summary, key_points: [{text, start_sec}]}. Raises if the LLM call fails."""
    segs = _teaching(segments)
    if not segs:
        return {"summary": "", "key_points": []}
    raw = complete_json(SUMMARY_SYSTEM, _numbered(segs), SUMMARY_SCHEMA)
    points = [
        {"text": str(p["text"]).strip(), "start_sec": segs[p["i"]]["start_sec"]}
        for p in raw.get("key_points", [])
        if isinstance(p, dict) and isinstance(p.get("i"), int) and 0 <= p["i"] < len(segs) and p.get("text")
    ]
    return {"summary": str(raw.get("summary", "")).strip(), "key_points": points}


QUIZ_SYSTEM = """You write multiple-choice practice questions from numbered lecture transcript segments.
Rules:
- Test understanding of what the segments teach. Use only facts stated in them.
- Each question has exactly 4 options and exactly one correct answer; wrong options must be plausible.
- answer_index is the 0-based position of the correct option. Vary its position.
- explanation: one sentence on why the answer is right.
- i: the number of the segment the question is based on.
- Spread the questions across different segments. Do not refer to "the segment" or "the lecture" in the wording."""

QUIZ_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "answer_index": {"type": "integer"},
            "explanation": {"type": "string"},
            "i": {"type": "integer"},
        },
        "required": ["question", "options", "answer_index", "explanation", "i"],
    },
}


def parse_quiz(raw, segs: list[dict]) -> list[dict]:
    """Keep only well-formed questions and attach the source lecture and timestamp."""
    out = []
    for q in raw if isinstance(raw, list) else []:
        if not isinstance(q, dict):
            continue
        options, answer, i = q.get("options"), q.get("answer_index"), q.get("i")
        if (
            not q.get("question")
            or not isinstance(options, list)
            or len(options) != 4
            or not isinstance(answer, int)
            or not 0 <= answer < 4
            or not isinstance(i, int)
            or not 0 <= i < len(segs)
        ):
            continue
        src = segs[i]
        out.append(
            {
                "question": str(q["question"]).strip(),
                "options": [str(o).strip() for o in options],
                "answer_index": answer,
                "explanation": str(q.get("explanation") or "").strip(),
                "lecture_id": src["lecture_id"],
                "lecture_number": src["lecture_number"],
                "start_sec": src["start_sec"],
            }
        )
    return out


def make_quiz(segments: list[dict], n: int) -> list[dict]:
    segs = _teaching(segments)
    if not segs:
        return []
    raw = complete_json(QUIZ_SYSTEM, f"Write {n} questions.\n\n{_numbered(segs)}", QUIZ_SCHEMA)
    return parse_quiz(raw, segs)[:n]
