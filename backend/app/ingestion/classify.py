"""Label each segment (teaching / off_topic / admin / exam_hint) and give it a short topic name."""

import json
import logging
from pathlib import Path

from app.generation.llm import complete_json

log = logging.getLogger(__name__)

LABELS = ("teaching", "off_topic", "admin", "exam_hint")
BATCH = 20

SYSTEM = """You label segments of a lecture transcript. For each numbered segment return:
- label: one of
  teaching  - explains course content, including analogies, examples and recaps
  off_topic - jokes, personal stories, channel promotion, chit-chat unrelated to the subject
  admin     - logistics: attendance, deadlines, assignments, schedule, course announcements
  exam_hint - the speaker says something is important for exams/interviews or likely to be asked
- topic: the subject being discussed, 2-5 words (e.g. "Binary Search"). Empty for off_topic/admin.
When a segment mixes kinds, pick the one covering most of it; prefer exam_hint if a hint is present.
Return exactly one item per segment, using the segment's number as i."""

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "label": {"type": "string", "enum": list(LABELS)},
            "topic": {"type": "string"},
        },
        "required": ["i", "label", "topic"],
    },
}


def parse_labels(items, n: int) -> list[dict]:
    """Normalise model output into n {label, topic} dicts; anything missing or invalid becomes teaching."""
    out = [{"label": "teaching", "topic": ""} for _ in range(n)]
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        i, label = item.get("i"), item.get("label")
        if isinstance(i, int) and 0 <= i < n and label in LABELS:
            out[i] = {"label": label, "topic": str(item.get("topic") or "").strip()[:80]}
    return out


def classify_chunks(chunks: list[dict], cache_path: Path) -> list[dict]:
    """Returns chunks with `label` and `topic` added. Cached on disk; a failed LLM call leaves
    everything as teaching (not cached) so ingestion still completes and can be redone later."""
    labels = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached) == len(chunks):
            labels = cached
    if labels is None:
        labels = []
        try:
            for start in range(0, len(chunks), BATCH):
                batch = chunks[start : start + BATCH]
                prompt = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(batch))
                labels += parse_labels(complete_json(SYSTEM, prompt, SCHEMA), len(batch))
            cache_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        except Exception:
            log.exception("Segment classification failed; labelling all segments as teaching")
            labels = parse_labels([], len(chunks))
    return [{**c, **l} for c, l in zip(chunks, labels)]
