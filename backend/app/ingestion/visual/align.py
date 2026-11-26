"""Attach on-screen text to transcript segments by time."""

MAX_CHARS = 1200


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def visual_text_for(start: float, end: float, reads: list[dict]) -> str:
    """Text of the frames on screen during [start, end]. A frame counts from its start_sec (when
    its content began to go up) to its end_sec (when it was captured, at its fullest). When one
    frame's text is contained in a later one (the page grew), only the fuller one is kept."""
    texts = [r["text"].strip() for r in reads if r["text"].strip() and r["start_sec"] < end and r["end_sec"] > start]
    kept: list[str] = []
    for i, t in enumerate(texts):
        n = _norm(t)
        if any(n in _norm(o) for o in texts[i + 1 :]) or any(n == _norm(k) for k in kept):
            continue
        kept.append(t)
    out = "\n".join(kept)
    return out if len(out) <= MAX_CHARS else out[:MAX_CHARS].rsplit(" ", 1)[0] + " ..."


def attach(chunks: list[dict], reads: list[dict]) -> list[dict]:
    return [{**c, "visual_text": visual_text_for(c["start_sec"], c["end_sec"], reads)} for c in chunks]
