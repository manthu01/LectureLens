"""Read key frames with a vision model: what is written on the slide/screen/board, plus the layout.

Layout detection and reading happen in the same call, so routing costs nothing extra. The text
is what a student would copy down: headings, bullet points, handwriting, equations (LaTeX) and a
short note for each diagram. App chrome (menus, toolbars, window titles) and the face cam are
ignored. Frames are sent several per call to stay inside free-tier request limits.
"""

import io
import json
import logging
from collections import Counter
from pathlib import Path

from app.config import settings
from app.generation.llm import Blocked, complete_json

log = logging.getLogger(__name__)

LAYOUTS = ("screen_only", "facecam_screen", "facecam_only", "board_close", "board_far", "unknown")
BATCH = 6
MAX_WIDTH = 1280

SYSTEM = """You read frames from a recorded lecture video. For each numbered image return:
- layout: one of
  screen_only     - a slide, document, notebook or drawing app fills the frame, no camera
  facecam_screen  - screen content with a small camera picture of the speaker
  facecam_only    - only the speaker (talking head), no readable teaching content
  board_close     - a whiteboard/blackboard filmed up close, writing readable
  board_far       - a board filmed from far away (back of the class), writing small or blurry
- text: the teaching content visible, as a student would copy it into notes: titles, headings,
  bullet points, handwritten words, equations (LaTeX), table contents. Describe each diagram or
  figure in one line in square brackets, e.g. [diagram: data -> model -> predictions, with a
  feedback arrow from server back to model]. Fix obvious handwriting misspellings only when sure.
  Ignore app menus, toolbars, window titles, account names, dates/times of the app, and the face
  cam. Empty if there is no teaching content.
- readable: 0 to 1, how confidently the content could be read (1 = clean text; 0 = nothing legible).
Return exactly one item per image, using the image's number (given before it) as i."""

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "layout": {"type": "string", "enum": list(LAYOUTS)},
            "text": {"type": "string"},
            "readable": {"type": "number"},
        },
        "required": ["i", "layout", "text", "readable"],
    },
}


def _jpeg(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > MAX_WIDTH:
            im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()


def parse_reads(items, n: int) -> list[dict]:
    """Normalise model output into n {layout, text, readable} dicts; missing ones become unknown/empty."""
    out = [{"layout": "unknown", "text": "", "readable": 0.0} for _ in range(n)]
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        i = item.get("i")
        if not (isinstance(i, int) and 0 <= i < n):
            continue
        layout = item.get("layout") if item.get("layout") in LAYOUTS else "unknown"
        try:
            readable = min(max(float(item.get("readable") or 0), 0.0), 1.0)
        except (TypeError, ValueError):
            readable = 0.0
        out[i] = {"layout": layout, "text": str(item.get("text") or "").strip(), "readable": readable}
    return out


def _read_batch(batch: list[dict], frame_dir: Path) -> list[dict]:
    images = [_jpeg(frame_dir / f["file"]) for f in batch]
    prompt = f"Read the {len(batch)} images above, numbered [0] to [{len(batch) - 1}]."
    try:
        return parse_reads(complete_json(SYSTEM, prompt, SCHEMA, images=images, model=settings.vision_model), len(batch))
    except Blocked as e:
        # A frame showing copyrighted text (a web page, a book) can trip the RECITATION filter
        # and take the whole batch with it: read the frames one by one, and record a frame that is
        # still refused as unreadable rather than stopping the lecture.
        if len(batch) > 1:
            return [r for f in batch for r in _read_batch([f], frame_dir)]
        log.warning("Vision model refused frame %s (%s); left empty", batch[0]["file"], e)
        return [{"layout": "unknown", "text": "", "readable": 0.0, "blocked": str(e)}]


def read_frames(frames: list[dict], frame_dir: Path) -> list[dict]:
    """Add layout/text/readable to each frame dict. Cached per frame in frame_dir/reads.json, so a
    run stopped by a rate limit resumes where it left off; raises on the first failed call."""
    cache_path = frame_dir / "reads.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [f for f in frames if f["file"] not in cache]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        for f, r in zip(batch, _read_batch(batch, frame_dir)):
            cache[f["file"]] = r
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return [{**f, **cache[f["file"]]} for f in frames]


def lecture_layout(reads: list[dict]) -> tuple[str, float]:
    """(layout_type, visual_quality_score) for a lecture.

    Layout: the layout on screen for the most time (a short talking-head intro doesn't make a
    screen lecture "facecam_only", and one slide doesn't make a talking-head lecture a screen one).
    Quality: time-weighted mean readability of the frames with content; 0 if none have any.
    """
    def dur(r: dict) -> float:
        return max(r["end_sec"] - r["start_sec"], 1.0)

    weights: Counter = Counter()
    for r in reads:
        weights[r["layout"]] += dur(r)
    layout = weights.most_common(1)[0][0] if weights else "unknown"
    content = [r for r in reads if r["text"] and r["layout"] not in ("facecam_only", "unknown")]
    if not content:
        return layout, 0.0
    quality = sum(r["readable"] * dur(r) for r in content) / sum(dur(r) for r in content)
    return layout, round(quality, 3)
