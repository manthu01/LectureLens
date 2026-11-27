"""Read key frames with a vision model: what is written on the slide/screen/board, plus the layout.

Layout detection and reading happen in the same call, so routing costs nothing extra. The text
is what a student would copy down: headings, bullet points, handwriting, equations (LaTeX) and a
short note for each diagram. App chrome (menus, toolbars, window titles) and the face cam are
ignored. Frames are sent several per call to stay inside free-tier request limits.

A frame that comes back hard to read (a board filmed from the back, a low-readability screen or
board) gets a second read of a cropped, upscaled copy (enhance.py); the better read is kept.
"""

import io
import json
import logging
from collections import Counter
from pathlib import Path

from app.config import settings
from app.generation.llm import Blocked, complete_json
from app.ingestion.visual import enhance

log = logging.getLogger(__name__)

LAYOUTS = ("screen_only", "facecam_screen", "facecam_only", "board_close", "board_far", "unknown")
BATCH = 6
MAX_WIDTH = 1280
CONTENT_LAYOUTS = ("screen_only", "facecam_screen", "board_close", "board_far")
RETRY_BELOW = 0.6  # a content frame read less confidently than this gets an enhanced second read

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


def _read_images(names: list[str], images: list[bytes]) -> list[dict]:
    """One vision call for the images (JPEG bytes); `names` (frame files) are for the log only."""
    prompt = f"Read the {len(images)} images above, numbered [0] to [{len(images) - 1}]."
    try:
        return parse_reads(complete_json(SYSTEM, prompt, SCHEMA, images=images, model=settings.vision_model), len(images))
    except Blocked as e:
        # A frame showing copyrighted text (a web page, a book) can trip the RECITATION filter
        # and take the whole batch with it: read the frames one by one, and record a frame that is
        # still refused as unreadable rather than stopping the lecture.
        if len(images) > 1:
            return [r for n, b in zip(names, images) for r in _read_images([n], [b])]
        log.warning("Vision model refused frame %s (%s); left empty", names[0], e)
        return [{"layout": "unknown", "text": "", "readable": 0.0, "blocked": str(e)}]


def _read_batch(batch: list[dict], frame_dir: Path) -> list[dict]:
    return _read_images([f["file"] for f in batch], [_jpeg(frame_dir / f["file"]) for f in batch])


def needs_enhance(r: dict) -> bool:
    """A read worth a second try on a cropped, upscaled copy: a far board, or a screen/board read
    with low confidence. "enhanced" is set once tried, so a rerun never tries a frame twice."""
    if "enhanced" in r or "blocked" in r:
        return False
    return r["layout"] == "board_far" or (r["layout"] in CONTENT_LAYOUTS and r["readable"] < RETRY_BELOW)


def better_read(first: dict, second: dict) -> dict:
    """The read with the higher readability, marked with which one won ("enhanced"). The layout
    stays the first read's: the crop shows the board up close, but it was filmed from afar."""
    if second["readable"] > first["readable"]:
        return {**first, "text": second["text"], "readable": second["readable"], "enhanced": True}
    return {**first, "enhanced": False}


def _enhance_reads(frames: list[dict], frame_dir: Path, cache: dict, cache_path: Path) -> None:
    """Second read of the hard-to-read frames, on crops batched like the first read. Saved after
    each call, so a rate limit here also resumes where it stopped."""
    todo = []
    for f in frames:
        if needs_enhance(cache[f["file"]]):
            crop = enhance.enhanced_jpeg(frame_dir / f["file"])
            if crop is None:  # no clear content region: a crop would add nothing
                cache[f["file"]] = {**cache[f["file"]], "enhanced": False}
            else:
                todo.append((f["file"], crop))
    _save(cache, cache_path)
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        for (name, _), r in zip(batch, _read_images([n for n, _ in batch], [b for _, b in batch])):
            cache[name] = better_read(cache[name], r)
        _save(cache, cache_path)


def _save(cache: dict, cache_path: Path) -> None:
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def read_frames(frames: list[dict], frame_dir: Path) -> list[dict]:
    """Add layout/text/readable to each frame dict. Cached per frame in frame_dir/reads.json, so a
    run stopped by a rate limit resumes where it left off; raises on the first failed call.
    Hard-to-read frames then get an enhanced second read, cached the same way."""
    cache_path = frame_dir / "reads.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [f for f in frames if f["file"] not in cache]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        for f, r in zip(batch, _read_batch(batch, frame_dir)):
            cache[f["file"]] = r
        _save(cache, cache_path)
    _enhance_reads(frames, frame_dir, cache, cache_path)
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
