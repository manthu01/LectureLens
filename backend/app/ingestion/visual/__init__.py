"""Visual pipeline: read the slides / screen / board shown in a lecture video.

download video -> key frames (frames.py) -> vision model reads each frame and names the layout
(read.py) -> text attached to the transcript segments shown alongside it (align.py).
Every step caches to data/courses/<course>/<lecture>/, so a rerun only redoes what is missing.
"""

import json
import logging
from pathlib import Path

from app.ingestion.visual import align, frames, read

log = logging.getLogger(__name__)


def run(source: str, work_dir: Path, chunks: list[dict]) -> tuple[list[dict], str, float]:
    """Returns (chunks with visual_text, layout_type, visual_quality_score).
    Raises on failure; the caller decides whether that is fatal."""
    from app.ingestion.acquire import download_video

    video = download_video(source, work_dir)
    frame_dir = work_dir / "frames"
    kept = frames.extract(video, frame_dir)
    reads = read.read_frames(kept, frame_dir)
    layout, quality = read.lecture_layout(reads)
    return align.attach(chunks, reads), layout, quality


def cached_reads(work_dir: Path) -> list[dict] | None:
    """Frame reads from an earlier run, or None if the visual pipeline never finished here."""
    frame_dir = work_dir / "frames"
    try:
        kept = json.loads((frame_dir / "frames.json").read_text(encoding="utf-8"))
        cache = json.loads((frame_dir / "reads.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not all(f["file"] in cache for f in kept):
        return None
    return [{**f, **cache[f["file"]]} for f in kept]
