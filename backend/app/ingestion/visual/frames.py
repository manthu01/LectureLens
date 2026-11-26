"""Key frames: the moments a lecture's screen or board holds content worth reading.

Decodes only the video's keyframes (YouTube encodes one about every 5 s), so a 25-minute lecture
takes seconds. Pixels that keep changing all lecture (a face-cam inset, a cursor) are masked out,
so a talking head never looks like new content. Writing only adds ink and a page turn
removes it, which tells the two apart. A frame is kept:
  - just before a big change (a page turn, a slide change, the board being wiped): the page at
    its fullest, the way a student would photograph it;
  - when writing has added enough since the last kept frame, so a long page is read in parts;
  - at the end of the video, if it differs from the last kept frame.
Each kept frame is valid from the previous kept frame's time up to its own: the content it shows
was put up (and talked about) during that span. A frame whose ink barely differs from the
previous kept frame is dropped as a duplicate.
"""

import json
from pathlib import Path

import numpy as np

SMALL = (320, 180)  # analysis resolution (w, h)
PIX = 30  # grey-level change that counts as a changed pixel
INK = 40  # a pixel this far from the frame's median grey is "ink" (text, strokes, figures)
VOLATILE = 0.25  # a pixel changing in more than this share of steps is masked as live video
REMOVED = 0.004  # share of the frame losing ink in one step = page turn / slide change / wipe
ADDED = 0.006  # share of the frame gaining ink since the last kept frame = enough new writing
LIVE = 0.03  # share of pixels changing in a step; two such steps in a row = live video, not writing
MIN_GAP = 8.0  # seconds; drop kept frames closer than this to the previous one


def _keyframes(video: Path):
    """Yield (time, PIL image) for each keyframe."""
    import av

    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = "NONKEY"
        for f in c.decode(s):
            if f.pts is not None:
                yield float(f.pts * s.time_base), f.to_image()


def select(times: list[float], greys: list[np.ndarray]) -> list[int]:
    """Indices of the frames to keep (see module docstring). Pure function, for tests.

    Writing only adds ink and a page turn removes it, so the two are told apart by ink lost
    versus ink gained, measured over the pixels that are not live video.
    """
    n = len(greys)
    if n == 0:
        return []
    changed = [np.abs(greys[i] - greys[i - 1]) > PIX for i in range(1, n)]
    mask = np.mean(changed, axis=0) <= VOLATILE if changed else np.ones_like(greys[0], dtype=bool)
    area = max(int(mask.sum()), 1)
    ink = [(np.abs(g - np.median(g)) > INK) & mask for g in greys]

    keep: list[int] = []
    ref = 0  # the frame new ink is measured against: last kept, or first frame of the current page
    live_prev = False
    for i in range(1, n):
        live = (changed[i - 1] & mask).sum() / area >= LIVE
        removed = (ink[i - 1] & ~ink[i]).sum() / area
        if removed >= REMOVED and not (live and live_prev):
            if not keep or keep[-1] != i - 1:  # keep the page at its fullest, just before the turn
                keep.append(i - 1)
            ref = i
        elif (ink[i] & ~ink[ref]).sum() / area >= ADDED and not (live and live_prev):
            keep.append(i)
            ref = i
        live_prev = live

    out: list[int] = []
    for i in sorted(set(keep)):
        if out and times[i] - times[out[-1]] < MIN_GAP:
            out[-1] = i  # two keeps close together: the later one shows more
            continue
        # Near-duplicate of the previous kept frame (nothing written or erased in between): skip.
        if out and max((ink[i] & ~ink[out[-1]]).sum(), (ink[out[-1]] & ~ink[i]).sum()) / area < ADDED / 2:
            continue
        out.append(i)
    if not out or (out[-1] != n - 1 and (ink[n - 1] ^ ink[out[-1]]).sum() / area >= ADDED / 2):
        out.append(n - 1)
    return out


def extract(video: Path, out_dir: Path) -> list[dict]:
    """Save kept frames as JPEGs in out_dir; return [{file, start_sec, end_sec}]. Cached in frames.json."""
    cache = out_dir / "frames.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    times, greys = [], []
    for t, img in _keyframes(video):  # first pass: small greyscale copies only
        times.append(t)
        greys.append(np.asarray(img.convert("L").resize(SMALL), dtype=np.int16))
    kept = select(times, greys)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {times[i] for i in kept}
    images = {t: img for t, img in _keyframes(video) if t in wanted}  # second pass: kept frames only
    result, prev_t = [], 0.0
    for i in kept:
        name = f"{int(times[i]):05d}.jpg"
        images[times[i]].save(out_dir / name, quality=85)
        result.append({"file": name, "start_sec": prev_t, "end_sec": times[i]})
        prev_t = times[i]
    cache.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
