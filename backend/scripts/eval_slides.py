"""How well are uploaded slides aligned to the transcript? Needs a lecture that has been through
the visual pipeline (its frame reads give the gold labels) and has slides uploaded.

    backend/.venv/Scripts/python backend/scripts/eval_slides.py --lecture 11

Gold: each key frame's text is matched to the slide containing the largest share of the frame's
words (at least 60%; containment rather than Jaccard, because slides that reveal their bullets one
by one are shown partly; frames matching no slide, e.g. a live Python shell, get none). A segment's gold
slide is the one on screen for the longest part of it; segments with no gold slide are skipped.
Writes the gold labels to eval/results/slides-gold-lecture<N>.json for spot checks.

Compared:
  transcript, path       the app's alignment without on-screen text: monotonic dynamic program
  transcript, best each  each segment independently gets its highest-scoring slide (no ordering)
  transcript, dense only the same path, scored by embeddings alone (no TF-IDF)
  + on-screen text       the app's alignment when the visual pipeline ran. The gold labels come
                         from the same frame reads, so this one is optimistic.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.ingestion import slides as slides_mod  # noqa: E402
from app.ingestion import visual  # noqa: E402
from app.retrieval import store  # noqa: E402

MIN_SHARE = 0.6
STOP = set("the a an of to and or is are in on for with by be as it this that from at if".split())


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 1}


def frame_slide(text: str, deck: list[dict]) -> int | None:
    fw = words(text)
    if len(fw) < 3:
        return None
    best, page = 0.0, None
    for s in deck:
        sw = words(s["text"])
        if sw:
            share = len(fw & sw) / len(fw)
            if share > best:
                best, page = share, s["page"]
    return page if best >= MIN_SHARE else None


def gold_for(seg: dict, frames: list[dict]) -> int | None:
    on_screen: dict[int, float] = {}
    for f in frames:
        if f["slide"] is None:
            continue
        overlap = min(seg["end_sec"], f["end_sec"]) - max(seg["start_sec"], f["start_sec"])
        if overlap > 0:
            on_screen[f["slide"]] = on_screen.get(f["slide"], 0) + overlap
    return max(on_screen, key=on_screen.get) if on_screen else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecture", type=int, required=True, help="lecture id")
    args = ap.parse_args()

    import sqlite3

    db = sqlite3.connect(settings.data_path / "lecturelens.db")
    course_id = db.execute("select course_id from lecture where id = ?", (args.lecture,)).fetchone()[0]
    work_dir = settings.data_path / "courses" / str(course_id) / str(args.lecture)
    deck = slides_mod.load(work_dir)
    reads = visual.cached_reads(work_dir)
    if not deck or not reads:
        sys.exit("needs uploaded slides and frame reads for this lecture")

    frames = [{**r, "slide": frame_slide(r["text"], deck)} for r in reads]
    segs = store.lecture_segments(args.lecture)
    for s in segs:
        s["gold"] = gold_for(s, frames)
    scored = [s for s in segs if s["gold"] is not None and s.get("label") != "off_topic"]

    def score(pred: list[int | None]) -> dict:
        p = {s["id"]: x for s, x in zip(segs, pred)}
        exact = sum(p[s["id"]] == s["gold"] for s in scored)
        near = sum(p[s["id"]] is not None and abs(p[s["id"]] - s["gold"]) <= 1 for s in scored)
        return {"exact": exact / len(scored), "within_1": near / len(scored)}

    plain = [{**s, "visual_text": ""} for s in segs]
    path_t = [c["slide_page"] for c in slides_mod.align(plain, deck, store.embed_dense)]
    path_v = [c["slide_page"] for c in slides_mod.align(segs, deck, store.embed_dense)]
    usable = [s for s in deck if s["text"].strip()]
    seg_texts, slide_texts = [s["text"] for s in plain], [s["text"] for s in usable]
    sim = slides_mod.similarity(seg_texts, slide_texts, store.embed_dense)
    best_each = [usable[int(j)]["page"] for j in np.argmax(sim, axis=1)]
    dense = slides_mod._z(store.embed_dense(seg_texts) @ store.embed_dense(slide_texts).T)
    dense_path = [usable[j]["page"] for j in slides_mod.monotonic_path(dense)]

    rows = {
        "transcript, path (app without visual)": score(path_t),
        "transcript, best slide per segment": score(best_each),
        "transcript, dense only, path": score(dense_path),
        "transcript + on-screen text, path (app)": score(path_v),
    }
    matched = sum(f["slide"] is not None for f in frames)
    print(f"Lecture {args.lecture}: {len(deck)} slides, {len(frames)} key frames ({matched} matched to a slide), "
          f"{len(segs)} segments ({len(scored)} with a gold slide)\n")
    print("| Alignment | Exact slide | Within 1 slide |\n|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['exact']:.2f} | {r['within_1']:.2f} |")

    out = ROOT / "eval" / "results" / f"slides-gold-lecture{args.lecture}.json"
    out.write_text(json.dumps({
        "frames": [{"file": f["file"], "start_sec": f["start_sec"], "end_sec": f["end_sec"], "slide": f["slide"],
                    "text": f["text"][:200]} for f in frames],
        "segments": [{"start_sec": s["start_sec"], "end_sec": s["end_sec"], "gold": s["gold"],
                      "transcript_path": a, "best_each": b, "with_visual": c}
                     for s, a, b, c in zip(segs, path_t, best_each, path_v)],
        "results": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
