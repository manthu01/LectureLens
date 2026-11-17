"""Retrieval ablations, in-process, without touching the app's index (no server, no LLM calls).

    backend/.venv/Scripts/python backend/scripts/ablate.py --course 1
    backend/.venv/Scripts/python backend/scripts/ablate.py --course 1 --only chunking

Each setup rebuilds a throwaway in-memory Qdrant index from the cached transcripts in
data/courses/<course>/<lecture>/transcript.json, using the app's own chunking, indexing and search
code (store.index_lecture / store.search), then scores the eval questions the same way as
eval/run.py: a hit is a segment from the gold lecture that overlaps the gold time range.

Ablations (one factor at a time; everything else stays at the app's settings):
  chunking    target chunk length 30/60/90/120 s. Classifier off for all four, because the cached
              labels only line up with the default 60 s chunks.
  embedding   dense model, with the app's chunking and cached classifier labels.
  classifier  the cached segment labels used for nothing, for the off-topic filter only, for the
              topic prefix only, or for both (the app).
  whisper     transcripts from other Whisper sizes (make them with scripts/transcribe_variant.py),
              classifier off. The gold time ranges come from the audio, so they hold for any transcript.

Caveat for chunking: a longer chunk overlaps a gold span more easily, so recall rises with size
by construction. Timestamp accuracy (the top hit starts within 30 s of the gold span) is the
counterweight: it's where the student lands when they click.
"""

import argparse
import json
import sqlite3
import statistics
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from fastembed import TextEmbedding  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.ingestion.chunk import chunk_transcript  # noqa: E402
from app.retrieval import store  # noqa: E402

K = 5
TOL = 30
CHUNKINGS = {  # name -> chunk_transcript kwargs; 60 s is the app default
    "30 s": dict(target_sec=30, min_sec=20, max_sec=45),
    "60 s (app)": dict(target_sec=60, min_sec=45, max_sec=90),
    "90 s": dict(target_sec=90, min_sec=70, max_sec=135),
    "120 s": dict(target_sec=120, min_sec=90, max_sec=180),
}
EMBEDDERS = [
    "BAAI/bge-small-en-v1.5",  # app default
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "thenlper/gte-base",
    "jinaai/jina-embeddings-v2-small-en",
    "ibm-granite/granite-embedding-small-english-r2",
]
CLASSIFIER = {  # name -> (keep off-topic label for filtering, prepend topic to embedded text)
    "off": (False, False),
    "filter only": (True, False),
    "topic prefix only": (False, True),
    "filter + topic (app)": (True, True),
}


@lru_cache
def embedder(name: str) -> TextEmbedding:
    return TextEmbedding(name)


WHISPER = {  # name -> transcript file per lecture; variants come from scripts/transcribe_variant.py
    "small (app)": "transcript.json",
    "medium": "transcript-medium.json",
    "large-v3": "transcript-large-v3.json",
}


def load_lectures(course: int, transcript: str = "transcript.json") -> list[dict]:
    db = sqlite3.connect(settings.data_path / "lecturelens.db")
    rows = db.execute(
        "select id, number, title, video_id from lecture where course_id = ? and status = 'indexed' order by number",
        (course,),
    ).fetchall()
    out = []
    for id_, number, title, video_id in rows:
        d = settings.data_path / "courses" / str(course) / str(id_)
        labels_path = d / "labels.json"
        out.append(
            {
                "meta": SimpleNamespace(id=id_, course_id=course, number=number, title=title, video_id=video_id),
                "segs": json.loads((d / transcript).read_text(encoding="utf-8")),
                "labels": json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else None,
            }
        )
    return out


def build(lectures: list[dict], model: str, chunking: dict, use_filter: bool, use_topic: bool) -> dict:
    """Index everything into a fresh in-memory collection; returns chunk stats and build time."""
    client = QdrantClient(":memory:")
    store._client = lambda: client  # the app's store functions now read and write this index
    store._dense = lambda: embedder(model)
    t = time.perf_counter()
    durations = []
    for lec in lectures:
        chunks = chunk_transcript(lec["segs"], **chunking)
        if use_filter or use_topic:
            labels = lec["labels"]
            if labels is None or len(labels) != len(chunks):
                raise RuntimeError(f"cached labels don't match the chunks of lecture {lec['meta'].number}")
            chunks = [
                {
                    **c,
                    "label": lab["label"] if use_filter else "teaching",
                    "topic": lab.get("topic", "") if use_topic else "",
                }
                for c, lab in zip(chunks, labels)
            ]
        durations += [c["end_sec"] - c["start_sec"] for c in chunks]
        store.index_lecture(lec["meta"], chunks)
    return {
        "chunks": len(durations),
        "mean_chunk_sec": statistics.mean(durations),
        "index_sec": time.perf_counter() - t,
    }


def is_hit(h: dict, q: dict) -> bool:
    return h["lecture_number"] == q["lecture"] and h["start_sec"] < q["gold_end_sec"] and h["end_sec"] > q["gold_start_sec"]


def score(course: int, questions: list[dict], mode: str) -> dict:
    r1 = r5 = rr = ts = 0.0
    lat = []
    for q in questions:
        t = time.perf_counter()
        hits = store.search(course, q["question"], K, mode)
        lat.append((time.perf_counter() - t) * 1000)
        rank = next((i for i, h in enumerate(hits, 1) if is_hit(h, q)), None)
        r1 += rank == 1
        r5 += rank is not None
        rr += 1 / rank if rank else 0
        if hits:
            top = hits[0]
            ts += top["lecture_number"] == q["lecture"] and q["gold_start_sec"] - TOL <= top["start_sec"] <= q["gold_end_sec"] + TOL
    n = len(questions)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": rr / n, "timestamp_acc": ts / n, "query_ms": statistics.median(lat)}


def run(course, lectures, questions, name, model, chunking, use_filter, use_topic, modes) -> dict:
    stats = build(lectures, model, chunking, use_filter, use_topic)
    res = {"setup": name, **stats}
    for mode in modes:
        res[mode] = score(course, questions, mode)
    line = " | ".join(f"{m} R@1 {res[m]['recall@1']:.2f} TS {res[m]['timestamp_acc']:.2f}" for m in modes)
    print(f"  {name}: {stats['chunks']} chunks, {line}", flush=True)
    return res


def table(rows: list[dict], mode: str, extra: list[tuple[str, str]]) -> list[str]:
    head = ["Setup", *(h for h, _ in extra), "Recall@1", f"Recall@{K}", "MRR", f"Timestamp ±{TOL}s"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        m = r[mode]
        cells = [r["setup"], *(fmt.format(**r) for _, fmt in extra)]
        cells += [f"{m['recall@1']:.2f}", f"{m[f'recall@{K}']:.2f}", f"{m['mrr']:.2f}", f"{m['timestamp_acc']:.2f}"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=int, default=1)
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--only", choices=["chunking", "embedding", "classifier", "whisper"])
    args = ap.parse_args()

    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    qs = [json.loads(l) for l in lines if l.strip() and not l.lstrip().startswith("//")]
    answerable = [q for q in qs if q.get("type") != "unanswerable"]
    lectures = load_lectures(args.course)
    app_chunking = CHUNKINGS["60 s (app)"]
    results, md = {}, [f"# Retrieval ablations ({len(answerable)} answerable questions, {len(lectures)} lectures)", ""]

    if args.only in (None, "chunking"):
        print("Chunking:")
        rows = [
            run(args.course, lectures, answerable, name, settings.embedding_model, ck, False, False, ["hybrid"])
            for name, ck in CHUNKINGS.items()
        ]
        results["chunking"] = rows
        md += ["## Chunk length (hybrid, classifier off)", ""]
        md += table(rows, "hybrid", [("Chunks", "{chunks}"), ("Mean length", "{mean_chunk_sec:.0f} s")]) + [""]

    if args.only in (None, "embedding"):
        print("Embedding:")
        rows = [
            run(args.course, lectures, answerable, m, m, app_chunking, True, True, ["dense", "hybrid"])
            for m in EMBEDDERS
        ]
        for r in rows:
            r["dense_ms"] = r["dense"]["query_ms"]
        results["embedding"] = rows
        md += ["## Embedding model, dense only (60 s chunks, classifier on)", ""]
        md += table(rows, "dense", [("Index time", "{index_sec:.0f} s"), ("Query", "{dense_ms:.0f} ms")]) + [""]
        md += ["## Embedding model, hybrid with BM25", ""] + table(rows, "hybrid", []) + [""]

    if args.only in (None, "classifier"):
        print("Classifier:")
        rows = [
            run(args.course, lectures, answerable, name, settings.embedding_model, app_chunking, f, t, ["dense", "hybrid"])
            for name, (f, t) in CLASSIFIER.items()
        ]
        results["classifier"] = rows
        md += ["## Segment classifier (60 s chunks, bge-small)", "", "Hybrid:", ""] + table(rows, "hybrid", [])
        md += ["", "Dense only:", ""] + table(rows, "dense", []) + [""]

    if args.only in (None, "whisper"):
        print("Whisper:")
        rows = []
        for name, fname in WHISPER.items():
            course_dir = settings.data_path / "courses" / str(args.course)
            if not all((course_dir / str(lec["meta"].id) / fname).exists() for lec in lectures):
                print(f"  {name}: no {fname} for every lecture, skipped")
                continue
            # Classifier off: the cached labels only line up with the app's own transcript.
            rows.append(
                run(args.course, load_lectures(args.course, fname), answerable, name,
                    settings.embedding_model, app_chunking, False, False, ["dense", "hybrid"])
            )
        if len(rows) > 1:
            results["whisper"] = rows
            md += ["## Whisper model (60 s chunks, bge-small, classifier off)", "", "Hybrid:", ""]
            md += table(rows, "hybrid", [("Chunks", "{chunks}")]) + ["", "Dense only:", ""] + table(rows, "dense", []) + [""]

    out = ROOT / "eval" / "results" / f"ablations-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"Saved {out.name}.json and .md")


if __name__ == "__main__":
    main()
