"""Run the visual pipeline on lectures that are already indexed, then re-index them with the
on-screen text. New lectures get it during normal ingest; this is for the ones added before.

    backend/.venv/Scripts/python backend/scripts/add_visual.py --course 1
    backend/.venv/Scripts/python backend/scripts/add_visual.py --course 1 --lectures 4 5

Stop the API server first: the embedded Qdrant store allows one process at a time.
Resumable: video, key frames and frame reads are cached per lecture, so after a rate-limit error
just run it again. A lecture is re-indexed only once all its frames have been read.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from sqlmodel import Session, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Lecture, engine, init_db  # noqa: E402
from app.ingestion import visual  # noqa: E402
from app.ingestion.chunk import chunk_transcript  # noqa: E402
from app.ingestion.classify import classify_chunks  # noqa: E402
from app.ingestion.transcribe import transcribe  # noqa: E402
from app.retrieval import store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=int, required=True)
    ap.add_argument("--lectures", type=int, nargs="*", help="lecture numbers (default: all indexed)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_db()

    with Session(engine) as session:
        q = select(Lecture).where(Lecture.course_id == args.course, Lecture.status == "indexed")
        lectures = [l for l in session.exec(q.order_by(Lecture.number)) if not args.lectures or l.number in args.lectures]
        failed = 0
        for lec in lectures:
            work_dir = settings.data_path / "courses" / str(lec.course_id) / str(lec.id)
            # Same chunks as at ingest: cached transcript and cached labels, nothing re-generated.
            chunks = chunk_transcript(transcribe(work_dir / "audio.m4a", work_dir / "transcript.json"))
            if settings.classify_segments:
                chunks = classify_chunks(chunks, work_dir / "labels.json")
            try:
                chunks, layout, quality = visual.run(lec.source_url, work_dir, chunks)
            except Exception as e:
                print(f"lecture {lec.number}: stopped ({str(e)[:200]}); run again to resume")
                failed += 1
                break  # most failures are rate limits; later lectures would hit the same wall
            n = store.index_lecture(lec, chunks)
            lec.layout_type, lec.visual_quality_score = layout, quality
            session.add(lec)
            session.commit()
            with_text = sum(1 for c in chunks if c.get("visual_text"))
            print(f"lecture {lec.number}: {layout}, quality {quality if quality is None else round(quality, 2)}, {with_text}/{n} segments with on-screen text")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
