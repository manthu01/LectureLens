"""End-to-end ingest of a short clip: local file -> Whisper -> chunks -> index -> search.

Slow (runs Whisper on CPU, ~1 min) and needs a cached lecture audio file to cut the clip from, so
it only runs when asked:  LECTURELENS_SLOW=1 uv run pytest tests/test_ingest_e2e.py
No network: segment labels and screen reading are switched off. Runs in a subprocess with its own
empty data directory, so the real database and index are never touched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "courses" / "1" / "4" / "audio.m4a"  # lecture 4: batch vs online learning

pytestmark = pytest.mark.skipif(
    os.environ.get("LECTURELENS_SLOW") != "1" or not SOURCE.exists(),
    reason="slow end-to-end test; set LECTURELENS_SLOW=1 (needs data/courses/1/4/audio.m4a)",
)


def cut(src: Path, dst: Path, start: float, seconds: float) -> None:
    """Copy `seconds` of audio from `start` into dst without re-encoding."""
    import av

    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        a = inp.streams.audio[0]
        o = out.add_stream_from_template(a)
        inp.seek(int(start / a.time_base), stream=a)
        first = None
        for packet in inp.demux(a):
            if packet.dts is None:
                continue
            t = float(packet.pts * a.time_base)
            if t < start:
                continue
            if t > start + seconds:
                break
            first = first if first is not None else packet.pts
            packet.pts -= first
            packet.dts -= first
            packet.stream = o
            out.mux(packet)


SCRIPT = """
import json, sys
from sqlmodel import Session
from app.db import Course, Lecture, engine, init_db
from app.ingestion.pipeline import run_ingest
from app.retrieval import store
init_db()
with Session(engine) as s:
    c = Course(name="test"); s.add(c); s.commit(); s.refresh(c)
    l = Lecture(course_id=c.id, number=1, source_url=sys.argv[1]); s.add(l); s.commit(); s.refresh(l)
    cid, lid = c.id, l.id
run_ingest(lid)
with Session(engine) as s:
    l = s.get(Lecture, lid)
    hits = store.search(cid, "disadvantages of batch learning", 3, "hybrid")
    print(json.dumps({"status": l.status, "error": l.error, "segments": len(store.lecture_segments(lid)),
                      "hits": [h["text"][:200] for h in hits]}))
"""


def test_ingest_short_clip_end_to_end(tmp_path):
    clip = tmp_path / "clip.m4a"
    cut(SOURCE, clip, start=455, seconds=90)  # the "disadvantages of batch ML" part
    env = {
        **os.environ,
        "DATA_DIR": str(tmp_path / "data"),
        "DATABASE_URL": "",
        "QDRANT_URL": "",
        "CLASSIFY_SEGMENTS": "false",
        "VISUAL_PIPELINE": "false",
    }
    res = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(clip)],
        cwd=ROOT / "backend", env=env, capture_output=True, text=True, timeout=900,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["status"] == "indexed", out["error"]
    assert out["segments"] >= 1
    assert any("batch" in h.lower() for h in out["hits"]), out["hits"]
