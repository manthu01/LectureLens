"""Uploaded notes: passage splitting, indexing next to the transcript, prompts and citations.

The index tests use an in-memory Qdrant and a tiny bag-of-words embedder, so they need no
network and no model download."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.generation.answer import _citation, _moment, build_prompt
from app.ingestion import notes
from app.retrieval import store

VOCAB = ["gradient", "descent", "learning", "rate", "batch", "online", "drift", "overfitting", "notes"]


def _bow(text: str) -> np.ndarray:
    v = np.array([text.lower().count(w) for w in VOCAB], dtype=float) + 1e-3
    return v / np.linalg.norm(v)


class _Dense:
    def embed(self, texts):
        return (_bow(t) for t in texts)

    def query_embed(self, q):
        return iter([_bow(q)])


class _Sparse:
    @staticmethod
    def _one(text):
        counts = {}
        for w in re.findall(r"[a-z]+", text.lower()):
            i = VOCAB.index(w) if w in VOCAB else None
            if i is not None:
                counts[i] = counts.get(i, 0) + 1
        idx = sorted(counts) or [0]
        return SimpleNamespace(indices=np.array(idx), values=np.array([float(counts.get(i, 0.01)) for i in idx]))

    def embed(self, texts):
        return (self._one(t) for t in texts)

    def query_embed(self, q):
        return iter([self._one(q)])


@pytest.fixture
def index(monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(store, "_client", lambda: client)
    monkeypatch.setattr(store, "_dense", lambda: _Dense())
    monkeypatch.setattr(store, "_sparse", lambda: _Sparse())
    return client


LECTURE = SimpleNamespace(id=7, course_id=1, number=4, title="Optimisation", video_id="abc")
CHUNKS = [
    {"start_sec": 0, "end_sec": 60, "text": "today batch learning", "label": "teaching"},
    {"start_sec": 60, "end_sec": 120, "text": "online learning and drift", "label": "teaching"},
]
PASSAGES = [
    {"page": 1, "topic": "Gradient descent", "text": "gradient descent with a learning rate"},
    {"page": 2, "topic": "", "text": "overfitting notes"},
]


# --- splitting ---------------------------------------------------------------------------------


def _para(word: str, n: int) -> str:
    return " ".join([word] * (n - 1)) + "."


def test_split_text_follows_headings_and_paragraphs():
    md = "\n".join([
        "# Gradient descent", "", _para("alpha", 100), "", _para("beta", 100), "",
        "## Learning rate", "", _para("gamma", 60), "",
        "## Tiny", "", "Just a line.", "",
        "## Next", "", _para("delta", 50),
    ])
    out = notes.split_text(md)
    assert [p["page"] for p in out] == list(range(1, len(out) + 1))
    # 100 + 100 words reach the target, so the first section is one passage; the heading starts a new one.
    assert out[0]["topic"] == "Gradient descent" and "alpha" in out[0]["text"] and "beta" in out[0]["text"]
    assert out[1]["topic"] == "Learning rate" and out[1]["text"].startswith("gamma")
    # "Learning rate" (60 words) is long enough to close at the next heading; "Tiny" (3 words) is
    # not, so the "Next" section joins it instead of leaving a 3-word passage.
    assert out[2]["topic"] == "Tiny" and "delta" in out[2]["text"]
    assert all(len(p["text"].split()) <= notes.MAX_WORDS for p in out)


def test_split_text_splits_a_long_paragraph_at_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))  # 1000 words
    out = notes.split_text(text)
    assert len(out) >= 4
    assert all(len(p["text"].split()) <= notes.MAX_WORDS for p in out)
    assert all(p["text"].endswith(".") for p in out)
    assert " ".join(p["text"] for p in out).split() == text.split()


def test_split_long_cuts_runs_without_punctuation():
    pieces = notes.split_long("word " * 600)
    assert sum(len(p.split()) for p in pieces) == 600
    assert all(len(p.split()) <= notes.MAX_WORDS for p in pieces)


def test_split_pages_keeps_page_numbers_and_skips_empty_pages():
    out = notes.split_pages(["Intro to gradient descent.", "   ", "word " * 400])
    assert out[0] == {"page": 1, "topic": "", "text": "Intro to gradient descent."}
    assert {p["page"] for p in out[1:]} == {3} and len(out) >= 3


def _pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_save_pdf_and_markdown(tmp_path):
    passages = notes.save(tmp_path, "Week4.PDF", _pdf(["Gradient descent basics", "Learning rate schedules"]))
    assert [(p["page"], p["text"]) for p in passages] == [(1, "Gradient descent basics"), (2, "Learning rate schedules")]
    assert notes.file_path(tmp_path).name == "notes.pdf"
    assert notes.load(tmp_path) == passages
    # Re-upload as Markdown replaces the PDF.
    notes.save(tmp_path, "notes.md", b"# Title\n\nSome text about drift.")
    assert notes.file_path(tmp_path).name == "notes.md"
    assert notes.load(tmp_path)[0]["topic"] == "Title"


def test_save_rejects_bad_files_and_keeps_old_notes(tmp_path):
    notes.save(tmp_path, "a.txt", b"Old notes text.")
    with pytest.raises(ValueError):
        notes.save(tmp_path, "a.docx", b"x")
    with pytest.raises(ValueError):
        notes.save(tmp_path, "a.md", b"   \n\n  ")
    assert notes.load(tmp_path)[0]["text"] == "Old notes text."
    assert notes.file_path(tmp_path).name == "notes.txt"
    assert not list(tmp_path.glob("upload-*"))


# --- index -------------------------------------------------------------------------------------


def test_reindexing_the_transcript_keeps_notes(index):
    store.index_lecture(LECTURE, CHUNKS)
    assert store.index_notes(LECTURE, PASSAGES) == 2
    store.index_lecture(LECTURE, CHUNKS[:1])  # e.g. slides re-linked, or a retry
    hits = store.search(1, "gradient descent learning rate", k=10)
    sources = sorted(h.get("source") for h in hits)
    assert sources == ["notes", "notes", "transcript"]


def test_reuploading_notes_replaces_only_notes(index):
    store.index_lecture(LECTURE, CHUNKS)
    store.index_notes(LECTURE, PASSAGES)
    store.index_notes(LECTURE, PASSAGES[:1])
    hits = store.search(1, "learning", k=10)
    assert sum(h["source"] == "notes" for h in hits) == 1
    assert sum(h["source"] == "transcript" for h in hits) == 2


def test_lecture_segments_excludes_notes(index):
    store.index_lecture(LECTURE, CHUNKS)
    store.index_notes(LECTURE, PASSAGES)
    segs = store.lecture_segments(LECTURE.id)
    assert [s["start_sec"] for s in segs] == [0, 60]
    assert all(s.get("source") != "notes" for s in segs)
    assert store.course_segments_by_label(1, "teaching") == segs


def test_transcript_points_without_source_still_count_as_transcript(index):
    """Points indexed before notes existed have no `source` field."""
    store.index_lecture(LECTURE, CHUNKS)
    pts, _ = index.scroll(store.COLLECTION, limit=10, with_payload=True)
    index.overwrite_payload(store.COLLECTION, payload={k: v for k, v in pts[0].payload.items() if k != "source"},
                            points=[pts[0].id])
    store.index_notes(LECTURE, PASSAGES)
    assert len(store.lecture_segments(LECTURE.id)) == 2
    store.index_lecture(LECTURE, CHUNKS)  # deletes the source-less point too, keeps notes
    assert len(store.lecture_segments(LECTURE.id)) == 2
    assert sum(h["source"] == "notes" for h in store.search(1, "notes", k=10)) == 2


def test_search_returns_notes_hits(index):
    store.index_lecture(LECTURE, CHUNKS)
    store.index_notes(LECTURE, PASSAGES)
    for mode in store.MODES:
        top = store.search(1, "gradient descent", k=3, mode=mode)[0]
        assert top["source"] == "notes" and top["page"] == 1 and top["start_sec"] is None
        assert top["lecture_number"] == 4 and top["label"] == "teaching"


def test_rebuilt_index_gets_notes_back_from_disk(index, tmp_path):
    """After a schema change the collection is rebuilt by re-ingesting; notes.json is re-indexed."""
    from app.ingestion.pipeline import reindex_notes

    notes.save(tmp_path, "n.md", b"# Gradient descent\n\nGradient descent and the learning rate.")
    assert not index.collection_exists(store.COLLECTION)  # a fresh index, as after a schema change
    store.index_lecture(LECTURE, CHUNKS)
    reindex_notes(LECTURE, tmp_path)
    assert store.search(1, "gradient descent", k=3)[0]["source"] == "notes"


def test_notes_ids_dont_collide_with_transcript_ids(index):
    store.index_lecture(LECTURE, CHUNKS)
    store.index_notes(LECTURE, PASSAGES)
    pts, _ = index.scroll(store.COLLECTION, limit=10)
    assert len({p.id for p in pts}) == 4


# --- answers -----------------------------------------------------------------------------------

NOTES_HIT = {
    "id": "n1", "source": "notes", "page": 2, "lecture_id": 7, "lecture_number": 4,
    "lecture_title": "Optimisation", "video_id": "abc", "start_sec": None, "end_sec": None,
    "text": "overfitting notes",
}
SEG_HIT = {
    "id": "s1", "lecture_id": 7, "lecture_number": 4, "lecture_title": "Optimisation", "video_id": "abc",
    "start_sec": 65, "end_sec": 125, "text": "online learning",
}


def test_build_prompt_labels_notes_by_page():
    prompt = build_prompt("q?", [SEG_HIT, NOTES_HIT])
    assert "[S1] Lecture 4: Optimisation (1:05-2:05)" in prompt
    assert "[S2] Lecture 4 notes, page 2 (Optimisation)\noverfitting notes" in prompt


def test_citation_carries_source_and_page():
    c = _citation(2, NOTES_HIT)
    assert c["source"] == "notes" and c["page"] == 2 and c["start_sec"] is None
    s = _citation(1, SEG_HIT)  # no `source` in the payload: an old transcript point
    assert s["source"] == "transcript" and s["page"] is None
    assert _moment(NOTES_HIT) != _moment({**NOTES_HIT, "page": 3}) and _moment(SEG_HIT) == (7, 65)


# --- endpoints ---------------------------------------------------------------------------------


@pytest.fixture
def api(index, tmp_path, monkeypatch):
    """The API against a temp data dir, an in-memory SQLite and the in-memory index.
    No lifespan: init_db would touch the real database."""
    from fastapi.testclient import TestClient
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine

    from app.config import settings
    from app.db import Course, Lecture, get_session
    from app.main import app

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "read_only", False)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(Course(id=1, name="ML"))
        s.add(Lecture(id=7, course_id=1, number=4, title="Optimisation", source_url="x", status="indexed"))
        s.commit()

    def session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_notes_endpoints(api, monkeypatch):
    from app.config import settings

    body = b"# Gradient descent\n\nGradient descent needs a learning rate.\n"
    r = api.post("/lectures/7/notes", params={"filename": "week4.md"}, content=body)
    assert r.status_code == 200 and r.json()["notes_name"] == "week4.md"
    assert api.get("/lectures/7/notes").json() == [
        {"page": 1, "topic": "Gradient descent", "text": "Gradient descent needs a learning rate."}
    ]
    f = api.get("/lectures/7/notes/file")
    assert f.status_code == 200 and f.content == body and f.headers["content-type"].startswith("text/markdown")
    hits = store.search(1, "gradient descent", k=5)
    assert hits and hits[0]["source"] == "notes"
    assert api.get("/lectures/7/transcript").json() == []  # notes are not transcript segments

    assert api.post("/lectures/7/notes", params={"filename": "a.docx"}, content=b"x").status_code == 400
    assert api.post("/lectures/7/notes", params={"filename": "a.md"}, content=b"").status_code == 400
    assert api.get("/lectures/99/notes/file").status_code == 404

    monkeypatch.setattr(settings, "read_only", True)
    assert api.post("/lectures/7/notes", params={"filename": "b.md"}, content=body).status_code == 403
