"""Embedding + Qdrant: indexing, dense/BM25/hybrid search, segment lookup.

Two kinds of points share the collection: transcript segments (no `source` field, or
`source: "transcript"`; they have start_sec/end_sec) and passages of uploaded notes
(`source: "notes"`, with `page` and start_sec/end_sec = None). Search returns both; lecture_segments
returns transcript segments only, since the transcript view, timeline, summary and quiz need times.
"""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings

# v2: named vectors ("dense" + sparse "bm25"). Bump the suffix whenever the schema changes;
# startup re-indexes lectures from their cached transcripts when the collection is missing.
COLLECTION = "segments_v2"
# Bumped on every index write, so caches keyed on it (answers) go stale when lectures change.
index_version = 0
MODES = ("dense", "bm25", "hybrid")


@lru_cache
def _dense():
    from fastembed import TextEmbedding

    return TextEmbedding(settings.embedding_model)


@lru_cache
def _sparse():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding("Qdrant/bm25")


@lru_cache
def _client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return QdrantClient(path=str(settings.data_path / "qdrant"))


def _sv(e) -> models.SparseVector:
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def _is(key: str, value) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def collection_exists() -> bool:
    return _client().collection_exists(COLLECTION)


def _ensure_collection(dim: int) -> None:
    if not _client().collection_exists(COLLECTION):
        create_collection(_client(), dim)


def create_collection(client: QdrantClient, dim: int) -> None:
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("course_id", "lecture_id"):
        client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.INTEGER)
    for field in ("label", "source"):
        client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)


NOTES = _is("source", "notes")


def _lecture_filter(lecture_id: int, source: str | None) -> models.Filter:
    """Points of a lecture: all of them (source None), its transcript segments, or its notes.
    Transcript points written before notes existed have no `source`, so "transcript" means "not notes"."""
    must = [_is("lecture_id", lecture_id)]
    if source == "notes":
        return models.Filter(must=must + [NOTES])
    if source == "transcript":
        return models.Filter(must=must, must_not=[NOTES])
    return models.Filter(must=must)


def delete_lecture(lecture_id: int, source: str | None = None) -> None:
    """Delete a lecture's points: all, or only "transcript" or "notes" ones."""
    global index_version
    index_version += 1
    if collection_exists():
        _client().delete(
            COLLECTION, points_selector=models.FilterSelector(filter=_lecture_filter(lecture_id, source))
        )


def embed_text(c: dict) -> str:
    """What gets embedded for a segment: topic (so short topic queries match), the transcript and
    the on-screen text and the aligned uploaded slide. The payload keeps them apart."""
    parts = [f"{c['topic']}." if c.get("topic") else "", c["text"]]
    if c.get("visual_text"):
        parts.append("On screen: " + c["visual_text"])
    if c.get("slide_text"):
        parts.append("Slide: " + c["slide_text"])
    return " ".join(p for p in parts if p)


def embed_dense(texts: list[str]):
    """Unit-length dense vectors (numpy array) with the app's embedding model."""
    import numpy as np

    v = np.asarray(list(_dense().embed(texts)), dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _vectors(texts: list[str]) -> list[dict]:
    dense = [v.tolist() for v in _dense().embed(texts)]
    sparse = [_sv(e) for e in _sparse().embed(texts)]
    _ensure_collection(len(dense[0]))
    return [{"dense": d, "bm25": s} for d, s in zip(dense, sparse)]


def _upsert(points: list[models.PointStruct]) -> None:
    for i in range(0, len(points), 128):
        _client().upsert(COLLECTION, points=points[i : i + 128])
    global index_version
    index_version += 1


def index_lecture(lecture, chunks: list[dict]) -> int:
    """Replace the transcript segments of a lecture in the index (its notes stay). Chunks may carry
    `label`, `topic`, `visual_text` (what was on screen) and `slide_page`/`slide_text` (aligned slide)."""
    if not chunks:
        return 0
    vectors = _vectors([embed_text(c) for c in chunks])
    delete_lecture(lecture.id, "transcript")
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lecture/{lecture.id}/{i}")),
            vector=v,
            payload={
                "source": "transcript",
                "course_id": lecture.course_id,
                "lecture_id": lecture.id,
                "lecture_number": lecture.number,
                "lecture_title": lecture.title,
                "video_id": lecture.video_id,
                "index": i,
                "start_sec": c["start_sec"],
                "end_sec": c["end_sec"],
                "text": c["text"],
                "label": c.get("label", "teaching"),
                "topic": c.get("topic", ""),
                "visual_text": c.get("visual_text", ""),
                "slide_page": c.get("slide_page"),
                "slide_text": c.get("slide_text", ""),
            },
        )
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    _upsert(points)
    return len(points)


def index_notes(lecture, passages: list[dict]) -> int:
    """Replace the notes passages of a lecture in the index (its transcript stays).
    Passages are [{page, topic, text}] from ingestion.notes. Ids are derived from
    "lecture/{id}/notes/{i}", so they never collide with transcript ids ("lecture/{id}/{i}")."""
    passages = [p for p in passages if p.get("text", "").strip()]
    if not passages:
        delete_lecture(lecture.id, "notes")
        return 0
    vectors = _vectors([embed_text(p) for p in passages])
    delete_lecture(lecture.id, "notes")
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lecture/{lecture.id}/notes/{i}")),
            vector=v,
            payload={
                "source": "notes",
                "course_id": lecture.course_id,
                "lecture_id": lecture.id,
                "lecture_number": lecture.number,
                "lecture_title": lecture.title,
                "video_id": lecture.video_id,
                "index": i,
                "page": p["page"],
                "start_sec": None,  # notes have no time in the video
                "end_sec": None,
                "text": p["text"],
                "label": "teaching",
                "topic": p.get("topic", ""),
            },
        )
        for i, (p, v) in enumerate(zip(passages, vectors))
    ]
    _upsert(points)
    return len(points)


def search(
    course_id: int, query: str, k: int, mode: str = "hybrid", include_off_topic: bool = False
) -> list[dict]:
    """Top-k points of a course (transcript segments and notes passages). Returns payload dicts
    with `id` and `score` added."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not collection_exists():
        return []
    flt = models.Filter(
        must=[_is("course_id", course_id)],
        must_not=[] if include_off_topic else [_is("label", "off_topic")],
    )
    client = _client()
    dense_q = next(iter(_dense().query_embed(query))).tolist() if mode != "bm25" else None
    sparse_q = _sv(next(iter(_sparse().query_embed(query)))) if mode != "dense" else None
    if mode == "hybrid":
        res = client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_q, using="dense", limit=max(k, 30), filter=flt),
                models.Prefetch(query=sparse_q, using="bm25", limit=max(k, 30), filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
    else:
        res = client.query_points(
            COLLECTION,
            query=dense_q if mode == "dense" else sparse_q,
            using=mode,
            query_filter=flt,
            limit=k,
            with_payload=True,
        )
    return [{**p.payload, "id": str(p.id), "score": p.score} for p in res.points]


def _scroll(flt: models.Filter) -> list[dict]:
    if not collection_exists():
        return []
    out, offset = [], None
    while True:
        points, offset = _client().scroll(COLLECTION, scroll_filter=flt, limit=512, offset=offset, with_payload=True)
        out += [{**p.payload, "id": str(p.id)} for p in points]
        if offset is None:
            break
    return sorted(out, key=lambda s: (s["lecture_number"], s["start_sec"]))


def lecture_segments(lecture_id: int) -> list[dict]:
    """Transcript segments of a lecture in time order (notes passages excluded)."""
    return _scroll(_lecture_filter(lecture_id, "transcript"))


def course_segments_by_label(course_id: int, label: str) -> list[dict]:
    """Transcript segments of a course with a label (exam hints, announcements)."""
    return _scroll(models.Filter(must=[_is("course_id", course_id), _is("label", label)], must_not=[NOTES]))
