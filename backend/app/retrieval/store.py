"""Embedding + Qdrant: indexing, dense/BM25/hybrid search, segment lookup."""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings

# v2: named vectors ("dense" + sparse "bm25"). Bump the suffix whenever the schema changes;
# startup re-indexes lectures from their cached transcripts when the collection is missing.
COLLECTION = "segments_v2"
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
    client = _client()
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("course_id", "lecture_id"):
        client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.INTEGER)
    client.create_payload_index(COLLECTION, "label", models.PayloadSchemaType.KEYWORD)


def delete_lecture(lecture_id: int) -> None:
    if collection_exists():
        _client().delete(
            COLLECTION,
            points_selector=models.FilterSelector(filter=models.Filter(must=[_is("lecture_id", lecture_id)])),
        )


def index_lecture(lecture, chunks: list[dict]) -> int:
    """Replace all segments of a lecture in the index. Chunks may carry `label` and `topic`."""
    if not chunks:
        return 0
    # The topic is embedded with the text so short topic queries match; payload keeps raw text.
    texts = [f"{c['topic']}. {c['text']}" if c.get("topic") else c["text"] for c in chunks]
    dense = [v.tolist() for v in _dense().embed(texts)]
    sparse = [_sv(e) for e in _sparse().embed(texts)]
    _ensure_collection(len(dense[0]))
    delete_lecture(lecture.id)
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lecture/{lecture.id}/{i}")),
            vector={"dense": d, "bm25": s},
            payload={
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
            },
        )
        for i, (c, d, s) in enumerate(zip(chunks, dense, sparse))
    ]
    for i in range(0, len(points), 128):
        _client().upsert(COLLECTION, points=points[i : i + 128])
    return len(points)


def search(
    course_id: int, query: str, k: int, mode: str = "hybrid", include_off_topic: bool = False
) -> list[dict]:
    """Top-k segments of a course. Returns payload dicts with `id` and `score` added."""
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
    return _scroll(models.Filter(must=[_is("lecture_id", lecture_id)]))


def course_segments_by_label(course_id: int, label: str) -> list[dict]:
    return _scroll(models.Filter(must=[_is("course_id", course_id), _is("label", label)]))
