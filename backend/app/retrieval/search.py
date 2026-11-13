"""Query flow: retrieval (dense / BM25 / hybrid RRF) -> optional cross-encoder relevance score.

The cross-encoder does not reorder results: on the eval set, re-sorting by its score lowered
recall@1 (0.96 -> 0.91, see eval/results/rerank-sweep-*.md). Its score is used only as the
confidence signal for refusing questions the lectures don't cover.
"""

import math
from functools import lru_cache

from app.config import settings
from app.retrieval import store


@lru_cache
def _reranker(model: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model)


def retrieve(
    course_id: int,
    question: str,
    *,
    mode: str | None = None,
    rerank: bool | None = None,
    k: int | None = None,
    include_off_topic: bool = False,
) -> list[dict]:
    """Top-k segments for a question. Arguments left as None fall back to .env settings.

    When scoring, each returned hit gets `rerank_score` in 0..1 (sigmoid of the cross-encoder
    logit); the order stays as retrieved.
    """
    mode = mode or settings.retrieval_mode
    k = k or settings.top_k
    rerank = bool(settings.reranker_model) if rerank is None else rerank and bool(settings.reranker_model)
    hits = store.search(course_id, question, k, mode, include_off_topic)
    if not rerank or not hits:
        return hits
    logits = _reranker(settings.reranker_model).rerank(question, [h["text"] for h in hits])
    for h, logit in zip(hits, logits):
        h["rerank_score"] = 1 / (1 + math.exp(-logit))
    return hits
