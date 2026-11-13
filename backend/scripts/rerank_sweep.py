"""Compare reranking setups on the eval set, in-process (no server needed).

    cd backend
    uv run python scripts/rerank_sweep.py --course 1

For each question the hybrid candidates are fetched once, then every reranker model scores them.
Variants per model and candidate count N:
  rerank      order the N candidates by cross-encoder score (what the app does today)
  rrf-blend   reciprocal-rank fusion of the hybrid rank and the rerank rank
Metrics match eval/run.py (hit = gold lecture and overlapping time range). Latency is the
rerank step alone on this machine's CPU, median per question. "unans >= gate" counts uncovered
questions whose top score clears the lowest top score of any answerable question, i.e. the ones
a score threshold cannot reject without also rejecting a real question.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
]
CANDIDATES = [10, 20, 30]
K = 5
TOL = 30
RRF_K = 60


def is_hit(h: dict, q: dict) -> bool:
    return h["lecture_number"] == q["lecture"] and h["start_sec"] < q["gold_end_sec"] and h["end_sec"] > q["gold_start_sec"]


def metrics(ranked: list[list[dict]], answerable: list[dict]) -> dict:
    r1 = r5 = rr = ts = 0.0
    for hits, q in zip(ranked, answerable):
        rank = next((i for i, h in enumerate(hits[:K], 1) if is_hit(h, q)), None)
        r1 += rank == 1
        r5 += rank is not None
        rr += 1 / rank if rank else 0
        top = hits[0]
        ts += top["lecture_number"] == q["lecture"] and q["gold_start_sec"] - TOL <= top["start_sec"] <= q["gold_end_sec"] + TOL
    n = len(answerable)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": rr / n, "timestamp_acc": ts / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=int, default=1)
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    args = ap.parse_args()

    qs = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    answerable = [q for q in qs if q.get("type") != "unanswerable"]
    unanswerable = [q for q in qs if q.get("type") == "unanswerable"]
    n_max = max(CANDIDATES)
    cands = {q["id"]: store.search(args.course, q["question"], n_max, "hybrid") for q in qs}

    rows = [{"setup": "hybrid, no rerank", **metrics([cands[q["id"]] for q in answerable], answerable)}]
    for name in MODELS:
        model = TextCrossEncoder(name)
        list(model.rerank("warm up", ["warm up"]))
        for n in CANDIDATES:
            ranked, blended, top_ok, top_un, lat = [], [], [], [], []
            for q in qs:
                hits = cands[q["id"]][:n]
                t = time.perf_counter()
                scores = [1 / (1 + math.exp(-s)) for s in model.rerank(q["question"], [h["text"] for h in hits])]
                lat.append((time.perf_counter() - t) * 1000)
                order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
                rerank_rank = {i: r for r, i in enumerate(order, 1)}
                fused = sorted(range(len(hits)), key=lambda i: 1 / (RRF_K + i + 1) + 1 / (RRF_K + rerank_rank[i]), reverse=True)
                if q.get("type") == "unanswerable":
                    top_un.append(scores[order[0]])
                else:
                    ranked.append([hits[i] for i in order])
                    blended.append([hits[i] for i in fused])
                    top_ok.append(scores[order[0]])
            gate = min(top_ok)
            extra = {
                "rerank ms": round(statistics.median(lat)),
                "unans >= gate": f"{sum(s >= gate for s in top_un)}/{len(top_un)}",
                "min answerable": round(gate, 3),
            }
            rows.append({"setup": f"{name.split('/')[1]}, N={n}, rerank", **metrics(ranked, answerable), **extra})
            rows.append({"setup": f"{name.split('/')[1]}, N={n}, rrf-blend", **metrics(blended, answerable), **extra})
            print(f"done {name} N={n}", flush=True)

    cols = ["setup", "recall@1", "recall@5", "mrr", "timestamp_acc", "rerank ms", "unans >= gate", "min answerable"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    table = "\n".join(lines)
    print(f"\n{len(answerable)} answerable, {len(unanswerable)} unanswerable\n\n{table}")
    out = ROOT / "eval" / "results" / f"rerank-sweep-{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
