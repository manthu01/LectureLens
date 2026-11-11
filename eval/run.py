"""Retrieval evaluation. Needs the backend running; uses only the standard library.

    python eval/run.py --course 1
    python eval/run.py --course 1 --questions eval/questions.jsonl --api http://localhost:8000

Each line of the questions file is JSON:
    {"id": "q1", "question": "...", "lecture": 1, "gold_start_sec": 283, "gold_end_sec": 344,
     "gold_answer": "...", "type": "factual"}
`lecture` is the lecture number shown in the UI. Questions with "type": "unanswerable" have no
gold span; they are left out of retrieval metrics and used to report rerank scores for tuning
RERANK_THRESHOLD.

A retrieved segment is a hit when it is from the gold lecture and overlaps the gold time range.
"""

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path

CONFIGS = [
    {"name": "dense", "mode": "dense", "rerank": False},
    {"name": "bm25", "mode": "bm25", "rerank": False},
    {"name": "hybrid", "mode": "hybrid", "rerank": False},
    {"name": "hybrid+rerank", "mode": "hybrid", "rerank": True},
    {"name": "hybrid+rerank, off-topic kept", "mode": "hybrid", "rerank": True, "include_off_topic": True},
]
K = 5
TIMESTAMP_TOLERANCE_SEC = 30


def search(api: str, course: int, question: str, cfg: dict) -> list[dict]:
    body = {"question": question, "k": K, **{k: v for k, v in cfg.items() if k != "name"}}
    req = urllib.request.Request(
        f"{api}/courses/{course}/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read())


def is_hit(hit: dict, q: dict) -> bool:
    return (
        hit["lecture_number"] == q["lecture"]
        and hit["start_sec"] < q["gold_end_sec"]
        and hit["end_sec"] > q["gold_start_sec"]
    )


def evaluate(api: str, course: int, questions: list[dict], cfg: dict) -> dict:
    answerable = [q for q in questions if q.get("type") != "unanswerable"]
    unanswerable = [q for q in questions if q.get("type") == "unanswerable"]
    r1 = r5 = rr = ts = 0.0
    latencies, top_ok, top_un, misses = [], [], [], []
    for q in answerable:
        t = time.perf_counter()
        hits = search(api, course, q["question"], cfg)
        latencies.append((time.perf_counter() - t) * 1000)
        rank = next((i for i, h in enumerate(hits, 1) if is_hit(h, q)), None)
        r1 += rank == 1
        r5 += rank is not None
        rr += 1 / rank if rank else 0
        if hits:
            top = hits[0]
            ts += (
                top["lecture_number"] == q["lecture"]
                and q["gold_start_sec"] - TIMESTAMP_TOLERANCE_SEC
                <= top["start_sec"]
                <= q["gold_end_sec"] + TIMESTAMP_TOLERANCE_SEC
            )
            if "rerank_score" in top:
                top_ok.append(top["rerank_score"])
        if rank is None:
            misses.append(q["id"])
    for q in unanswerable:
        hits = search(api, course, q["question"], cfg)
        if hits and "rerank_score" in hits[0]:
            top_un.append(hits[0]["rerank_score"])
    n = max(len(answerable), 1)
    return {
        "config": cfg["name"],
        "n": len(answerable),
        "recall@1": r1 / n,
        f"recall@{K}": r5 / n,
        "mrr": rr / n,
        "timestamp_acc": ts / n,
        "latency_ms_median": statistics.median(latencies) if latencies else 0,
        "misses": misses,
        "top_rerank_answerable": sorted(round(s, 3) for s in top_ok),
        "top_rerank_unanswerable": sorted(round(s, 3) for s in top_un),
    }


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=int, required=True)
    ap.add_argument("--questions", default=str(here / "questions.jsonl"))
    ap.add_argument("--api", default="http://127.0.0.1:8000")  # "localhost" adds ~2s per request on Windows
    args = ap.parse_args()

    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(l) for l in lines if l.strip() and not l.lstrip().startswith("//")]
    results = [evaluate(args.api, args.course, questions, cfg) for cfg in CONFIGS]

    cols = ["recall@1", f"recall@{K}", "mrr", "timestamp_acc"]
    table = [
        f"| Config | {' | '.join(cols)} | median ms |",
        f"|---|{'---|' * (len(cols) + 1)}",
        *(
            f"| {r['config']} | {' | '.join(f'{r[c]:.2f}' for c in cols)} | {r['latency_ms_median']:.0f} |"
            for r in results
        ),
    ]
    n_un = sum(q.get("type") == "unanswerable" for q in questions)
    print(f"{results[0]['n']} answerable questions, {n_un} unanswerable\n")
    print("\n".join(table))
    for r in results:
        if r["misses"]:
            print(f"\n{r['config']}: no hit in top {K} for {', '.join(r['misses'])}")
    rr = next(r for r in results if r["config"] == "hybrid+rerank")
    print("\nTop rerank score per question (for choosing RERANK_THRESHOLD):")
    print("  answerable:  ", rr["top_rerank_answerable"])
    print("  unanswerable:", rr["top_rerank_unanswerable"])

    out_dir = here / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / f"{stamp}.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"\nSaved to {out_dir / stamp}.json and .md")


if __name__ == "__main__":
    main()
