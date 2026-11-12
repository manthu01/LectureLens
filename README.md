# LectureLens

**Ctrl+F for your entire semester of lectures.** Ask a question, get an answer grounded only in what was taught, and jump to the exact second the lecturer explained it.

> Status: work in progress. Search, cited answers, summaries, topic timelines and quizzes work. Reading slides or the board from the video is not built yet.

**Live demo:** https://lecturelens-app.vercel.app (read-only, over CampusX's "100 Days of Machine Learning" lectures; the first request after a quiet spell takes a few extra seconds while the backend starts).

## What it does

- **Ingests lectures** from a YouTube video, a whole playlist, or a local media file.
- **Transcribes with timestamps** using Whisper, running locally. Hindi/Hinglish lectures are translated to English during transcription.
- **Labels every segment** as teaching, off-topic, admin or exam hint, and names its topic.
- **Answers questions with citations.** Each claim links to a lecture and timestamp; clicking it seeks the video.
- **Refuses when the lectures don't cover it**, and shows the closest moments instead of guessing.
- **Summarises each lecture** with timestamped key points and a clickable topic timeline that greys out off-topic stretches.
- **Generates practice quizzes** from a lecture's own content, each question linked to the moment it was explained.

### Why not just ask a chatbot?

| General chatbot | LectureLens |
|---|---|
| Can't take in 40+ hours of video | Transcribes and indexes the whole course |
| Answers from general knowledge | Answers only from your lectures, or says "not covered" |
| No timestamps | Jumps to the exact moment in the video |
| No proof of accuracy | Measured with an evaluation set |

## How it works

```
Ingestion    YouTube / file ─► audio ─► Whisper ─► ~60s chunks ─► LLM labels ─► embeddings ─► Qdrant
                                                                                (dense + BM25)

Question     hybrid search (dense + BM25, RRF) ─► cross-encoder rerank ─► confidence gate
                 ─► Gemini answers from the top segments, citing [lecture · timestamp]
```

| Layer | Choice |
|---|---|
| Backend | FastAPI, SQLite (SQLModel) |
| Transcription | faster-whisper (`small`, CPU, translate mode) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, no PyTorch) |
| Vector store | Qdrant: dense + BM25 sparse vectors, fused with RRF. Embedded on disk locally, Qdrant Cloud when deployed |
| Reranker | `Xenova/ms-marco-MiniLM-L-12-v2` cross-encoder |
| LLM | Gemini (answers and segment labels), model set in `.env` |
| Frontend | Next.js 16, Tailwind |

Every model name and key is read from `.env`.

## Run it locally

Requirements: Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node 20+, and a Gemini API key. No Docker, GPU or ffmpeg needed.

```bash
# 1. Configure
cp .env.example .env          # then set GEMINI_API_KEY

# 2. Backend  (http://localhost:8000, API docs at /docs)
cd backend
uv sync
uv run uvicorn app.main:app

# 3. Frontend (http://localhost:3000)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000, create a course, and paste a YouTube video or playlist URL. Lectures are processed one at a time; on a laptop CPU, expect several minutes per lecture. If the backend restarts, unfinished lectures resume automatically.

## Deploying a public demo

Transcription is too heavy for free hosting, so the demo is **read-only**: lectures are processed locally, then a snapshot of the index is baked into the backend image. Visitors can search, ask, read summaries and take quizzes, but cannot add lectures.

```bash
cd backend
uv run python scripts/make_snapshot.py     # stop the backend first; writes backend/snapshot/
docker build -t lecturelens-api .
docker run -p 7860:7860 -e GEMINI_API_KEY=... -e LLM_MODEL=... lecturelens-api
```

The image sets `READ_ONLY=true` and limits each visitor to 10 LLM-backed requests per minute (`RATE_LIMIT_PER_MIN`). It uses about 700 MB of RAM, so it needs a host with at least 1 GB. Set `CORS_ORIGINS` on the backend to the frontend's URL, and `NEXT_PUBLIC_API_URL` on the frontend to the backend's URL.

The live demo runs both halves on Vercel. `scripts/build_space.py` writes `deploy/space/` with a `Dockerfile.vercel`, which Vercel builds as a container:

```bash
uv run python scripts/build_space.py
cd ../deploy/space && vercel link --project lecturelens-api && vercel deploy --prod
```

## Evaluation

`eval/questions.jsonl` holds questions with the lecture and time range where each is answered, plus questions the lectures do not cover.

```bash
python eval/run.py --course 1     # backend must be running
```

This compares dense, BM25, hybrid and hybrid + rerank retrieval and reports Recall@1, Recall@5, MRR and timestamp accuracy (top result within ±30s of the gold span). Results are saved to `eval/results/`.

### Results

66 questions over the 5 indexed lectures: 55 with a gold lecture and time range (factual, conceptual, "where does he explain…", course info) and 11 the lectures do not cover. Run on 2026-11-13 against the deployed backend; latency includes the network round trip.

| Retrieval | Recall@1 | Recall@5 | MRR | Timestamp ±30s | Median latency |
|---|---|---|---|---|---|
| Dense | 0.93 | 0.98 | 0.95 | 0.87 | 494 ms |
| BM25 | 0.80 | 0.98 | 0.88 | 0.76 | 475 ms |
| Hybrid (RRF) | **0.95** | 0.98 | **0.96** | **0.89** | 505 ms |
| Hybrid + rerank | 0.89 | **1.00** | 0.93 | 0.87 | 5790 ms |
| Hybrid + rerank, off-topic kept | 0.89 | 1.00 | 0.93 | 0.87 | 5790 ms |

What this shows so far:

- Hybrid retrieval beats either retriever alone at rank 1. BM25 on its own struggles because the transcripts are machine translations with misspelt terms.
- The reranker gets the right passage into the top 5 every time, but costs rank-1 accuracy here and makes search about 11x slower. It needs a closer look before it earns its place.
- Filtering off-topic segments changes nothing on this set, as almost no questions target off-topic talk.
- Refusals: 9 of the 11 uncovered questions score near 0 with the reranker. Two near misses score high ("bias-variance trade-off" 0.80, "deploying with Flask" 0.52) because lecture 1 lists them as upcoming topics, so for these the refusal relies on the LLM rather than the score threshold.

Caveats: 57 of the 66 questions are drafts written from the transcripts (marked `"status": "draft"`) and have not yet been checked against the videos. The set is small, and covers only transcript content, not slides or the board. Treat these numbers as a first baseline.

## Tests

```bash
cd backend
uv run pytest
```

## Project layout

```
backend/app/
  ingestion/    acquire, transcribe, chunk, classify, pipeline
  retrieval/    store (Qdrant), search (hybrid + rerank)
  generation/   llm, answer (prompt, citations, SSE stream)
  api.py        REST endpoints
frontend/       Next.js app: course list, player + transcript, chat
eval/           question set and retrieval evaluation script
```

## Content note

Lecture videos are played through YouTube embeds and are not redistributed. Only use lectures you have the right to process.
