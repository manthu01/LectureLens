# LectureLens

**Ctrl+F for your entire semester of lectures.** Ask a question, get an answer grounded only in what was taught, and jump to the exact second the lecturer explained it.

> Status: work in progress. Weeks 1–2 of the [build plan](LECTURE_SEARCH_PROJECT.md) are done. The visual pipeline (slides/board), summaries, quizzes and deployment are not built yet.

## What it does

- **Ingests lectures** from a YouTube video, a whole playlist, or a local media file.
- **Transcribes with timestamps** using Whisper, running locally. Hindi/Hinglish lectures are translated to English during transcription.
- **Labels every segment** as teaching, off-topic, admin or exam hint, and names its topic.
- **Answers questions with citations.** Each claim links to a lecture and timestamp; clicking it seeks the video.
- **Refuses when the lectures don't cover it**, and shows the closest moments instead of guessing.

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

## Evaluation

`eval/questions.jsonl` holds questions with the lecture and time range where each is answered, plus questions the lectures do not cover.

```bash
python eval/run.py --course 1     # backend must be running
```

This compares dense, BM25, hybrid and hybrid + rerank retrieval and reports Recall@1, Recall@5, MRR and timestamp accuracy (top result within ±30s of the gold span). Results are saved to `eval/results/`.

The current question file is a small starter set used to check that the script works. Real results will be added here once the full evaluation set is written.

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
