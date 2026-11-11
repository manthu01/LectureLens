# LectureLens — "Ctrl+F for your entire semester of lectures"

> Working name. Rename freely.

## 1. Problem Statement

Students sit through 40+ hours of lectures per subject. When exams come, they can't find *where* the professor explained something, what was flagged as important, or what was actually covered vs. what is in the textbook. Lecture recordings are long, messy (jokes, attendance, side talk), and come in very different camera setups.

**LectureLens** lets a student upload a semester of lecture recordings, slides and notes, then:
- Ask any question and get an answer **grounded only in what was taught**, with a **clickable timestamp** that jumps the video to that exact moment.
- See each lecture broken into a **topic timeline** with off-topic parts marked.
- Get **exam hints** and **announcements** the professor mentioned in passing.
- Get **summaries** and **practice quizzes** generated only from class content.

### One-line pitch
"It's Ctrl+F for your entire semester of lectures. Ask a question, jump to the exact second your professor explained it."

### Why not just use ChatGPT / Claude directly?
| General chatbot | LectureLens |
|---|---|
| Can't ingest 40+ hours of video | Transcribes and indexes the whole semester |
| Answers from internet knowledge | Answers only from *your* class; says "not covered" otherwise |
| No timestamps | Jumps to the exact moment in the video |
| Forgets between chats | Persistent, grows weekly, shareable with the class |
| No proof of accuracy | Measured with a custom evaluation set |

---

## 2. Scope

### MVP (must have)
1. Upload/ingest lecture videos (local file or YouTube URL).
2. Timestamped transcription.
3. Chunking + segment classification (teaching / off-topic / admin / exam_hint).
4. Hybrid retrieval (BM25 + dense) with reranking.
5. Q&A with citations and clickable timestamps; refuses when unsupported.
6. Web UI: course page, lecture list, video player that seeks to timestamps, chat panel.
7. Per-lecture summary and topic timeline.
8. Evaluation set (50–100 questions) + eval script with metrics.

### V2 (should have)
- Visual pipeline: slide/screen OCR, board reading via vision model, slide-to-transcript alignment.
- Automatic lecture-layout detection and routing.
- Exam Hints and Announcements pages.
- Practice quiz generation and answer checking.
- "Skip the fluff" stat: minutes of actual teaching per lecture.
- Upload of separate slides (PDF/PPTX) and notes.

### Stretch
- Back-of-class / low-quality board handling (crop + upscale).
- Multilingual / Hinglish lectures (transcribe + translate).
- Multiple users / shared class workspace.
- "Exam mode": stitched clip playlist for a unit.

### Out of scope
- Live lecture streaming.
- Commercial distribution of copyrighted lectures.

---

## 3. High-Level Architecture

```
                ┌───────────────────────────────────────────────┐
                │                 Frontend (Web)                │
                │  Course page · Lecture player · Chat · Quiz   │
                └───────────────┬───────────────────────────────┘
                                │ REST / SSE (streaming answers)
                ┌───────────────▼───────────────────────────────┐
                │              Backend API (FastAPI)            │
                │  /courses /lectures /ingest /ask /summary ... │
                └──────┬───────────────┬───────────────┬────────┘
                       │               │               │
          ┌────────────▼───┐   ┌───────▼────────┐  ┌───▼──────────────┐
          │ Ingestion Jobs │   │ Retrieval +    │  │  Metadata DB     │
          │ (worker queue) │   │ Answer Engine  │  │  (Postgres/SQLite)│
          └──────┬─────────┘   └───────┬────────┘  └──────────────────┘
                 │                     │
   ┌─────────────▼──────────┐   ┌──────▼─────────┐
   │ Processing Pipeline    │   │ Vector Store   │
   │ audio→ASR→chunk→label  │──▶│ (Qdrant)       │
   │ video→frames→OCR/VLM   │   │ + BM25 index   │
   │ align → embed → index  │   └────────────────┘
   └────────────────────────┘
                 │
          ┌──────▼──────┐
          │ File storage│  videos, audio, frames, transcripts
          └─────────────┘
```

---

## 4. Tech Stack (defaults — swap as needed)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | ML ecosystem |
| Backend API | FastAPI | Async, simple, auto docs |
| Job queue | RQ + Redis (or Celery); simple background tasks for MVP | Ingestion is slow |
| Video download | yt-dlp | YouTube/NPTEL/OCW |
| Media processing | ffmpeg | Extract audio, frames |
| Transcription | faster-whisper (large-v3 or distil) with word timestamps; optional WhisperX for alignment/diarization | Free, local, accurate |
| Scene change detection | PySceneDetect or frame-diff with OpenCV | Capture slide/board changes |
| OCR | PaddleOCR or Tesseract for slides | Clean printed text |
| Vision model | Multimodal LLM (configurable) for board/handwriting/diagrams | Handles messy boards |
| Embeddings | `BAAI/bge-m3` or `bge-small-en-v1.5` (local) or an API embedding model | Good quality, free option |
| Vector DB | Qdrant (Docker) | Metadata filters, hybrid support |
| Sparse search | BM25 (`rank_bm25`) or Qdrant sparse vectors | Exact terms, formulas, names |
| Reranker | `BAAI/bge-reranker-v2-m3` | Big accuracy gain |
| LLM | Configurable via env (e.g. Claude API, model set in `.env`); local fallback via Ollama | Answering, labeling, summaries |
| Metadata DB | SQLite (MVP) → Postgres | Courses, lectures, segments |
| Frontend | Next.js + Tailwind (or Streamlit for a fast MVP) | Video player + chat |
| Video player | HTML5 `<video>` / react-player with seek-to-timestamp | Core demo moment |
| Eval | Custom scripts + RAGAS (optional) | Metrics |
| Deployment | Docker Compose; frontend on Vercel, backend on Render/Railway/a VM | Live demo link |

All model names, API keys and paths live in `.env` — never hardcode.

---

## 5. Data Model

### Course
`id, name, professor, description, created_at`

### Lecture
`id, course_id, title, number, source_url, video_path, audio_path, duration_sec, layout_type (screen_only | facecam_screen | board_close | board_far | unknown), visual_quality_score, status (queued | transcribing | processing | indexed | failed), created_at`

### TranscriptWord (optional, for precise seeking)
`lecture_id, word, start_sec, end_sec`

### Segment (core unit of retrieval)
```
id
lecture_id
start_sec, end_sec
text                    # transcript text for this segment
label                   # teaching | off_topic | admin | exam_hint
label_confidence
topic                   # short topic name, e.g. "Binary Search"
visual_text             # OCR/VLM text of slide/board shown during this segment
slide_id                # if aligned to an uploaded slide
summary                 # 1-line summary (optional)
embedding_id            # pointer into Qdrant
```

### Slide / Frame
`id, lecture_id, timestamp_sec, image_path, ocr_text, vlm_description, source (frame | uploaded_pdf)`

### ExamHint / Announcement
`id, lecture_id, segment_id, text, timestamp_sec, type (exam_hint | deadline | announcement)`

### QAPair (eval + logging)
`id, question, answer, cited_segments, latency_ms, feedback`

---

## 6. Processing Pipeline (Ingestion)

Each step is idempotent and saves output to disk so failures can resume.

### Step 1 — Acquire
- Input: YouTube URL / local video / uploaded slides PDF / notes.
- `yt-dlp` downloads video; store at `data/courses/{course}/{lecture}/video.mp4`.

### Step 2 — Extract audio
- `ffmpeg -i video.mp4 -ac 1 -ar 16000 audio.wav`

### Step 3 — Transcribe (audio-first backbone)
- faster-whisper with `word_timestamps=True`, VAD filter on.
- Output: `transcript.json` = list of `{start, end, text, words:[...]}`.
- Works the same for every lecture layout.
- Background chatter is mostly suppressed by VAD + dominant-speaker transcription.

### Step 4 — Chunking
- Build segments of ~45–90 seconds (or ~150–300 tokens), splitting on sentence boundaries and pauses > 1.5s.
- 10–15% overlap between neighbours.
- Keep exact `start_sec` / `end_sec` for every segment.
- Optional: topic segmentation (embedding similarity drop between windows) to create the topic timeline.

### Step 5 — Segment classification
- Labels: `teaching`, `off_topic` (jokes, stories, chit-chat), `admin` (attendance, logistics), `exam_hint` ("important for exam", "this will come").
- MVP: LLM classifies batches of segments with a strict JSON schema.
- Improvement: keyword rules for exam hints/deadlines + LLM; later a small fine-tuned classifier trained on LLM labels (cheaper, faster, good experiment for the README).
- Extract exam hints and announcements into their own tables.
- Analogies/real-life examples inside teaching stay as `teaching`.

### Step 6 — Visual pipeline (V2)
1. **Layout detection:** sample ~10 frames, ask the vision model (or a small classifier) to pick a layout type. Store `layout_type`.
2. **Key frame extraction:** scene-change detection; for boards, also capture frame *just before* large content drops (erasing).
3. **Route by layout:**
   - `screen_only` → full-frame OCR.
   - `facecam_screen` → detect screen region (largest rectangle / vision model bounding box), crop, OCR.
   - `board_close` → vision model reads handwriting, equations (LaTeX), diagrams.
   - `board_far` → detect & crop board, upscale (e.g. Real-ESRGAN), then vision model; if confidence low, mark unreadable and rely on audio + uploaded material.
4. **Visual quality score** per lecture (avg OCR confidence / readability).
5. Deduplicate near-identical frames (perceptual hash).

### Step 7 — Alignment
- Attach to each segment the text of the frame(s) visible between `start_sec` and `end_sec` → `visual_text`.
- If slides PDF uploaded: match frame OCR to slide text (fuzzy/embedding match) → `slide_id`.

### Step 8 — Embed & index
- Text to embed = `topic + transcript text + visual_text` (keep a version without visual_text for ablation).
- Upsert into Qdrant with payload: `course_id, lecture_id, lecture_number, start_sec, end_sec, label, topic`.
- Build BM25 index over same texts.
- Off-topic segments are indexed but **excluded by default via filter** (so a toggle can include them).

### Step 9 — Derived content
- Per-lecture summary (map-reduce over teaching segments).
- Topic timeline (chapters with timestamps).
- "Skip the fluff" stats: teaching minutes vs total.

---

## 7. Retrieval & Answering

### Query flow
1. **Query understanding:** optional rewrite (expand abbreviations, fix typos); detect filters ("in lecture 5", "unit 3").
2. **Hybrid retrieval:** top-30 dense + top-30 BM25, filtered to `course_id` and `label != off_topic`; merge with Reciprocal Rank Fusion.
3. **Rerank:** cross-encoder reranker → top-5/6.
4. **Context build:** include neighbouring segment text for continuity; include visual_text.
5. **Answer generation:** LLM prompt rules:
   - Answer **only** from provided segments.
   - Cite each claim with segment IDs → rendered as `[Lecture 7 · 23:14]` clickable links.
   - If segments don't support an answer: respond "This doesn't seem to be covered in your lectures" and show closest moments.
   - Short by default; offer to elaborate.
6. **Confidence gate:** if top reranker score < threshold → refusal path without calling the LLM for an answer.
7. Stream response to the frontend via SSE.

### Other endpoints using retrieval
- **Summarize lecture/unit**
- **Generate quiz** from teaching segments of chosen lectures (MCQ + short answer, each with source timestamp).
- **Check answer**: grade the student's answer against source segments.
- **Exam hints** list across course.

---

## 8. API Design (FastAPI)

```
POST   /courses                         create course
GET    /courses                         list
GET    /courses/{id}                    detail + lectures
POST   /courses/{id}/lectures           add lecture (url or upload) → enqueue ingest
GET    /lectures/{id}                   metadata, status, timeline, stats
GET    /lectures/{id}/transcript        segments with labels
GET    /lectures/{id}/summary
GET    /lectures/{id}/video             stream video (range requests)
POST   /courses/{id}/ask                {question, include_off_topic?} → SSE stream
GET    /courses/{id}/exam-hints
GET    /courses/{id}/announcements
POST   /courses/{id}/quiz               {lecture_ids, n} → questions
POST   /quiz/check                      {question_id, answer} → feedback
GET    /jobs/{id}                       ingestion progress
```

Ask response (final event):
```json
{
  "answer": "…",
  "citations": [
    {"segment_id": "s_812", "lecture_id": 7, "lecture_title": "Normalization",
     "start_sec": 1394, "end_sec": 1452, "snippet": "…"}
  ],
  "supported": true
}
```

---

## 9. Frontend

### Pages
1. **Home / Courses:** list courses, create course.
2. **Course page:** lecture cards (status, duration, teaching minutes), Exam Hints tab, Announcements tab, chat box.
3. **Lecture page:** video player (left), chat + transcript (right), topic timeline bar under the video with off-topic regions greyed out, summary panel.
4. **Quiz page:** pick lectures/unit → questions → feedback with "watch explanation" timestamp buttons.

### Key interactions
- Clicking a citation opens that lecture and **seeks the player** to `start_sec`.
- Transcript auto-scrolls with playback; segments coloured by label.
- "Skip the fluff" toggle auto-skips off-topic segments during playback.
- Ingestion progress bar (transcribing → processing → indexed).

---

## 10. Evaluation (the differentiator)

### Eval dataset
- `eval/questions.jsonl`, 50–100 hand-written items:
```json
{"id":"q1","question":"What is the worst-case time of quicksort?",
 "lecture_id":4,"gold_start_sec":1820,"gold_end_sec":1905,
 "gold_answer":"O(n^2) when pivot is always smallest/largest",
 "type":"factual"}
```
- Include types: factual, conceptual, "where did he say", exam-hint, **unanswerable** (not covered → should refuse), visual-only (only on board/slide).

### Metrics
- **Retrieval:** Recall@k, MRR — a hit = retrieved segment overlaps gold time range.
- **Timestamp accuracy:** % of answers whose top citation is within ±30s of gold.
- **Answer quality:** correctness (LLM-as-judge vs gold + manual spot check), faithfulness (claims supported by cited segments).
- **Refusal:** precision/recall on unanswerable questions.
- **Classifier:** precision/recall/F1 for segment labels on ~200 hand-labelled segments.
- **Latency & cost** per query.

### Experiments to run & report (ablations)
| Experiment | Compare |
|---|---|
| Chunk size | 30s vs 60s vs 90s; fixed vs pause-based |
| Retrieval | BM25 vs dense vs hybrid |
| Reranker | with vs without |
| Off-topic filtering | on vs off |
| Visual text | transcript-only vs transcript + OCR/VLM |
| Embedding model | small vs large |
| Classifier | LLM vs fine-tuned small model (accuracy, cost, speed) |

`python -m eval.run --config configs/hybrid_rerank.yaml` → writes `eval/results/{run}.json` and a markdown table. Put the final table in the README.

---

## 11. Project Structure

```
lecturelens/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app
│   │   ├── config.py              # settings from .env
│   │   ├── db/ (models.py, session.py)
│   │   ├── api/ (courses.py, lectures.py, ask.py, quiz.py)
│   │   ├── ingestion/
│   │   │   ├── acquire.py         # yt-dlp, uploads
│   │   │   ├── audio.py           # ffmpeg
│   │   │   ├── transcribe.py      # faster-whisper
│   │   │   ├── chunk.py
│   │   │   ├── classify.py
│   │   │   ├── visual/ (layout.py, frames.py, ocr.py, vlm.py, align.py)
│   │   │   ├── embed_index.py
│   │   │   ├── derive.py          # summaries, timeline, stats
│   │   │   └── pipeline.py        # orchestrates steps, resumable
│   │   ├── retrieval/ (hybrid.py, rerank.py, filters.py)
│   │   ├── generation/ (llm.py, prompts.py, answer.py, quiz.py)
│   │   └── workers/ (queue.py, tasks.py)
│   ├── tests/
│   └── requirements.txt
├── frontend/                      # Next.js app
├── eval/
│   ├── questions.jsonl
│   ├── labels_segments.jsonl
│   ├── run.py
│   └── results/
├── configs/                       # retrieval/chunking experiment configs
├── data/                          # gitignored media + artifacts
├── docker-compose.yml             # api, worker, redis, qdrant, (postgres)
├── .env.example
└── README.md
```

---

## 12. Configuration (`.env.example`)

```
LLM_PROVIDER=anthropic
LLM_MODEL=                # e.g. latest Claude model string
VISION_MODEL=
ANTHROPIC_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda       # or cpu
QDRANT_URL=http://localhost:6333
DATABASE_URL=sqlite:///./lecturelens.db
REDIS_URL=redis://localhost:6379
DATA_DIR=./data
RERANK_THRESHOLD=0.3
```

---

## 13. Datasets / Content Sources

- **MIT OpenCourseWare:** videos + slides + notes + transcripts (CC BY-NC-SA; portfolio use OK, credit it).
- **NPTEL (IIT/IISc) on YouTube:** Indian teaching style/accents; mix of board and screen lectures — great for layout variety.
- **Stanford / CMU courses on YouTube** with public slides.
- **Own college lectures:** only with professor permission.
- **Research datasets for extra eval:** LPM (lecture video + slides, CMU), AVLectures, VT-SSum (transcript summaries).

Plan: start with one course you know well (~20 lectures), screen-only first; then add a board-based NPTEL course to show layout handling.

---

## 14. Build Plan (≈4 weeks part-time)

### Week 1 — End-to-end skeleton
- [ ] Repo, Docker Compose (Qdrant, Redis), FastAPI hello world
- [ ] Download 5 lectures, extract audio, transcribe with timestamps
- [ ] Simple chunking, embed, index in Qdrant
- [ ] `/ask` with dense retrieval + LLM answer + timestamp citations
- [ ] Minimal UI (Streamlit OK) with video seeking
- [ ] Start writing eval questions as you watch

### Week 2 — Retrieval quality + noise handling
- [ ] Segment classification + exam hints/announcements extraction
- [ ] BM25 + hybrid fusion + reranker
- [ ] Off-topic filtering, refusal path with threshold
- [ ] Eval script v1 with retrieval metrics; first baseline numbers

### Week 3 — Product + visuals
- [ ] Next.js UI: course page, lecture page, timeline, chat with SSE
- [ ] Summaries, topic timeline, "skip the fluff" stats
- [ ] Visual pipeline for screen_only + facecam_screen; alignment into segments
- [ ] Quiz generation + checking

### Week 4 — Rigor + polish
- [ ] Board lectures via vision model; layout detection
- [ ] Complete eval set (incl. unanswerable + visual-only); run all ablations
- [ ] Deploy (live link), demo video (2 min), README with architecture diagram + results table
- [ ] Stretch: back-of-class handling, Hinglish

Rule: always keep the system working end to end; improve one stage at a time and measure.

---

## 15. Error Handling & Edge Cases

- Long videos: process audio in chunks; resumable pipeline with per-step status.
- Failed downloads/transcriptions: mark lecture `failed` with reason, allow retry.
- Unreadable visuals: low confidence → skip visual_text, flag in UI.
- Multiple speakers (student questions): optional diarization; label student questions so answers prefer professor speech.
- Hinglish/code-mixed audio: Whisper multilingual; store original + English text (stretch).
- Duplicate questions/cost: cache embeddings and answers for identical queries.
- Copyright: don't redistribute videos publicly; demo with openly licensed courses.

---

## 16. Testing

- Unit tests: chunking boundaries/timestamps, RRF fusion, citation parsing, label JSON parsing.
- Integration test: ingest one short sample video (2–3 min) end to end.
- Eval runs as regression check before merging retrieval changes.

---

## 17. README / Resume Deliverables

- Live demo link + 2-minute demo video (ask question → jump to exact moment).
- Architecture diagram.
- Results table from ablations, e.g. "Hybrid + rerank + off-topic filtering improved Recall@5 from X% to Y%, timestamp accuracy Z%".
- Clear "why not ChatGPT" section.

### Resume bullet template
> Built LectureLens, a multimodal RAG system that indexes 40+ hours of lecture video and answers questions with clickable timestamps; designed a 100-question eval set and improved Recall@5 from X% to Y% via hybrid retrieval, reranking and an off-topic segment classifier; adaptive visual pipeline handles screen, face-cam and whiteboard lecture formats.

---

## 18. Instructions for Claude Code

- Build in the order of Section 14; finish Week 1 fully before starting Week 2.
- Keep every model/provider configurable via `.env`.
- Each ingestion step reads/writes files under `data/` so it can resume.
- Write tests for chunking, fusion and citation logic.
- After any retrieval change, run `eval/run.py` and record results.
- Ask before adding heavy dependencies or paid services.
