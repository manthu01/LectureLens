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
- Upload of separate slides (PDF/PPTX) and notes. (Slides and notes done 2026-11-27.)

### Stretch
- [x] Back-of-class / low-quality board handling (crop + upscale). Implemented but untested on real board footage, since the demo courses have no board lectures.
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

### Status (updated 2026-11-27)

**Done 2026-11-27: notes upload** (`backend/app/ingestion/notes.py`, `POST /lectures/{id}/notes?filename=` raw body, max 20 MB; `GET /lectures/{id}/notes[/file]`). PDF/MD/TXT notes are split into ~150-250-word passages (per page for PDF, by heading/paragraph otherwise) and indexed synchronously in the same collection as `source: "notes"` points (`page`, start/end = null); search and answers use them, `lecture_segments` (transcript, timeline, summary, quiz) excludes them, re-indexing a lecture's transcript keeps them, and a rebuilt collection gets them back from `notes.json`. Citations show "L4 · notes p.2" and open the file at that page; "Play all cited clips" skips them. Not evaluated on real notes yet.

**Done 2026-11-27: slide upload + alignment** (`backend/app/ingestion/slides.py`, `POST /lectures/{id}/slides?filename=` with the raw file as body, so the read-only deploy needs no multipart library; `GET /lectures/{id}/slides[/file]`). PDF via pypdf, PPTX via python-pptx (incl. speaker notes). Alignment: z-scored dense similarity + z-scored TF-IDF similarity, then a monotonic dynamic program (forward jump 0.25 sd, backward 1.0 sd); off-topic segments get no slide. Each segment stores `slide_page`/`slide_text`; slide text is embedded ("Slide: ..."), shown to the answering LLM and the cross-encoder; UI has Add slides, Slide N links (PDF #page=N) and slide numbers on citation chips. Upload on an indexed lecture re-aligns from the index (no transcript needed). Eval (`backend/scripts/eval_slides.py`, gold from frame reads): dev MIT 6.0001 L1 exact 0.74 / within-1 0.94 with on-screen text (0.49/0.77 transcript only); held-out L2 0.82/0.97 (0.63/0.87 transcript only; dense-only 0.71/0.87 beat dense+TF-IDF there, so the TF-IDF choice is unsettled).

**Done 2026-11-27: visual pipeline (V2)** (`backend/app/ingestion/visual/`). Downloads the 720p video-only stream; decodes keyframes only (~every 5 s) with PyAV; keeps a frame just before ink is removed (page turn / slide change / wipe) or when enough new ink has appeared since the last kept frame; pixels that change in >25% of steps (face-cam inset) are masked; consecutive near-identical frames are dropped by ink difference (an 8x8 perceptual hash was too coarse: it merged pages with new handwriting). Gemini (`VISION_MODEL`, default the LLM model) reads 6 frames per call and returns layout + text (handwriting, code, LaTeX, one-line diagram notes) + readability; a batch refused by the RECITATION filter is retried frame by frame and a still-refused frame is left empty. Lecture `layout_type` = layout on screen longest; `visual_quality_score` = time-weighted readability. Each segment gets `visual_text` = text of frames overlapping it (fuller versions of a growing page win); it is embedded as `topic. text On screen: ...`, scored by the cross-encoder, shown to the answering LLM, and shown in the transcript UI. Deviations from the plan below: no Tesseract/PaddleOCR (the course is handwriting in OneNote, which OCR reads badly, and one vision call covers layout + text); no board crop/upscale (no board lectures in the demo course); no Frame table (reads cached as JSON per lecture, like transcripts). Existing lectures processed with `backend/scripts/add_visual.py`: lectures 1-2 talking head, 3-10 facecam_screen, readability 0.89-0.98. Eval: 10 visual-only questions (`type: visual`): hybrid R@1 0.40 -> 0.90, R@5 0.70 -> 1.00; transcript questions unchanged (0.95). End to end 9/10 answered correctly, 11/11 uncovered still refused. Not done: slide PDF upload/alignment, board_far crop + upscale (no test data).

Live demo: https://lecturelens-app.vercel.app (10 lectures of CampusX "100 Days of Machine Learning"; lectures 6-10 added 2026-11-23, retrieval eval rerun: hybrid R@1 0.95, `eval/results/20261123-050402.md`).

**Done: answer-quality eval** (`eval/judge.py`). Decided: the grader is Gemini, not Claude (Claude would need paid API credits). Answers come from `gemini-3.5-flash-lite` and are graded by a different, larger model, `gemini-3.6-flash` (`3.8-flash` was the first choice but kept returning 503/504 errors on the free tier). For each answer, the grader splits it into claims and checks each one against the retrieved excerpts and against the excerpt it cites. It also checks whether the answer addresses the question and whether it agrees with the gold answer (yes/partly/no). Refusals are counted from the answer text, with no grader.
- Done: all 66 answers generated and saved in `eval/results/judge-20261116-134851.json`. Refusals: 11/11 uncovered questions refused (9 by the score gate, 2 by the LLM: u3, u10), and 0 of the 55 answerable questions refused.
- Grading: 26 of 55 answers graded (q4 was cleared for re-grading after a fix: the grader now sees lecture titles, as the answering model does). On the free tier, the 20-a-day limit plus frequent 503/504 overloads gave about one grade per day, so the script makes no retries and stops at the first error. Results so far are written up in the README as **partial**. To grade more when convenient: `backend/.venv/Scripts/python eval/judge.py --answers eval/results/judge-20261116-134851.json`. To refresh the report without API calls, add `--report-only`. When the count changes, update the README table.
- DONE 2026-11-27: all 55 graded. 121/122 claims supported by the excerpts (q35 has one unsupported claim), 55/55 answers address the question, 54/55 match the gold answer (q36 answers only "in Lecture 3").
- Hand labels: done (`eval/results/judge-20261116-134851-labels.md`; all 20 faithful; q25, q53, q19 "partly" match gold). Agreement: faithful 20/20, gold 17/20 (q25, q53, q19: the grader is lenient on incomplete answers). Run `eval/judge.py --agreement eval/results/judge-20261116-134851-labels.md` to refresh it.

**Ablations** (`backend/scripts/ablate.py`, results in README and `eval/results/ablations-20261118-024509.md`). Done: chunk length (60 s best on every metric; 120 s drops timestamp accuracy to 0.64), embedding model (keep bge-small; base-size models are no better and 5x slower to index), classifier (topic prefix +2 questions; off-topic filter no effect on this set). Whisper size (done 2026-11-19, `eval/results/ablations-20261119-132809.md`): `medium` is no better than `small` (hybrid R@1 0.91 vs 0.93, dense 0.85 vs 0.91), so the app keeps `small` and `large-v3` is skipped. History of the run, kept for reference: Whisper `medium` re-transcription (`backend/scripts/transcribe_variant.py`, writes `transcript-medium.json` per lecture, skips lectures already done). As of 2026-11-19, lectures 1 to 4 are done; lecture 5 (about 19.5 min of audio, about 25 to 30 min to transcribe) is still to do. Measured speed on this CPU: 1.3x to 1.85x real time, about 1.4 GB of RAM. Both runs so far were stopped by Claude Code because the whole system ran low on memory while the session was idle (the 2026-11-18 run left no output; the 2026-11-19 run was stopped at about 10:13 during lecture 5). To resume, close memory-heavy apps first, then run `backend/.venv/Scripts/python -u backend/scripts/transcribe_variant.py --course 1 --model medium > eval/logs/whisper-medium.log 2>&1`. For long unattended runs, start Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` so it does not stop background jobs. After lecture 5, run `ablate.py --only whisper` and add the result to the README. Then decide whether `large-v3` is worth running (estimated 4 to 6 h on CPU); run it only if `medium` clearly beats `small`.

Then: demo video (script ready, recording declined for now). Slide/board reading done 2026-11-27.
Small fixes: done (error message shows the real backend URL; demo has 10 lectures).

Recent results:
- Retrieval (66 questions): hybrid Recall@1 0.96, Recall@5 1.00, MRR 0.98.
- Reranker sweep (5 cross-encoders): re-sorting never beat hybrid at rank 1, so the hybrid order is kept and MiniLM-L-6 only scores hits for the refusal gate. A search takes ~0.9 s, down from ~5.6 s.
- Refusals: 9/11 uncovered questions fall below the score threshold. The 2 near misses (bias-variance, Flask) pass it, but the LLM refuses both.
- 57 of the 66 eval questions were checked against the transcript text only, not the videos (`"status": "skimmed"`).

### Week 1 — End-to-end skeleton
- [x] Repo, FastAPI app (changed: no Docker Compose/Redis; Qdrant runs embedded locally and on Qdrant Cloud when deployed)
- [x] Download 5 lectures, extract audio, transcribe with timestamps (faster-whisper `small`, translate mode for Hinglish)
- [x] Simple chunking, embed, index in Qdrant
- [x] `/ask` with retrieval + LLM answer + timestamp citations
- [x] UI with video seeking (went straight to Next.js instead of Streamlit)
- [x] Eval questions (66: 55 with gold spans, 11 the lectures don't cover)

### Week 2 — Retrieval quality + noise handling
- [x] Segment classification + exam hints/announcements extraction
- [x] BM25 + hybrid fusion + reranker (the reranker scores hits for the refusal gate instead of reordering them; see the reranker sweep in `eval/results/`)
- [x] Off-topic filtering, refusal path with threshold
- [x] Eval script v1 with retrieval metrics; first baseline numbers

### Week 3 — Product + visuals
- [x] Next.js UI: course page with lecture panels, timeline, chat with SSE (no separate lecture page)
- [x] Summaries, topic timeline, "skip the fluff" stats (teaching minutes vs total)
- [x] Visual pipeline for screen_only + facecam_screen; alignment into segments (vision model reads all layouts; no OCR engine)
- [x] Quiz generation + checking (multiple choice, checked in the UI)

### Week 4 — Rigor + polish
- [x] Board lectures via vision model; layout detection (board layouts supported by the prompt, untested: the demo course has no board lectures)
- [ ] Complete eval set (incl. unanswerable + visual-only); run all ablations. Partly done: the unanswerable questions exist, retrieval-mode and reranker ablations are done; the answer-quality eval is partly graded (26/55); chunking, embedding and classifier ablations are done; Whisper and on-screen-text ablations done; 10 visual-only questions added 2026-11-27; answer grading complete (55/55)
- [ ] Deploy (live link), demo video (2 min), README with architecture diagram + results table. Partly done: deployed on Vercel and the README has a results table and an ASCII architecture sketch; still to do are the demo video and a proper diagram
- [ ] Stretch: back-of-class handling, Hinglish. Hinglish is partly handled: Whisper translates the speech to English. Back-of-class handling done 2026-11-27 (`visual/enhance.py`): a `board_far` frame, or a screen/board frame read with readability < 0.6, is cropped to the region holding ink (NumPy: pixels far from a local blur, counted per cell, borders ignored, padded, minimum size), upscaled to 1600 px wide with Pillow LANCZOS + autocontrast + unsharp mask, and read again (crops batched 6 per call); the read with the higher readability is kept and marked `enhanced` in `reads.json`, so a rerun never retries a frame. Frames read below 0.3 add no `visual_text`, and the lecture list shows "screen · 97% readable" or flags "board hard to read" (quality < 0.5). Implemented but untested on real board footage, since the demo courses have no board lectures; on the existing screen lectures the few low-readability frames fill the screen, so no crop is made and nothing changes

Rule: always keep the system working end to end; improve one stage at a time and measure.

---

## 15. Error Handling & Edge Cases

- Long videos: process audio in chunks; resumable pipeline with per-step status.
- Failed downloads/transcriptions: mark lecture `failed` with reason, allow retry.
- Unreadable visuals: low confidence → skip visual_text, flag in UI. Done: frames read below 0.3 add no visual_text; lectures with visual quality < 0.5 are flagged "hard to read" in the lecture list.
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
