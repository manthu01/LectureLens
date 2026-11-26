"""Uploaded slides (PDF or PPTX): extract each slide's text and align slides to transcript segments.

Alignment: each segment (its transcript plus its on-screen text, when the visual pipeline ran) is
compared with each slide two ways, by dense embeddings and by TF-IDF word overlap. Slides are terse
keyword lists, so word overlap matters: on the development lecture it beat embeddings alone (see
scripts/eval_slides.py). Each segment's similarities are standardised (z-scores across slides) so
the two measures, and the jump costs below, are on the same scale. Lecturers mostly
move forward through their deck, so instead of taking the best slide per segment independently,
a dynamic program picks the sequence of slides with the highest total similarity, where moving
forward costs a little and going back costs more. This keeps one strong-but-wrong match (a recap
slide, a repeated title) from making the alignment jump around.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Jump costs, in standard deviations of a segment's similarity across slides.
FORWARD_JUMP = 0.25  # moving to a later slide than the previous segment's
BACK_JUMP = 1.0  # moving to an earlier slide
_STOP = set("the a an of to and or is are in on for with by be as it this that from at if".split())
MAX_SLIDE_CHARS = 1500


def extract_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [{"page": i, "text": _clean(p.extract_text() or "")} for i, p in enumerate(reader.pages, 1)]


def extract_pptx(path: Path) -> list[dict]:
    from pptx import Presentation

    out = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        parts = [
            "\n".join(p.text for p in shape.text_frame.paragraphs if p.text.strip())
            for shape in slide.shapes
            if shape.has_text_frame
        ]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Notes: " + notes)
        out.append({"page": i, "text": _clean("\n".join(p for p in parts if p))})
    return out


def _clean(text: str) -> str:
    # Bullet glyphs from symbol fonts come out as private-use characters; drop them.
    text = "".join(ch for ch in text if not 0xE000 <= ord(ch) <= 0xF8FF and ch not in "▪◦•●�")
    lines = [" ".join(l.split()) for l in text.splitlines()]
    return "\n".join(l for l in lines if l)[:MAX_SLIDE_CHARS]


def extract(path: Path) -> list[dict]:
    """[{page, text}] for a .pdf or .pptx file, 1-based pages."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".pptx":
        return extract_pptx(path)
    raise ValueError("Slides must be a .pdf or .pptx file")


def monotonic_path(sim: np.ndarray, forward: float = FORWARD_JUMP, back: float = BACK_JUMP) -> list[int]:
    """Best slide index per segment for a (segments x slides) similarity matrix: maximises total
    similarity minus jump costs. O(segments x slides). Pure function, for tests."""
    n, m = sim.shape
    if n == 0 or m == 0:
        return []
    score = sim[0].astype(float)
    came_from = np.zeros((n, m), dtype=int)
    for i in range(1, n):
        # For each slide j, the best previous slide: j itself, an earlier one (k < j, cost `forward`)
        # or a later one (k > j, cost `back`). Running maxima from each side make this O(m).
        before, before_k = np.full(m, -np.inf), np.zeros(m, dtype=int)
        for j in range(1, m):
            better = score[j - 1] > before[j - 1]
            before[j], before_k[j] = (score[j - 1], j - 1) if better else (before[j - 1], before_k[j - 1])
        after, after_k = np.full(m, -np.inf), np.zeros(m, dtype=int)
        for j in range(m - 2, -1, -1):
            better = score[j + 1] > after[j + 1]
            after[j], after_k[j] = (score[j + 1], j + 1) if better else (after[j + 1], after_k[j + 1])
        new = np.empty(m)
        for j in range(m):
            val, k = max((score[j], j), (before[j] - forward, before_k[j]), (after[j] - back, after_k[j]))
            new[j] = val + sim[i, j]
            came_from[i, j] = k
        score = new
    path = [int(np.argmax(score))]
    for i in range(n - 1, 0, -1):
        path.append(int(came_from[i, path[-1]]))
    return path[::-1]


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def tfidf_similarity(seg_texts: list[str], slide_texts: list[str]) -> np.ndarray:
    """Cosine similarity of sublinear TF-IDF vectors, IDF taken over the slides."""
    docs = [_tokens(t) for t in slide_texts]
    df = Counter(w for d in docs for w in set(d))
    vocab = {w: i for i, w in enumerate(df)}
    idf = np.array([math.log((len(docs) + 1) / (df[w] + 1)) + 1 for w in vocab])

    def vec(words: list[str]) -> np.ndarray:
        v = np.zeros(len(vocab))
        for w, c in Counter(words).items():
            if w in vocab:
                v[vocab[w]] = 1 + math.log(c)
        v *= idf
        n = np.linalg.norm(v)
        return v / n if n else v

    a = np.array([vec(_tokens(t)) for t in seg_texts]).reshape(len(seg_texts), len(vocab))
    b = np.array([vec(d) for d in docs]).reshape(len(docs), len(vocab))
    return a @ b.T


def _z(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


def similarity(seg_texts: list[str], slide_texts: list[str], embed) -> np.ndarray:
    """(segments x slides) score: standardised dense similarity + standardised TF-IDF similarity."""
    dense = np.asarray(embed(seg_texts), dtype=float) @ np.asarray(embed(slide_texts), dtype=float).T
    return _z(dense) + _z(tfidf_similarity(seg_texts, slide_texts))


def align(chunks: list[dict], slides: list[dict], embed) -> list[dict]:
    """Add `slide_page` and `slide_text` to each chunk. `embed(texts) -> array of unit vectors`.
    Slides with no text (pure images) can't be matched and are skipped; off-topic segments get none."""
    usable = [s for s in slides if s["text"].strip()]
    if not chunks or not usable:
        return [{**c, "slide_page": None, "slide_text": ""} for c in chunks]
    seg_texts = [(c["text"] + " " + c.get("visual_text", "")).strip() for c in chunks]
    path = monotonic_path(similarity(seg_texts, [s["text"] for s in usable], embed))
    out = []
    for c, j in zip(chunks, path):
        if c.get("label") == "off_topic":
            out.append({**c, "slide_page": None, "slide_text": ""})
        else:
            out.append({**c, "slide_page": usable[j]["page"], "slide_text": usable[j]["text"]})
    return out


def save(work_dir: Path, filename: str, data: bytes) -> list[dict]:
    """Store the uploaded file and its extracted text; returns the slides."""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".pptx"):
        raise ValueError("Slides must be a .pdf or .pptx file")
    work_dir.mkdir(parents=True, exist_ok=True)
    for old in work_dir.glob("slides.*"):
        old.unlink()
    path = work_dir / f"slides{suffix}"
    path.write_bytes(data)
    slides = extract(path)
    (work_dir / "slides.json").write_text(json.dumps(slides, ensure_ascii=False, indent=1), encoding="utf-8")
    return slides


def load(work_dir: Path) -> list[dict] | None:
    p = work_dir / "slides.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def file_path(work_dir: Path) -> Path | None:
    return next((p for p in work_dir.glob("slides.*") if p.suffix in (".pdf", ".pptx")), None)
