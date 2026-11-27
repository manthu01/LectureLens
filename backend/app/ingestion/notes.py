"""Uploaded written notes for a lecture (PDF, Markdown or plain text), split into passages.

Notes have no timestamps, so they are not aligned to segments like slides are. Each passage is
indexed as its own point (`source: "notes"`, see store.index_notes) and can be retrieved and cited
next to the transcript; citations open the notes file at the passage's page.

Passages aim for TARGET_WORDS words and never exceed MAX_WORDS: long enough to carry a whole idea,
short enough that one passage is about one thing (the transcript segments are ~60 s, ~150 words).
PDF passages never cross a page, so a citation can open the right page. Markdown/text passages
follow headings and paragraphs; their `page` is the passage number.
"""

import json
import re
from pathlib import Path

TARGET_WORDS = 180  # flush a passage once it has this many words
MAX_WORDS = 250  # a single paragraph longer than this is split
MIN_WORDS = 40  # a passage shorter than this is not closed at a heading; it joins the next section
SUFFIXES = (".pdf", ".md", ".txt")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> int:
    return len(text.split())


def split_long(text: str, max_words: int = MAX_WORDS, target: int = TARGET_WORDS) -> list[str]:
    """Split text longer than max_words into pieces of about `target` words, at sentence ends
    where possible (a sentence longer than max_words is cut at word boundaries)."""
    if _words(text) <= max_words:
        return [text] if text.strip() else []
    pieces, cur = [], []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        while len(words) > max_words:  # a run-on "sentence" (e.g. a PDF page without punctuation)
            if cur:
                pieces.append(" ".join(cur))
                cur = []
            pieces.append(" ".join(words[:target]))
            words = words[target:]
        if cur and len(cur) + len(words) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur += words
        if len(cur) >= target:
            pieces.append(" ".join(cur))
            cur = []
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _blocks(text: str) -> list[tuple[str, str]]:
    """(heading, paragraph) pairs in order; heading is the nearest Markdown heading above."""
    heading, out, para = "", [], []

    def close() -> None:
        if para:
            out.append((heading, " ".join(" ".join(l.split()) for l in para)))
            para.clear()

    for line in text.replace("\r\n", "\n").split("\n"):
        m = _HEADING.match(line)
        if m:
            close()
            heading = m.group(1).strip()
        elif not line.strip():
            close()
        else:
            para.append(line)
    close()
    return out


def split_text(text: str) -> list[dict]:
    """Passages of a Markdown or plain-text document: [{page, topic, text}], page = passage number.

    Paragraphs are packed into passages up to TARGET_WORDS; a new heading starts a new passage
    unless the current one is still tiny. Each passage keeps its section heading as `topic`
    (embedded with it, like a segment's topic)."""
    passages: list[tuple[str, str]] = []
    cur_heading, cur = "", []

    def flush() -> None:
        if cur:
            passages.append((cur_heading, " ".join(cur)))
            cur.clear()

    for heading, para in _blocks(text):
        if heading != cur_heading and cur and sum(map(_words, cur)) >= MIN_WORDS:
            flush()
        if not cur:
            cur_heading = heading
        for piece in split_long(para):
            if cur and sum(map(_words, cur)) + _words(piece) > MAX_WORDS:
                flush()
                cur_heading = heading
            cur.append(piece)
            if sum(map(_words, cur)) >= TARGET_WORDS:
                flush()
                cur_heading = heading
    flush()
    return [{"page": i, "topic": h, "text": t} for i, (h, t) in enumerate(passages, 1)]


def split_pages(pages: list[str]) -> list[dict]:
    """Passages of a paged document (PDF): each page on its own, long pages split further.
    `page` is the 1-based page number, so several passages can share one page."""
    out = []
    for n, page in enumerate(pages, 1):
        text = " ".join(page.split())
        out += [{"page": n, "topic": "", "text": piece} for piece in split_long(text)]
    return out


def _pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def extract(path: Path) -> list[dict]:
    """Passages [{page, topic, text}] of a .pdf, .md or .txt file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return split_pages(_pdf_pages(path))
    if suffix in (".md", ".txt"):
        return split_text(path.read_bytes().decode("utf-8", errors="replace"))
    raise ValueError("Notes must be a .pdf, .md or .txt file")


def save(work_dir: Path, filename: str, data: bytes) -> list[dict]:
    """Store the uploaded file and its passages (notes.json); returns the passages.
    The new file is read before the old notes are replaced, so a bad upload leaves them intact."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError("Notes must be a .pdf, .md or .txt file")
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / f"upload-notes{suffix}"
    tmp.write_bytes(data)
    try:
        passages = extract(tmp)
        if not passages:
            raise ValueError("No text found in the notes (scanned images can't be read yet)")
    except Exception:
        tmp.unlink()
        raise
    for old in work_dir.glob("notes.*"):
        old.unlink()
    tmp.replace(work_dir / f"notes{suffix}")
    (work_dir / "notes.json").write_text(json.dumps(passages, ensure_ascii=False, indent=1), encoding="utf-8")
    return passages


def load(work_dir: Path) -> list[dict] | None:
    p = work_dir / "notes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def file_path(work_dir: Path) -> Path | None:
    return next((p for p in work_dir.glob("notes.*") if p.suffix in SUFFIXES), None)
