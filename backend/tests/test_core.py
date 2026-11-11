from app.generation.answer import cited_indices, fmt_time
from app.ingestion.chunk import chunk_transcript
from app.ingestion.classify import parse_labels


def _segs(n: int, dur: float = 10.0, gap: float = 0.0) -> list[dict]:
    out, t = [], 0.0
    for i in range(n):
        out.append({"start": t, "end": t + dur, "text": f"s{i}"})
        t += dur + gap
    return out


def test_chunk_empty():
    assert chunk_transcript([]) == []


def test_chunk_covers_everything_with_exact_timestamps():
    segs = _segs(30)  # 300s of speech
    chunks = chunk_transcript(segs)
    assert chunks[0]["start_sec"] == 0.0
    assert chunks[-1]["end_sec"] == 300.0
    assert all(c["end_sec"] - c["start_sec"] <= 90.0 for c in chunks)
    words = " ".join(c["text"] for c in chunks).split()
    assert set(words) == {f"s{i}" for i in range(30)}


def test_chunk_overlap_repeats_last_segment():
    chunks = chunk_transcript(_segs(30), overlap_segs=1)
    for a, b in zip(chunks, chunks[1:]):
        assert a["text"].split()[-1] == b["text"].split()[0]
        assert b["start_sec"] < a["end_sec"]


def test_chunk_no_overlap_is_contiguous():
    chunks = chunk_transcript(_segs(30), overlap_segs=0)
    for a, b in zip(chunks, chunks[1:]):
        assert b["start_sec"] == a["end_sec"]


def test_chunk_splits_on_pause_after_min_length():
    segs = _segs(5) + [{"start": 55.0, "end": 65.0, "text": "after"}]  # 5s pause at 50s
    chunks = chunk_transcript(segs, overlap_segs=0)
    assert [c["end_sec"] for c in chunks] == [50.0, 65.0]


def test_cited_indices_order_dedupe_and_bounds():
    assert cited_indices("A [S2]. B [S1][S2]. C [S9] [S0].", 3) == [2, 1]
    assert cited_indices("no citations", 3) == []


def test_parse_labels_fills_gaps_and_rejects_bad_items():
    items = [
        {"i": 0, "label": "off_topic", "topic": ""},
        {"i": 2, "label": "exam_hint", "topic": " Gradient Descent "},
        {"i": 1, "label": "nonsense", "topic": "x"},  # invalid label
        {"i": 7, "label": "admin", "topic": ""},  # out of range
        "garbage",
    ]
    assert parse_labels(items, 3) == [
        {"label": "off_topic", "topic": ""},
        {"label": "teaching", "topic": ""},
        {"label": "exam_hint", "topic": "Gradient Descent"},
    ]
    assert parse_labels(None, 2) == [{"label": "teaching", "topic": ""}] * 2


def test_fmt_time():
    assert fmt_time(75) == "1:15"
    assert fmt_time(3725.9) == "1:02:05"
