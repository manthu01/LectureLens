import numpy as np
import pytest

from app.generation.answer import build_prompt
from app.generation.llm import Blocked
from app.ingestion.visual.align import visual_text_for
from app.ingestion.visual.enhance import OUT_WIDTH, content_box, enhance
from app.ingestion.visual.frames import SMALL, select
from app.ingestion.visual.read import lecture_layout, parse_reads
from app.retrieval.store import embed_text

W, H = SMALL


def _page(lines: int, seed: int = 0) -> np.ndarray:
    """White page with `lines` dark text lines; a different seed gives a different page."""
    g = np.full((H, W), 240, dtype=np.int16)
    for k in range(lines):
        y = 20 + 12 * k + seed
        g[y : y + 3, 20 : 200 + (seed * 7) % 60] = 20
    return g


def test_select_keeps_page_before_turn_and_growing_writing():
    # page A gains lines over 0..50 s, page turn at 60 s, page B gains lines to the end.
    greys = [_page(n) for n in (1, 2, 4, 6, 8, 10)] + [_page(n, seed=5) for n in (1, 3, 5, 7)]
    times = [10.0 * i for i in range(len(greys))]
    kept = select(times, greys)
    assert 5 in kept  # page A at its fullest, just before the turn
    assert kept[-1] == len(greys) - 1  # page B at the end
    assert all(b - a >= 1 for a, b in zip(kept, kept[1:]))


def test_select_static_screen_keeps_one_frame():
    greys = [_page(3)] * 20
    assert select([5.0 * i for i in range(20)], greys) == [19]


def test_select_masks_live_video_region():
    # Content stays the same; a corner keeps changing like a face cam. Nothing new to read.
    rng = np.random.default_rng(0)
    greys = []
    for _ in range(20):
        g = _page(3)
        g[H - 50 :, W - 80 :] = rng.integers(0, 255, (50, 80))
        greys.append(g)
    assert select([5.0 * i for i in range(20)], greys) == [19]


def test_parse_reads_normalises():
    items = [
        {"i": 1, "layout": "board_close", "text": " x = 1 ", "readable": 1.7},
        {"i": 0, "layout": "nonsense", "text": "a", "readable": "bad"},
        {"i": 9, "layout": "screen_only", "text": "out of range", "readable": 1},
    ]
    out = parse_reads(items, 3)
    assert out[0] == {"layout": "unknown", "text": "a", "readable": 0.0}
    assert out[1] == {"layout": "board_close", "text": "x = 1", "readable": 1.0}
    assert out[2]["layout"] == "unknown" and out[2]["text"] == ""


def test_lecture_layout_ignores_short_talking_head():
    reads = [
        {"start_sec": 0, "end_sec": 60, "layout": "facecam_only", "text": "", "readable": 0},
        {"start_sec": 60, "end_sec": 600, "layout": "facecam_screen", "text": "t", "readable": 0.9},
        {"start_sec": 600, "end_sec": 700, "layout": "screen_only", "text": "t", "readable": 0.5},
    ]
    layout, quality = lecture_layout(reads)
    assert layout == "facecam_screen"
    assert 0.5 < quality < 0.9
    assert lecture_layout([{"start_sec": 0, "end_sec": 9, "layout": "facecam_only", "text": "", "readable": 0}]) == (
        "facecam_only", None)
    # Mostly talking head with one slide: still a talking-head lecture.
    mostly_cam = [
        {"start_sec": 0, "end_sec": 500, "layout": "facecam_only", "text": "", "readable": 0},
        {"start_sec": 500, "end_sec": 530, "layout": "screen_only", "text": "slide", "readable": 1},
    ]
    assert lecture_layout(mostly_cam) == ("facecam_only", 1.0)


def test_visual_text_overlap_and_superset():
    reads = [
        {"start_sec": 0, "end_sec": 50, "text": "Title"},
        {"start_sec": 50, "end_sec": 100, "text": "Title\n1. Lots of data"},
        {"start_sec": 100, "end_sec": 150, "text": "Next page"},
        {"start_sec": 150, "end_sec": 200, "text": ""},
    ]
    assert visual_text_for(10, 40, reads) == "Title"
    assert visual_text_for(40, 90, reads) == "Title\n1. Lots of data"  # the fuller version only
    assert visual_text_for(90, 160, reads) == "Title\n1. Lots of data\nNext page"
    assert visual_text_for(160, 190, reads) == ""


def test_on_screen_text_reaches_index_and_prompt():
    c = {"topic": "Batch ML", "text": "spoken words", "visual_text": "1. Lots of Data"}
    assert embed_text(c) == "Batch ML. spoken words On screen: 1. Lots of Data"
    assert embed_text({"text": "only speech"}) == "only speech"
    hit = {"lecture_number": 4, "lecture_title": "T", "start_sec": 0, "end_sec": 60, **c}
    assert "On screen: 1. Lots of Data" in build_prompt("q?", [hit])


def test_answer_cache_replays_and_invalidates(monkeypatch):
    from app.generation import answer
    from app.retrieval import store

    calls = []

    def fake(course_id, question, include_off_topic=False):
        calls.append(question)
        yield answer._sse("sources", {"sources": []})
        yield answer._sse("token", {"text": "Hel"})
        yield answer._sse("token", {"text": "lo"})
        yield answer._sse("final", {"answer": "Hello", "citations": [], "supported": True})

    monkeypatch.setattr(answer, "_answer_stream", fake)
    answer._cache.clear()
    first = list(answer.answer_stream(1, "What is ML?"))
    again = list(answer.answer_stream(1, "what is ml"))  # same up to case and punctuation
    assert len(calls) == 1
    assert again[-1] == first[-1] and '"Hello"' in again[1]  # tokens merged into one event

    monkeypatch.setattr(store, "index_version", store.index_version + 1)  # a lecture was re-indexed
    list(answer.answer_stream(1, "What is ML?"))
    assert len(calls) == 2

    def failing(course_id, question, include_off_topic=False):
        calls.append(question)
        yield answer._sse("error", {"message": "quota"})

    monkeypatch.setattr(answer, "_answer_stream", failing)
    list(answer.answer_stream(1, "new question"))
    list(answer.answer_stream(1, "new question"))
    assert calls[-2:] == ["new question", "new question"]  # errors are not cached
    answer._cache.clear()


def _far_board(seed: int = 0) -> np.ndarray:
    """A dark, noisy 1280x720 classroom with a small bright board holding dark writing."""
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(35, 8, (720, 1280)), 0, 255)
    g[250:450, 500:820] = 215
    for k in range(6):
        g[275 + 28 * k : 280 + 28 * k, 530 : 700 + 15 * k] = 40
    return g.astype(np.int16)


def _save_jpeg(g: np.ndarray, path):
    from PIL import Image

    Image.fromarray(g.astype(np.uint8)).save(path)


def test_content_box_finds_far_board():
    box = content_box(_far_board())
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 500 and y0 <= 250 and x1 >= 820 and y1 >= 450  # the whole board
    assert x0 >= 400 and y0 >= 150 and x1 <= 920 and y1 <= 550  # and not much of the room


def test_content_box_none_without_a_region():
    rng = np.random.default_rng(1)
    assert content_box(np.clip(rng.normal(35, 8, (720, 1280)), 0, 255).astype(np.int16)) is None  # dark room
    assert content_box(np.full((720, 1280), 240, dtype=np.int16)) is None  # blank page
    full = np.full((720, 1280), 240, dtype=np.int16)
    for y in range(40, 690, 25):  # writing all over the frame: a crop gains nothing
        full[y : y + 4, 40:1240] = 20
    assert content_box(full) is None


def test_enhance_upscales_crop():
    from PIL import Image

    out = enhance(Image.fromarray(_far_board().astype(np.uint8)), (480, 230, 840, 470))
    assert out.width == OUT_WIDTH and abs(out.height - round(240 * OUT_WIDTH / 360)) <= 1


def test_read_frames_enhances_hard_frames_once(tmp_path, monkeypatch):
    from app.ingestion.visual import read

    _save_jpeg(_far_board(), tmp_path / "a.jpg")  # far board
    _save_jpeg(_far_board(2), tmp_path / "b.jpg")  # far board whose crop the model refuses
    _save_jpeg(np.full((720, 1280), 240), tmp_path / "c.jpg")  # low readability, nothing to crop
    _save_jpeg(np.full((720, 1280), 240), tmp_path / "d.jpg")  # clean screen
    frames = [{"file": f"{n}.jpg", "start_sec": 10.0 * i, "end_sec": 10.0 * i + 10} for i, n in enumerate("abcd")]
    calls = []

    def fake(system, prompt, schema, images=(), model=""):
        calls.append(len(images))
        if len(calls) == 1:  # first read: all four frames in one call
            return [
                {"i": 0, "layout": "board_far", "text": "blur", "readable": 0.2},
                {"i": 1, "layout": "board_far", "text": "blur", "readable": 0.4},
                {"i": 2, "layout": "board_close", "text": "", "readable": 0.1},
                {"i": 3, "layout": "screen_only", "text": "Slide", "readable": 0.95},
            ]
        if len(calls) == 3:  # crop of a, alone after the batch of two crops was refused
            return [{"i": 0, "layout": "board_close", "text": "x = 1", "readable": 0.8}]
        raise Blocked("RECITATION")

    monkeypatch.setattr(read, "complete_json", fake)
    out = {r["file"]: r for r in read.read_frames(frames, tmp_path)}
    assert calls == [4, 2, 1, 1]
    assert out["a.jpg"] == {**frames[0], "layout": "board_far", "text": "x = 1", "readable": 0.8, "enhanced": True}
    assert out["b.jpg"]["text"] == "blur" and out["b.jpg"]["enhanced"] is False and "blocked" not in out["b.jpg"]
    assert out["c.jpg"]["enhanced"] is False  # tried, but no content region, so no call
    assert "enhanced" not in out["d.jpg"]

    def no_call(*a, **k):
        raise AssertionError("cached and already-enhanced frames must not be read again")

    monkeypatch.setattr(read, "complete_json", no_call)
    assert {r["file"]: r for r in read.read_frames(frames, tmp_path)} == out


def test_rate_limit_in_second_read_propagates_and_resumes(tmp_path, monkeypatch):
    from app.ingestion.visual import read

    _save_jpeg(_far_board(), tmp_path / "a.jpg")
    frames = [{"file": "a.jpg", "start_sec": 0.0, "end_sec": 10.0}]
    calls = []

    def fake(system, prompt, schema, images=(), model=""):
        calls.append(1)
        if len(calls) == 1:
            return [{"i": 0, "layout": "board_far", "text": "blur", "readable": 0.2}]
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(read, "complete_json", fake)
    with pytest.raises(RuntimeError):
        read.read_frames(frames, tmp_path)
    # The first read was saved; the rerun only makes the enhanced read.
    calls.clear()
    monkeypatch.setattr(read, "complete_json", lambda *a, **k: calls.append(1) or [
        {"i": 0, "layout": "board_close", "text": "y", "readable": 0.7}])
    assert read.read_frames(frames, tmp_path)[0]["text"] == "y" and len(calls) == 1


def test_visual_text_skips_unreadable_frames():
    reads = [
        {"start_sec": 0, "end_sec": 50, "text": "guessed scribble", "readable": 0.2},
        {"start_sec": 0, "end_sec": 50, "text": "Clear title", "readable": 0.9},
    ]
    assert visual_text_for(10, 40, reads) == "Clear title"
