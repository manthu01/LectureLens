import numpy as np

from app.generation.answer import build_prompt
from app.ingestion.visual.align import visual_text_for
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
        "facecam_only", 0.0)
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
