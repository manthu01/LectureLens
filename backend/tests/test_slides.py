import numpy as np
import pytest

from app.ingestion.slides import align, extract, monotonic_path, save


def test_monotonic_path_follows_the_best_slides():
    sim = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.1], [0.1, 0.2, 0.9]])
    assert monotonic_path(sim) == [0, 1, 2]


def test_monotonic_path_ignores_a_one_off_backward_spike():
    # Raw 0..1 similarities here, so this also checks the defaults don't over-smooth clear changes.
    # Segment 2 resembles slide 0 slightly better (e.g. a recap word), but jumping back and
    # forth costs more than it gains, so the path stays on slide 1.
    sim = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.75, 0.7, 0.1], [0.1, 0.8, 0.2], [0.1, 0.2, 0.9]])
    assert monotonic_path(sim) == [0, 1, 1, 1, 2]


def test_monotonic_path_allows_a_real_return_to_an_earlier_slide():
    # A long stretch clearly about slide 0 again: worth paying the backward cost.
    # Values on the z-score scale the app feeds in (+-1 standard deviation).
    sim = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    assert monotonic_path(sim) == [0, 1, 0, 0, 0]


def test_monotonic_path_empty():
    assert monotonic_path(np.zeros((0, 3))) == []
    assert monotonic_path(np.zeros((3, 0))) == []


def _fake_embed(texts):
    """Bag-of-words over a tiny vocabulary, unit length."""
    vocab = ["batch", "online", "drift", "cost", "hello"]
    v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float) + 1e-3
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_align_links_segments_and_skips_off_topic_and_empty_slides():
    chunks = [
        {"start_sec": 0, "end_sec": 60, "text": "batch learning trains batch by batch", "label": "teaching"},
        {"start_sec": 60, "end_sec": 120, "text": "hello everyone like and subscribe", "label": "off_topic"},
        {"start_sec": 120, "end_sec": 180, "text": "online learning handles drift", "label": "teaching",
         "visual_text": "concept drift"},
    ]
    slides = [
        {"page": 1, "text": "Batch learning"},
        {"page": 2, "text": ""},  # image-only slide
        {"page": 3, "text": "Online learning and concept drift"},
    ]
    out = align(chunks, slides, _fake_embed)
    assert [c["slide_page"] for c in out] == [1, None, 3]
    assert out[2]["slide_text"] == "Online learning and concept drift"


def test_tfidf_similarity_prefers_shared_rare_words():
    from app.ingestion.slides import tfidf_similarity

    sim = tfidf_similarity(["the turing machine primitives"], ["Turing primitives", "Python programs", ""])
    assert sim.shape == (1, 3)
    assert sim[0, 0] > sim[0, 1] and sim[0, 2] == 0


def test_align_without_slides_text():
    out = align([{"start_sec": 0, "end_sec": 1, "text": "x"}], [{"page": 1, "text": ""}], _fake_embed)
    assert out[0]["slide_page"] is None


def test_pptx_extract_and_save(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    for title, body in [("Batch ML", "Train once on all data"), ("Online ML", "Learn as data arrives")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.slides.add_slide(prs.slide_layouts[6]).shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1))
    path = tmp_path / "deck.pptx"
    prs.save(path)

    deck = extract(path)
    assert [d["page"] for d in deck] == [1, 2, 3]
    assert "Batch ML" in deck[0]["text"] and "Train once" in deck[0]["text"]
    assert deck[2]["text"] == ""

    saved = save(tmp_path / "lec", "My Deck.PPTX", path.read_bytes())
    assert saved == deck
    assert (tmp_path / "lec" / "slides.pptx").exists() and (tmp_path / "lec" / "slides.json").exists()


def test_save_rejects_other_types(tmp_path):
    with pytest.raises(ValueError):
        save(tmp_path, "notes.docx", b"x")
