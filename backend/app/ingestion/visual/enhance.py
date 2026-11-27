"""Crop a frame to its content and upscale it, for a second read of a hard-to-read frame.

A board filmed from the back of the class fills a small part of the frame; sent whole (and
downscaled to the vision model's working size) its writing is a few pixels tall. Cropping to the
region that holds the writing and upscaling it gives the model several times more pixels per
letter. Pure NumPy/PIL: no OpenCV, no super-resolution model, so it runs anywhere the rest of the
pipeline does. Untested on real board footage (the demo courses have none).
"""

import io
from pathlib import Path

import numpy as np

WORK_WIDTH = 320  # analysis resolution; the box is scaled back to the full frame
BLUR = 8  # px at WORK_WIDTH; the local background is a box blur this wide
INK = 40  # a pixel this far from its local background is ink (strokes, letters, board edges)
CELL = 8  # px at WORK_WIDTH; content is found per cell, so a stray speck can't stretch the box
CELL_INK = 0.04  # share of ink pixels that makes a cell hold content
BORDER = 0.02  # share of each side ignored: letterboxing, burnt-in logos and timestamps
PAD = 0.04  # share of the frame added around the box, so edge letters aren't cut
MIN_SIDE = 0.12  # a box narrower/shorter than this share of the frame is noise, not a board
MAX_AREA = 0.8  # a box covering more of the frame than this gains nothing from cropping
OUT_WIDTH = 1600


def content_box(grey: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the region holding ink in a greyscale frame, or None when there is no
    clear region (blank frame, or ink all over it so a crop would gain nothing).

    Ink is measured against the local background (a blur), not the frame's median, so dark
    strokes on a bright board inside a dark room count, and the room's sensor noise does not.
    """
    from PIL import Image, ImageFilter

    h, w = grey.shape
    small_h = max(round(h * WORK_WIDTH / w), 1)
    img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8)).resize((WORK_WIDTH, small_h))
    g = np.asarray(img, dtype=np.int16)
    bg = np.asarray(img.filter(ImageFilter.BoxBlur(BLUR)), dtype=np.int16)
    ink = np.abs(g - bg) > INK
    by, bx = max(round(small_h * BORDER), 1), max(round(WORK_WIDTH * BORDER), 1)
    ink[:by], ink[-by:], ink[:, :bx], ink[:, -bx:] = False, False, False, False

    cells = _content_cells(ink)
    if not cells.any():
        return None
    rows, cols = np.nonzero(cells)
    sy, sx = h / small_h, w / WORK_WIDTH
    x0, x1 = cols.min() * CELL * sx, (cols.max() + 1) * CELL * sx
    y0, y1 = rows.min() * CELL * sy, (rows.max() + 1) * CELL * sy
    px, py = PAD * w, PAD * h
    box = (max(int(x0 - px), 0), max(int(y0 - py), 0), min(int(x1 + px), w), min(int(y1 + py), h))
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw < MIN_SIDE * w or bh < MIN_SIDE * h or bw * bh > MAX_AREA * w * h:
        return None
    return box


def _content_cells(ink: np.ndarray) -> np.ndarray:
    """Cells dense enough in ink, minus isolated ones (a lone bright spot, a light fixture):
    writing spans several neighbouring cells."""
    h, w = ink.shape
    nh, nw = h // CELL, w // CELL
    dense = ink[: nh * CELL, : nw * CELL].reshape(nh, CELL, nw, CELL).mean(axis=(1, 3)) >= CELL_INK
    padded = np.pad(dense, 1)
    neighbours = sum(
        padded[1 + dy : 1 + dy + nh, 1 + dx : 1 + dx + nw]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    return dense & (neighbours > 0)


def enhance(img, box: tuple[int, int, int, int]):
    """Crop a PIL image to `box`, upscale to OUT_WIDTH with LANCZOS, then stretch contrast and
    sharpen a little: a far board is low-contrast and soft, and upscaling softens it further."""
    from PIL import Image, ImageFilter, ImageOps

    crop = img.convert("RGB").crop(box)
    if crop.width < OUT_WIDTH:
        crop = crop.resize((OUT_WIDTH, round(crop.height * OUT_WIDTH / crop.width)), Image.LANCZOS)
    crop = ImageOps.autocontrast(crop, cutoff=1)
    return crop.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=3))


def enhanced_jpeg(path: Path) -> bytes | None:
    """JPEG bytes of the cropped, upscaled frame, or None if it has no croppable content region."""
    from PIL import Image

    with Image.open(path) as im:
        box = content_box(np.asarray(im.convert("L"), dtype=np.int16))
        if box is None:
            return None
        buf = io.BytesIO()
        enhance(im, box).save(buf, "JPEG", quality=90)
        return buf.getvalue()
