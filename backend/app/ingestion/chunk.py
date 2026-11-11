def chunk_transcript(
    segs: list[dict],
    target_sec: float = 60.0,
    min_sec: float = 45.0,
    max_sec: float = 90.0,
    pause_sec: float = 1.5,
    overlap_segs: int = 1,
) -> list[dict]:
    """Group ASR segments ({start, end, text}) into retrieval chunks.

    A chunk closes when it reaches target_sec, when adding the next segment
    would exceed max_sec, or at a pause > pause_sec once it is at least min_sec.
    The last `overlap_segs` segments are repeated at the start of the next chunk.
    """
    chunks: list[dict] = []
    cur: list[dict] = []

    def emit() -> None:
        chunks.append(
            {
                "start_sec": cur[0]["start"],
                "end_sec": cur[-1]["end"],
                "text": " ".join(s["text"] for s in cur),
            }
        )

    for s in segs:
        if cur:
            cur_dur = cur[-1]["end"] - cur[0]["start"]
            gap = s["start"] - cur[-1]["end"]
            if (
                cur_dur >= target_sec
                or s["end"] - cur[0]["start"] > max_sec
                or (cur_dur >= min_sec and gap > pause_sec)
            ):
                emit()
                cur = cur[-overlap_segs:] if overlap_segs else []
        cur.append(s)
    if cur:
        emit()
    return chunks
