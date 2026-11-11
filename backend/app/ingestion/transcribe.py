import json
from functools import lru_cache
from pathlib import Path

from app.config import settings


@lru_cache
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe(audio_path: Path, out_path: Path) -> list[dict]:
    """Returns [{start, end, text}]. Cached at out_path so reruns are free."""
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    # faster-whisper decodes via PyAV, so no system ffmpeg is needed.
    segments, _info = _model().transcribe(str(audio_path), vad_filter=True, task=settings.whisper_task)
    result = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
