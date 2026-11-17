"""Re-transcribe a course's cached audio with another Whisper model, for the Whisper ablation.

    backend/.venv/Scripts/python backend/scripts/transcribe_variant.py --course 1 --model medium

Writes data/courses/<course>/<lecture>/transcript-<model>.json next to the app's transcript.json,
which is left alone. Same settings as the app (VAD filter, WHISPER_TASK, WHISPER_DEVICE,
WHISPER_COMPUTE_TYPE); only the model size changes. Lectures already done are skipped, so an
interrupted run resumes. Then compare with: backend/scripts/ablate.py --only whisper
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from faster_whisper import WhisperModel  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=int, default=1)
    ap.add_argument("--model", required=True, help="e.g. medium, large-v3")
    args = ap.parse_args()

    model = WhisperModel(args.model, device=settings.whisper_device, compute_type=settings.whisper_compute_type)
    course_dir = settings.data_path / "courses" / str(args.course)
    for d in sorted(p for p in course_dir.iterdir() if p.is_dir()):
        out = d / f"transcript-{args.model}.json"
        audio = next(d.glob("audio.*"), None)
        if out.exists() or audio is None:
            continue
        t = time.perf_counter()
        segments, info = model.transcribe(str(audio), vad_filter=True, task=settings.whisper_task)
        result = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        took = time.perf_counter() - t
        print(f"lecture dir {d.name}: {len(result)} segments, {info.duration / 60:.1f} min audio in "
              f"{took / 60:.1f} min ({took / info.duration:.2f}x real time)", flush=True)


if __name__ == "__main__":
    main()
