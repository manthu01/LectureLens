"""Assemble the folder to upload to a Hugging Face Docker Space.

    cd backend
    uv run python scripts/make_snapshot.py      # backend stopped
    uv run python scripts/build_space.py
    uv run hf upload <user>/<space> ../deploy/space . --repo-type space

Writes deploy/space/ at the project root (gitignored): the Dockerfile, the app,
the dependency lock, the data snapshot, and the README Spaces needs for its settings.
"""

import shutil
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND.parent / "deploy" / "space"

SPACE_README = """---
title: LectureLens API
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# LectureLens API

Read-only backend for [LectureLens](https://github.com/manthu01/LectureLens): search and
question answering over lecture transcripts, with timestamped citations.

The indexed lectures are from the "100 Days of Machine Learning" YouTube playlist by
**CampusX**. All teaching credit goes to CampusX. Transcripts are automatic English
translations and may contain errors; videos are played from YouTube, not hosted here.
"""


def main() -> None:
    snapshot = BACKEND / "snapshot"
    if not (snapshot / "lecturelens.db").exists():
        raise SystemExit("No snapshot found. Run scripts/make_snapshot.py first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in ("Dockerfile", "pyproject.toml", "uv.lock", ".dockerignore"):
        shutil.copy2(BACKEND / name, OUT / name)
    shutil.copytree(BACKEND / "app", OUT / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(snapshot, OUT / "snapshot", ignore=shutil.ignore_patterns(".lock"))
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Space bundle ready at {OUT}: {files} files, {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
