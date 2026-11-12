"""Copy the local index and database into backend/snapshot/ for the read-only deployment.

    cd backend
    uv run python scripts/make_snapshot.py

Stop the backend first: the embedded Qdrant folder can only be opened by one process.
The Docker image copies backend/snapshot/ in as its data directory.
"""

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient, models  # noqa: E402

from app.config import settings  # noqa: E402
from app.retrieval.store import COLLECTION, create_collection  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "snapshot"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    # Database: use SQLite's backup API so the copy is consistent.
    src_db = sqlite3.connect(settings.data_path / "lecturelens.db")
    dst_db = sqlite3.connect(OUT / "lecturelens.db")
    with dst_db:
        src_db.backup(dst_db)
    lectures = dst_db.execute("SELECT count(*) FROM lecture WHERE status = 'indexed'").fetchone()[0]
    missing = dst_db.execute("SELECT count(*) FROM lecture WHERE status = 'indexed' AND summary = ''").fetchone()[0]
    src_db.close()
    dst_db.close()

    # Index: copy just the current collection into a fresh embedded Qdrant.
    src = QdrantClient(path=str(settings.data_path / "qdrant"))
    dst = QdrantClient(path=str(OUT / "qdrant"))
    dim = src.get_collection(COLLECTION).config.params.vectors["dense"].size
    create_collection(dst, dim)
    count, offset = 0, None
    while True:
        points, offset = src.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=True)
        if points:
            dst.upsert(
                COLLECTION,
                points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            )
            count += len(points)
        if offset is None:
            break
    src.close()
    dst.close()

    print(f"Snapshot written to {OUT}: {lectures} indexed lectures, {count} segments")
    if missing:
        print(f"Note: {missing} lecture(s) have no cached summary; they will be generated on first view.")


if __name__ == "__main__":
    main()
