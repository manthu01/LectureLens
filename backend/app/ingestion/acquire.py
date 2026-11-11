from pathlib import Path

import yt_dlp

from app.config import settings


def _local_duration(path: Path) -> float:
    import av  # ships with faster-whisper

    with av.open(str(path)) as container:
        return float(container.duration or 0) / 1_000_000


def list_playlist(url: str, limit: int | None = None) -> list[dict]:
    """Return [{url, title, video_id, duration_sec}] for the videos in a playlist, in order."""
    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"deno": {}, "node": {}},
    }
    if limit:
        opts["playlistend"] = limit
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return [
        {
            "url": f"https://www.youtube.com/watch?v={e['id']}",
            "title": e.get("title") or "",
            "video_id": e["id"],
            "duration_sec": float(e.get("duration") or 0),
        }
        for e in (info.get("entries") or [])
        if e and e.get("id")
    ]


def download_audio(source: str, out_dir: Path) -> dict:
    """Resolve a lecture source to an audio file.

    `source` is either a path to a local media file (used in place) or a URL
    yt-dlp understands (audio track downloaded once into out_dir).
    """
    local = Path(source)
    if local.is_file():
        return {
            "audio_path": local,
            "title": local.stem,
            "duration_sec": _local_duration(local),
            "video_id": "",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.glob("audio.*") if p.suffix not in (".part", ".ytdl")]
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"deno": {}, "node": {}},
    }
    if settings.ytdlp_cookies_from_browser:
        opts["cookiesfrombrowser"] = (settings.ytdlp_cookies_from_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=not existing)
        path = existing[0] if existing else Path(ydl.prepare_filename(info))
    return {
        "audio_path": path,
        "title": info.get("title") or "",
        "duration_sec": float(info.get("duration") or 0),
        "video_id": info.get("id") or "",
    }
