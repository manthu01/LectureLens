"use client";

import { useEffect, useRef } from "react";

// Minimal typing for the parts of the YouTube IFrame API used here.
type YTPlayer = {
  seekTo: (sec: number, allowSeekAhead: boolean) => void;
  loadVideoById: (o: { videoId: string; startSeconds: number }) => void;
  playVideo: () => void;
  getCurrentTime: () => number;
  getVideoData?: () => { video_id?: string };
  destroy: () => void;
};
type YTNamespace = {
  Player: new (
    el: HTMLElement,
    o: { videoId: string; playerVars: Record<string, number>; events: { onReady: () => void } },
  ) => YTPlayer;
};
declare global {
  interface Window {
    YT?: YTNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}

let apiPromise: Promise<YTNamespace> | null = null;
function loadApi(): Promise<YTNamespace> {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  apiPromise ??= new Promise((resolve) => {
    const previous = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      previous?.();
      resolve(window.YT!);
    };
    const s = document.createElement("script");
    s.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(s);
  });
  return apiPromise;
}

export type Range = { start: number; end: number };

/**
 * YouTube player that seeks in place (no reload) when `start`/`nonce` change, reports the playback
 * position every half second, jumps over `skip` ranges, and calls `onStop` once when it reaches `stopAt`.
 */
export default function YouTubePlayer({
  videoId,
  start,
  nonce,
  onTime,
  skip = [],
  stopAt = null,
  onStop,
}: {
  videoId: string;
  start: number;
  nonce: number;
  onTime?: (sec: number) => void;
  skip?: Range[];
  stopAt?: number | null;
  onStop?: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const player = useRef<YTPlayer | null>(null);
  const ready = useRef(false);
  const loaded = useRef<string | null>(null); // video currently in the player
  // Latest props for the polling loop, which is set up once.
  const latest = useRef({ onTime, skip, stopAt, onStop, start });
  useEffect(() => {
    latest.current = { onTime, skip, stopAt, onStop, start };
  });
  const stopped = useRef(false);
  // Right after a seek the player can still report the old position; only honour `stopAt` once
  // playback has actually been seen inside [start, stopAt).
  const armed = useRef(false);

  // Create the player once.
  useEffect(() => {
    let cancelled = false;
    const el = document.createElement("div");
    host.current?.appendChild(el);
    loadApi().then((YT) => {
      if (cancelled) return;
      player.current = new YT.Player(el, {
        videoId,
        playerVars: { start: Math.floor(start), autoplay: 1, rel: 0 },
        events: {
          onReady: () => {
            ready.current = true;
          },
        },
      });
      loaded.current = videoId;
    });
    const poll = setInterval(() => {
      const p = player.current;
      if (!p || !ready.current) return;
      const t = p.getCurrentTime();
      const { onTime, skip, stopAt, onStop, start } = latest.current;
      onTime?.(t);
      const r = skip.find((r) => t >= r.start && t < r.end - 0.5);
      if (r) p.seekTo(r.end, true);
      if (stopAt != null && t >= start - 1 && t < stopAt) armed.current = true;
      if (stopAt != null && armed.current && t >= stopAt && !stopped.current) {
        stopped.current = true;
        onStop?.();
      }
    }, 500);
    return () => {
      cancelled = true;
      clearInterval(poll);
      player.current?.destroy();
      player.current = null;
      ready.current = false;
    };
    // Created once; later videoId/start changes are applied below without rebuilding the player.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Seek (or switch video) when asked to, without reloading the iframe.
  useEffect(() => {
    stopped.current = false;
    armed.current = false;
    const p = player.current;
    if (!p || !ready.current) return;
    if (loaded.current !== videoId) {
      p.loadVideoById({ videoId, startSeconds: Math.floor(start) });
      loaded.current = videoId;
    } else {
      p.seekTo(start, true);
      p.playVideo();
    }
  }, [videoId, start, nonce]);

  return <div ref={host} className="h-full w-full [&>iframe]:h-full [&>iframe]:w-full" />;
}
