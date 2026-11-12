"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import Chat from "@/components/chat";
import {
  addSource,
  type CourseDetail,
  fmtTime,
  getConfig,
  getCourse,
  getTimeline,
  getTranscript,
  type Lecture,
  retryLecture,
  type Segment,
  type Timeline,
} from "@/lib/api";
import { CourseNotes, QuizPanel, SummaryPanel, TimelineBar } from "@/components/lecture-panels";

const STATUS_STYLE: Record<Lecture["status"], string> = {
  queued: "text-muted",
  downloading: "text-amber-500",
  transcribing: "text-amber-500",
  indexing: "text-amber-500",
  indexed: "text-emerald-500",
  failed: "text-red-500",
};

const LABEL_STYLE: Record<Exclude<Segment["label"], "teaching">, string> = {
  off_topic: "bg-stone-500/15 text-muted",
  admin: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
  exam_hint: "bg-amber-500/20 text-amber-700 dark:text-amber-400",
};

const isBusy =(l: Lecture) => l.status !== "indexed" && l.status !== "failed";

export default function CourseView() {
  const courseId = Number(useParams<{ id: string }>().id);
  const [course, setCourse] = useState<CourseDetail | null>(null);
  const [error, setError] = useState("");
  // What the player shows. `nonce` forces a reload so clicking the same timestamp twice still seeks.
  const [current, setCurrent] = useState<{ lectureId: number; start: number; nonce: number } | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [timeline, setTimeline] = useState<{ lectureId: number; data: Timeline } | null>(null);
  const [tab, setTab] = useState<"transcript" | "summary" | "quiz">("transcript");
  const [url, setUrl] = useState("");
  const [limit, setLimit] = useState("");
  const [adding, setAdding] = useState(false);

  const refresh = useCallback(
    () =>
      getCourse(courseId)
        .then((c) => {
          setCourse(c);
          setError("");
        })
        .catch((e: Error) => setError(e.message)),
    [courseId],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  // The public demo is read-only: hide the controls for adding lectures.
  // null = not known yet; keep them hidden until the backend says otherwise.
  const [readOnly, setReadOnly] = useState<boolean | null>(null);
  useEffect(() => {
    getConfig()
      .then((c) => setReadOnly(c.read_only))
      .catch(() => setReadOnly(false));
  }, []);

  // Poll while anything is still processing.
  const processing = course?.lectures.some(isBusy) ?? false;
  useEffect(() => {
    if (!processing) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [processing, refresh]);

  const lecture = course?.lectures.find((l) => l.id === current?.lectureId) ?? null;
  const lectureId = lecture?.id;
  const lectureIndexed = lecture?.status === "indexed";
  useEffect(() => {
    if (!lectureId || !lectureIndexed) return;
    let stale = false;
    getTranscript(lectureId)
      .then((s) => !stale && setSegments(s))
      .catch(() => !stale && setSegments([]));
    getTimeline(lectureId)
      .then((data) => !stale && setTimeline({ lectureId, data }))
      .catch(() => {});
    return () => {
      stale = true;
    };
  }, [lectureId, lectureIndexed]);

  const seek = (id: number, start: number) =>
    setCurrent((prev) => ({ lectureId: id, start: Math.floor(start), nonce: (prev?.nonce ?? 0) + 1 }));

  async function add(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setAdding(true);
    try {
      await addSource(courseId, url.trim(), Number(limit) || undefined);
      setUrl("");
      setLimit("");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAdding(false);
    }
  }

  if (!course) return <p className="p-6 text-sm text-muted">{error || "Loading…"}</p>;

  const shownSegments = lectureIndexed ? segments.filter((s) => s.lecture_id === lectureId) : [];
  const isPlaylist = /[?&]list=/.test(url);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-baseline gap-3 border-b border-line px-4 py-3">
        <Link href="/" className="text-sm text-muted hover:text-foreground">
          ← Courses
        </Link>
        <h1 className="font-semibold">{course.name}</h1>
        {course.professor && <span className="text-sm text-muted">{course.professor}</span>}
        {lecture && (
          <button onClick={() => setCurrent(null)} className="text-sm text-accent hover:underline">
            Exam hints & announcements
          </button>
        )}
        {error &&<span className="ml-auto text-sm text-red-500">{error}</span>}
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)_400px]">
        {/* Lectures */}
        <aside className="flex min-h-0 flex-col border-b border-line lg:border-b-0 lg:border-r">
          <ul className="max-h-56 min-h-0 flex-1 overflow-y-auto lg:max-h-none">
            {course.lectures.map((l) => (
              <li key={l.id}>
                <button
                  onClick={() => seek(l.id, 0)}
                  className={`block w-full border-b border-line px-3 py-2 text-left text-sm hover:bg-accent-soft ${
                    l.id === lectureId ? "bg-accent-soft" : ""
                  }`}
                >
                  <span className="line-clamp-2">
                    <span className="text-muted">{l.number}.</span> {l.title || l.source_url}
                  </span>
                  <span className="mt-0.5 flex gap-2 text-xs">
                    <span className={STATUS_STYLE[l.status]}>{l.status}</span>
                    {l.duration_sec > 0 && <span className="text-muted">{fmtTime(l.duration_sec)}</span>}
                  </span>
                </button>
              </li>
            ))}
            {course.lectures.length === 0 && <li className="p-3 text-sm text-muted">No lectures yet.</li>}
          </ul>
          <form onSubmit={add} className={`space-y-2 border-t border-line p-3 ${readOnly !== false ? "hidden" : ""}`}>
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="YouTube video or playlist URL"
              className="w-full rounded-md border border-line bg-panel px-2 py-1.5 text-sm outline-none focus:border-accent"
            />
            <div className="flex gap-2">
              {isPlaylist && (
                <input
                  value={limit}
                  onChange={(e) => setLimit(e.target.value.replace(/\D/g, ""))}
                  placeholder="First N (all)"
                  className="w-28 rounded-md border border-line bg-panel px-2 py-1.5 text-sm outline-none focus:border-accent"
                />
              )}
              <button
                disabled={adding || !url.trim()}
                className="flex-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 dark:text-black"
              >
                {adding ? "Adding…" : isPlaylist ? "Add playlist" : "Add lecture"}
              </button>
            </div>
          </form>
        </aside>

        {/* Player + transcript */}
        <section className="flex min-h-0 flex-col border-b border-line lg:border-b-0 lg:border-r">
          {!lecture ? (
            <CourseNotes courseId={courseId} description={course.description} onSeek={seek} />
          ) : (
            <>
              <div className="aspect-video w-full shrink-0 bg-black lg:max-h-[55vh]">
                {lecture.video_id ? (
                  <iframe
                    key={`${lecture.id}-${current!.start}-${current!.nonce}`}
                    className="h-full w-full"
                    src={`https://www.youtube.com/embed/${lecture.video_id}?start=${current!.start}&autoplay=1&rel=0`}
                    title={lecture.title}
                    allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
                    allowFullScreen
                  />
                ) : (
                  <p className="p-6 text-sm text-stone-300">
                    No video for this lecture (added from a local file). Jump point: {fmtTime(current!.start)}
                  </p>
                )}
              </div>
              {lectureIndexed && timeline?.lectureId === lecture.id && (
                <TimelineBar
                  timeline={timeline.data}
                  position={current!.start}
                  onSeek={(start) => seek(lecture.id, start)}
                />
              )}
              <div className="flex items-center gap-4 border-b border-line px-4 py-2 text-sm">
                <span className="min-w-0 flex-1 truncate font-medium">
                  {lecture.number}. {lecture.title}
                </span>
                {lectureIndexed &&
                  (["transcript", "summary", "quiz"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTab(t)}
                      className={`capitalize ${tab === t ? "font-medium text-accent" : "text-muted hover:text-foreground"}`}
                    >
                      {t}
                    </button>
                  ))}
              </div>
              {lectureIndexed && tab === "summary" && (
                <div className="min-h-0 flex-1 overflow-y-auto text-sm">
                  <SummaryPanel lectureId={lecture.id} onSeek={(start) => seek(lecture.id, start)} />
                </div>
              )}
              {lectureIndexed && tab === "quiz" && (
                <div className="min-h-0 flex-1 overflow-y-auto text-sm">
                  <QuizPanel key={lecture.id} courseId={courseId} lectureId={lecture.id} onSeek={seek} />
                </div>
              )}
              <div
                className={`min-h-0 flex-1 overflow-y-auto p-2 text-sm ${
                  lectureIndexed && tab !== "transcript" ? "hidden" : ""
                }`}
              >
                {lecture.status === "failed" && (
                  <div className="p-2">
                    <p className="text-red-500">Processing failed: {lecture.error}</p>
                    <button
                      onClick={() => retryLecture(lecture.id).then(refresh)}
                      className="mt-2 rounded-md border border-line px-3 py-1 hover:border-accent"
                    >
                      Retry
                    </button>
                  </div>
                )}
                {isBusy(lecture) && (
                  <p className="p-2 text-muted">Transcript appears when processing finishes ({lecture.status}).</p>
                )}
                {shownSegments.map((s) => {
                  const active = current!.start >= Math.floor(s.start_sec) && current!.start < s.end_sec;
                  return (
                    <button
                      key={s.id}
                      onClick={() => seek(lecture.id, s.start_sec)}
                      className={`flex w-full gap-3 rounded px-2 py-1.5 text-left hover:bg-accent-soft ${
                        active ? "bg-accent-soft" : ""
                      }`}
                    >
                      <span className="w-12 shrink-0 font-mono text-xs leading-5 text-accent">
                        {fmtTime(s.start_sec)}
                      </span>
                      <span className={`leading-5 ${s.label === "off_topic" ? "text-muted" : ""}`}>
                        {s.label !== "teaching" && (
                          <span className={`mr-2 rounded px-1.5 py-0.5 text-xs ${LABEL_STYLE[s.label]}`}>
                            {s.label.replace("_", " ")}
                          </span>
                        )}
                        {s.topic && <span className="mr-2 font-medium">{s.topic}:</span>}
                        {s.text}
                      </span>
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </section>

        {/* Chat */}
        <section className="min-h-[24rem] lg:min-h-0">
          <Chat
            courseId={courseId}
            ready={course.lectures.some((l) => l.status === "indexed")}
            onSeek={(c) => seek(c.lecture_id, c.start_sec)}
          />
        </section>
      </div>
    </div>
  );
}
