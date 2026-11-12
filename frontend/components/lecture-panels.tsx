"use client";

import { useEffect, useState } from "react";
import {
  fmtTime,
  getAnnouncements,
  getExamHints,
  getQuiz,
  getSummary,
  type QuizQuestion,
  type Segment,
  type Summary,
  type Timeline,
} from "@/lib/api";

type Seek = (lectureId: number, startSec: number) => void;

function TimeChip({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="shrink-0 rounded bg-accent-soft px-1.5 py-0.5 font-mono text-xs text-accent hover:underline"
    >
      {label}
    </button>
  );
}

const CHAPTER_COLOR = {
  off_topic: "bg-stone-400/40",
  admin: "bg-sky-500/60",
  exam_hint: "bg-amber-500",
} as const;

/** Clickable bar of chapters under the video; off-topic stretches are greyed out. */
export function TimelineBar({
  timeline,
  position,
  onSeek,
}: {
  timeline: Timeline;
  position: number;
  onSeek: (startSec: number) => void;
}) {
  const { chapters } = timeline;
  if (chapters.length === 0) return null;
  const span = chapters[chapters.length - 1].end_sec - chapters[0].start_sec || 1;
  const active = chapters.find((c) => position >= Math.floor(c.start_sec) && position < c.end_sec);
  let teachingIndex = 0;
  return (
    <div className="border-b border-line px-4 py-2">
      <div className="flex h-3 w-full gap-px overflow-hidden rounded">
        {chapters.map((c, i) => {
          const color =
            c.label === "teaching"
              ? teachingIndex++ % 2
                ? "bg-accent/55"
                : "bg-accent/85"
              : CHAPTER_COLOR[c.label];
          return (
            <button
              key={i}
              onClick={() => onSeek(c.start_sec)}
              title={`${c.title} · ${fmtTime(c.start_sec)}`}
              aria-label={`${c.title}, starts at ${fmtTime(c.start_sec)}`}
              style={{ width: `${((c.end_sec - c.start_sec) / span) * 100}%` }}
              className={`h-full min-w-0.5 ${color} ${c === active ? "ring-2 ring-inset ring-foreground" : ""} hover:opacity-70`}
            />
          );
        })}
      </div>
      <div className="mt-1.5 flex justify-between gap-3 text-xs text-muted">
        <span className="truncate">{active ? active.title : "Hover or click the bar to jump between topics"}</span>
        <span className="shrink-0">
          {Math.round(timeline.teaching_sec / 60)} of {Math.round(timeline.total_sec / 60)} min teaching
        </span>
      </div>
    </div>
  );
}

export function SummaryPanel({ lectureId, onSeek }: { lectureId: number; onSeek: (startSec: number) => void }) {
  const [state, setState] = useState<{ id: number; summary?: Summary; error?: string } | null>(null);

  useEffect(() => {
    let stale = false;
    getSummary(lectureId)
      .then((summary) => !stale && setState({ id: lectureId, summary }))
      .catch((e: Error) => !stale && setState({ id: lectureId, error: e.message }));
    return () => {
      stale = true;
    };
  }, [lectureId]);

  const current = state?.id === lectureId ? state : null;
  if (!current) return <p className="p-4 text-muted">Summarising this lecture…</p>;
  if (current.error) return <p className="p-4 text-red-500">{current.error}</p>;
  const s = current.summary!;
  if (!s.summary) return <p className="p-4 text-muted">No teaching content found to summarise.</p>;
  return (
    <div className="space-y-4 p-4">
      <p className="leading-relaxed">{s.summary}</p>
      <ul className="space-y-2">
        {s.key_points.map((p, i) => (
          <li key={i} className="flex items-start gap-3">
            <TimeChip label={fmtTime(p.start_sec)} onClick={() => onSeek(p.start_sec)} />
            <span className="leading-5">{p.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function QuizPanel({ courseId, lectureId, onSeek }: { courseId: number; lectureId: number; onSeek: Seek }) {
  const [questions, setQuestions] = useState<QuizQuestion[] | null>(null);
  const [picked, setPicked] = useState<Record<number, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function generate() {
    setLoading(true);
    setError("");
    try {
      setQuestions(await getQuiz(courseId, [lectureId], 5));
      setPicked({});
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const answered = Object.keys(picked).length;
  const correct = questions?.filter((q, i) => picked[i] === q.answer_index).length ?? 0;

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-3">
        <button
          onClick={generate}
          disabled={loading}
          className="rounded-md bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50 dark:text-black"
        >
          {loading ? "Writing questions…" : questions ? "New questions" : "Generate 5 questions"}
        </button>
        {questions && answered > 0 && (
          <span className="text-muted">
            {correct} of {answered} correct
          </span>
        )}
        {error && <span className="text-red-500">{error}</span>}
      </div>
      {!questions && !loading && (
        <p className="text-muted">Questions are written only from what this lecture teaches.</p>
      )}
      {questions?.length === 0 && <p className="text-muted">No questions could be generated for this lecture.</p>}
      {questions?.map((q, i) => {
        const choice = picked[i];
        const done = choice !== undefined;
        return (
          <div key={i} className="rounded-lg border border-line bg-panel p-3">
            <p className="font-medium">
              {i + 1}. {q.question}
            </p>
            <div className="mt-2 space-y-1.5">
              {q.options.map((opt, j) => {
                const isAnswer = j === q.answer_index;
                const style = !done
                  ? "border-line hover:border-accent"
                  : isAnswer
                    ? "border-emerald-500 bg-emerald-500/10"
                    : j === choice
                      ? "border-red-500 bg-red-500/10"
                      : "border-line opacity-60";
                return (
                  <button
                    key={j}
                    disabled={done}
                    onClick={() => setPicked((p) => ({ ...p, [i]: j }))}
                    className={`block w-full rounded-md border px-3 py-1.5 text-left ${style}`}
                  >
                    {opt}
                    {done && isAnswer && <span className="ml-2 text-xs text-emerald-600">correct answer</span>}
                    {done && j === choice && !isAnswer && <span className="ml-2 text-xs text-red-500">your answer</span>}
                  </button>
                );
              })}
            </div>
            {done && (
              <div className="mt-2 flex items-start gap-3 text-muted">
                <TimeChip
                  label={`Watch · ${fmtTime(q.start_sec)}`}
                  onClick={() => onSeek(q.lecture_id, q.start_sec)}
                />
                <span className="leading-5">{q.explanation}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function NoteList({ title, empty, items, onSeek }: { title: string; empty: string; items: Segment[]; onSeek: Seek }) {
  return (
    <section>
      <h2 className="text-sm font-medium uppercase tracking-wide text-muted">{title}</h2>
      {items.length === 0 ? (
        <p className="mt-2 text-muted">{empty}</p>
      ) : (
        <ul className="mt-2 space-y-2">
          {items.map((s) => (
            <li key={s.id} className="flex items-start gap-3">
              <TimeChip
                label={`L${s.lecture_number} · ${fmtTime(s.start_sec)}`}
                onClick={() => onSeek(s.lecture_id, s.start_sec)}
              />
              <span className="line-clamp-3 leading-5">{s.text}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** Course-wide exam hints and announcements, shown when no lecture is open. */
export function CourseNotes({ courseId, onSeek }: { courseId: number; onSeek: Seek }) {
  const [notes, setNotes] = useState<{ hints: Segment[]; announcements: Segment[] } | null>(null);

  useEffect(() => {
    let stale = false;
    Promise.all([getExamHints(courseId), getAnnouncements(courseId)])
      .then(([hints, announcements]) => !stale && setNotes({ hints, announcements }))
      .catch(() => !stale && setNotes({ hints: [], announcements: [] }));
    return () => {
      stale = true;
    };
  }, [courseId]);

  return (
    <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-6 text-sm">
      <p className="text-muted">Pick a lecture on the left, or ask a question and click a timestamp.</p>
      {notes && (
        <>
          <NoteList
            title="Exam hints"
            empty="No moments flagged as important for exams yet."
            items={notes.hints}
            onSeek={onSeek}
          />
          <NoteList
            title="Announcements and course info"
            empty="No announcements found yet."
            items={notes.announcements}
            onSeek={onSeek}
          />
        </>
      )}
    </div>
  );
}
