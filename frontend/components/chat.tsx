"use client";

import { useEffect, useRef, useState } from "react";
import { ask, type Citation, fmtTime } from "@/lib/api";

type Message =
  | { role: "user"; text: string }
  | {
      role: "assistant";
      text: string;
      sources: Citation[];
      closest: Citation[]; // shown when the question isn't covered
      done: boolean;
      error?: string;
    };

function Chip({ c, onSeek }: { c: Citation; onSeek: (c: Citation) => void }) {
  return (
    <button
      onClick={() => onSeek(c)}
      title={c.lecture_title}
      className="mx-0.5 inline-flex items-center rounded bg-accent-soft px-1.5 py-0.5 align-baseline font-mono text-xs text-accent hover:underline"
    >
      L{c.lecture_number} · {fmtTime(c.start_sec)}
    </button>
  );
}

/** Renders answer text, turning [S#] tags into clickable timestamp chips. */
function AnswerText({ text, sources, onSeek }: { text: string; sources: Citation[]; onSeek: (c: Citation) => void }) {
  const byTag = new Map(sources.map((s) => [s.tag, s]));
  return (
    <p className="whitespace-pre-wrap leading-relaxed">
      {text.split(/(\[S\d+\])/g).map((part, i) => {
        const tag = /^\[(S\d+)\]$/.exec(part)?.[1];
        if (!tag) return part;
        const c = byTag.get(tag);
        return c ? <Chip key={i} c={c} onSeek={onSeek} /> : null;
      })}
    </p>
  );
}

export default function Chat({
  courseId,
  ready,
  onSeek,
}: {
  courseId: number;
  ready: boolean;
  onSeek: (c: Citation) => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function patchLast(patch: (m: Extract<Message, { role: "assistant" }>) => Partial<Message>) {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role !== "assistant") return prev;
      return [...prev.slice(0, -1), { ...last, ...patch(last) } as Message];
    });
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: "", sources: [], closest: [], done: false },
    ]);
    await ask(courseId, question, {
      sources: (sources) => patchLast(() => ({ sources })),
      token: (t) => patchLast((m) => ({ text: m.text + t })),
      final: (f) => patchLast(() => ({ text: f.answer, closest: f.supported ? [] : f.citations, done: true })),
      error: (message) => patchLast(() => ({ error: message, done: true })),
    });
    setBusy(false);
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 text-sm">
        {messages.length === 0 && (
          <p className="text-muted">
            {ready
              ? "Ask anything about these lectures. Answers come only from what was taught, with timestamps you can click."
              : "Questions will work once at least one lecture has finished processing."}
          </p>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="ml-8 rounded-lg bg-accent px-3 py-2 text-white dark:text-black">
              {m.text}
            </div>
          ) : (
            <div key={i} className="mr-4 rounded-lg border border-line bg-panel px-3 py-2">
              {m.error ? (
                <p className="text-red-500">{m.error}</p>
              ) : m.text ? (
                <AnswerText text={m.text} sources={m.sources} onSeek={onSeek} />
              ) : (
                <p className="text-muted">Searching your lectures…</p>
              )}
              {m.closest.length > 0 && (
                <div className="mt-2 border-t border-line pt-2 text-xs text-muted">
                  Closest moments:
                  {m.closest.map((c) => (
                    <Chip key={c.segment_id} c={c} onSeek={onSeek} />
                  ))}
                </div>
              )}
            </div>
          ),
        )}
        <div ref={bottom} />
      </div>
      <form onSubmit={submit} className="flex gap-2 border-t border-line p-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your lectures…"
          disabled={!ready}
          className="min-w-0 flex-1 rounded-md border border-line bg-panel px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-50"
        />
        <button
          disabled={!ready || busy || !input.trim()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black"
        >
          Ask
        </button>
      </form>
    </div>
  );
}
