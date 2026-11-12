"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { type Course, createCourse, getConfig, listCourses } from "@/lib/api";

export default function Home() {
  const [courses, setCourses] = useState<Course[] | null>(null);
  const [error, setError] = useState("");
  const [name, setName] = useState("");
  const [professor, setProfessor] = useState("");
  const [saving, setSaving] = useState(false);

  const [readOnly, setReadOnly] = useState(false);

  useEffect(() => {
    getConfig()
      .then((c) => setReadOnly(c.read_only))
      .catch(() => {});
    listCourses()
      .then(setCourses)
      .catch(() => setError("Could not reach the backend at localhost:8000. Is it running?"));
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    try {
      const c = await createCourse(name.trim(), professor.trim());
      setCourses((prev) => [...(prev ?? []), c]);
      setName("");
      setProfessor("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-2xl px-4 py-12">
      <h1 className="text-3xl font-semibold tracking-tight">LectureLens</h1>
      <p className="mt-1 text-muted">Ask a question, jump to the exact second it was explained.</p>

      {error && <p className="mt-6 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm">{error}</p>}

      <h2 className="mt-10 text-sm font-medium uppercase tracking-wide text-muted">Your courses</h2>
      <ul className="mt-3 space-y-2">
        {courses?.map((c) => (
          <li key={c.id}>
            <Link
              href={`/courses/${c.id}`}
              className="block rounded-lg border border-line bg-panel px-4 py-3 transition hover:border-accent"
            >
              <span className="font-medium">{c.name}</span>
              {c.professor && <span className="ml-2 text-sm text-muted">{c.professor}</span>}
            </Link>
          </li>
        ))}
        {courses?.length === 0 && <li className="text-sm text-muted">No courses yet. Create one below.</li>}
        {!courses && !error && <li className="text-sm text-muted">Loading…</li>}
      </ul>

      <form onSubmit={submit} className={`mt-8 flex flex-col gap-2 sm:flex-row ${readOnly ? "hidden" : ""}`}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Course name"
          className="flex-1 rounded-md border border-line bg-panel px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <input
          value={professor}
          onChange={(e) => setProfessor(e.target.value)}
          placeholder="Professor / channel (optional)"
          className="flex-1 rounded-md border border-line bg-panel px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <button
          disabled={saving || !name.trim()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:text-black"
        >
          Create
        </button>
      </form>
    </main>
  );
}
