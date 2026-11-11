export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Course = { id: number; name: string; professor: string; description: string };

export type Lecture = {
  id: number;
  course_id: number;
  title: string;
  number: number;
  source_url: string;
  video_id: string;
  duration_sec: number;
  status: "queued" | "downloading" | "transcribing" | "indexing" | "indexed" | "failed";
  error: string;
};

export type CourseDetail = Course & { lectures: Lecture[] };

export type Segment = {
  id: string;
  lecture_id: number;
  start_sec: number;
  end_sec: number;
  text: string;
  label: "teaching" | "off_topic" | "admin" | "exam_hint";
  topic: string;
};

export type Citation = {
  tag: string;
  segment_id: string;
  lecture_id: number;
  lecture_number: number;
  lecture_title: string;
  video_id: string;
  start_sec: number;
  end_sec: number;
  snippet: string;
};

export type FinalAnswer = { answer: string; citations: Citation[]; supported: boolean };

async function req<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(API + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export const listCourses = () => req<Course[]>("/courses");
export const createCourse = (name: string, professor: string) => req<Course>("/courses", { name, professor });
export const getCourse = (id: number) => req<CourseDetail>(`/courses/${id}`);
export const getTranscript = (lectureId: number) => req<Segment[]>(`/lectures/${lectureId}/transcript`);
export const retryLecture = (lectureId: number) => req<Lecture>(`/lectures/${lectureId}/retry`, {});

/** Adds a single video, or every video of a playlist when the URL has a list= parameter. */
export function addSource(courseId: number, url: string, limit?: number) {
  return /[?&]list=/.test(url)
    ? req<Lecture[]>(`/courses/${courseId}/playlist`, { url, limit: limit || null })
    : req<Lecture>(`/courses/${courseId}/lectures`, { url });
}

export async function ask(
  courseId: number,
  question: string,
  on: {
    sources: (s: Citation[]) => void;
    token: (t: string) => void;
    final: (f: FinalAnswer) => void;
    error: (message: string) => void;
  },
) {
  let res: Response;
  try {
    res = await fetch(`${API}/courses/${courseId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    return on.error("Could not reach the backend. Is it running?");
  }
  if (!res.ok || !res.body) return on.error(`Request failed (${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let finished = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop() ?? "";
    for (const block of blocks) {
      const event = /^event: (.+)$/m.exec(block)?.[1];
      const data = /^data: (.+)$/m.exec(block)?.[1];
      if (!event || !data) continue;
      const payload = JSON.parse(data);
      if (event === "sources") on.sources(payload.sources);
      else if (event === "token") on.token(payload.text);
      else if (event === "final") {
        finished = true;
        on.final(payload);
      } else if (event === "error") {
        finished = true;
        on.error(payload.message);
      }
    }
  }
  if (!finished) on.error("The answer stream ended unexpectedly.");
}

export function fmtTime(sec: number): string {
  const t = Math.floor(sec);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}
