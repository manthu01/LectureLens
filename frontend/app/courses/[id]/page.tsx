import { Suspense } from "react";
import CourseView from "@/components/course-view";

export default function CoursePage() {
  return (
    <Suspense fallback={<p className="p-6 text-sm text-muted">Loading…</p>}>
      <CourseView />
    </Suspense>
  );
}
