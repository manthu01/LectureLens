"""Answer-quality evaluation with an LLM judge. Runs the real answer pipeline in-process
(needs the local index under data/), then has a separate, larger Gemini model grade each answer.

    backend/.venv/Scripts/python eval/judge.py --course 1
    backend/.venv/Scripts/python eval/judge.py --course 1 --answers eval/results/judge-<stamp>.json   # resume grading
    backend/.venv/Scripts/python eval/judge.py --agreement eval/results/judge-<stamp>-human.jsonl

What is measured:
- Refusal (no judge needed): the answer is the fixed "not covered" text. Unanswerable questions
  should be refused; answerable ones should not. Split by who refused: the rerank gate (no LLM
  call) or the LLM itself.
- Faithfulness: the judge splits each non-refused answer into claims and checks each one against
  the retrieved excerpts (supported by any excerpt) and against the excerpts it cites.
- Relevance: does the answer address the question.
- Correctness: does it agree with the hand-written gold answer (yes / partly / no).

The judge is a different model from the one that writes answers, but it is still Gemini, so the
script also writes a blind sheet of 20 answers to label by hand; --agreement reports how often the
judge agrees with those labels.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))

from app.config import settings  # noqa: E402
from app.generation.answer import NOT_COVERED, answer_stream  # noqa: E402
from app.retrieval.search import retrieve  # noqa: E402

# 3.8-flash was the first choice but returned 503/504 for long stretches on the free tier.
JUDGE_MODEL = "gemini-3.6-flash"
HUMAN_SAMPLE = 20

JUDGE_SYSTEM = """You grade answers from a lecture question-answering system. The system was told to
answer ONLY from the numbered lecture excerpts and to cite them with tags like [S2].

Do this carefully:
1. Split the answer into its individual factual claims (skip filler like "In the lecture,").
   For each claim, list the tags it cites (e.g. ["S2"]; [] if none).
2. supported_by_excerpts: true only if the claim is stated or directly implied by ANY excerpt.
   Outside knowledge does not count, even if the claim is true.
3. supported_by_cited: true only if the excerpts that this claim cites support it. False if it cites none.
4. answers_question: true if the answer actually addresses what was asked.
5. matches_reference: compare with the reference answer. "yes" = same key points, "partly" = some
   key points or partly wrong, "no" = missing or contradicts it. Use "no_reference" if none is given.
Excerpts are automatic transcripts and may contain transcription errors; judge meaning, not wording."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "cited": {"type": "array", "items": {"type": "string"}},
                    "supported_by_excerpts": {"type": "boolean"},
                    "supported_by_cited": {"type": "boolean"},
                },
                "required": ["claim", "cited", "supported_by_excerpts", "supported_by_cited"],
            },
        },
        "answers_question": {"type": "boolean"},
        "matches_reference": {"type": "string", "enum": ["yes", "partly", "no", "no_reference"]},
    },
    "required": ["claims", "answers_question", "matches_reference"],
}


class DailyQuotaExceeded(RuntimeError):
    pass


class StopGrading(RuntimeError):
    """Grading stopped early; progress is saved and the run can be resumed later."""


def with_retries(fn, what: str, attempts: int = 16):
    """Free-tier Gemini returns 503/504 when overloaded, sometimes for an hour; keep backing off
    (capped at 5 min, ~1 h in all). A per-day 429 won't clear until tomorrow, so stop at once."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" in msg and "per day" in msg.lower():
                raise DailyQuotaExceeded(f"{what}: daily free-tier quota used up; run again tomorrow") from None
            if attempt == attempts - 1:
                raise
            wait = min(15 * 2**attempt, 300)
            print(f"  {what}: {str(e)[:100]} -- retrying in {wait}s", flush=True)
            time.sleep(wait)


# ---------- 1. generate answers with the real pipeline ----------


def run_pipeline(course: int, question: str) -> dict:
    """Consume the same SSE stream the UI gets. Raises on an `error` event so it can be retried."""
    final, got_sources = None, False
    for event in answer_stream(course, question):
        name, data = event.split("\n", 1)
        name = name.removeprefix("event: ")
        payload = json.loads(data.removeprefix("data: "))
        if name == "sources":
            got_sources = True
        elif name == "final":
            final = payload
        elif name == "error":
            raise RuntimeError(payload["message"])
    answer = final["answer"]
    refused = answer.strip() == NOT_COVERED
    return {
        "answer": answer,
        "supported": final["supported"],
        "refused": refused,
        # No `sources` event means the rerank gate refused before any LLM call.
        "refused_by": None if not refused else ("llm" if got_sources else "gate"),
    }


def generate(course: int, questions: list[dict]) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        # Retrieval is deterministic, so this returns the same hits answer_stream used,
        # with full text (the stream only carries 240-char snippets).
        hits = retrieve(course, q["question"])
        out = with_retries(lambda: run_pipeline(course, q["question"]), q["id"])
        rows.append(
            {
                "id": q["id"],
                "type": q.get("type"),
                "question": q["question"],
                "gold_answer": q.get("gold_answer"),
                "excerpts": [
                    {
                        "tag": f"S{j}",
                        "lecture": h["lecture_number"],
                        "lecture_title": h["lecture_title"],
                        "start_sec": h["start_sec"],
                        "end_sec": h["end_sec"],
                        "text": h["text"],
                    }
                    for j, h in enumerate(hits, 1)
                ],
                **out,
            }
        )
        print(f"[{i}/{len(questions)}] {q['id']}: {'refused (' + out['refused_by'] + ')' if out['refused'] else 'answered'}", flush=True)
    return rows


# ---------- 2. judge ----------


def judge_client():
    from google import genai
    from google.genai import types

    # The judge model can take close to a minute per call under load.
    return genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=300_000))


def judge_one(client, row: dict) -> dict:
    from google.genai import types

    # Same header the answering model saw (build_prompt), so naming the lecture counts as supported.
    excerpts = "\n\n".join(
        f"[{e['tag']}] Lecture {e['lecture']}: {e.get('lecture_title', '')}\n{e['text']}" for e in row["excerpts"]
    )
    prompt = (
        f"Lecture excerpts:\n\n{excerpts}\n\n"
        f"Question: {row['question']}\n\n"
        f"Reference answer: {row['gold_answer'] or '(none)'}\n\n"
        f"Answer to grade:\n{row['answer']}"
    )
    config = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM,
        temperature=0,
        response_mime_type="application/json",
        response_schema=JUDGE_SCHEMA,
    )
    res = client.models.generate_content(model=JUDGE_MODEL, contents=prompt, config=config)
    return json.loads(res.text)


def judge_all(rows: list[dict], pace: float, save) -> None:
    """Grades rows that are not refused and not already graded; saves after each one so a run can resume."""
    client = judge_client()
    todo = [r for r in rows if not r["refused"] and "judge" not in r]
    for i, r in enumerate(todo, 1):
        # No retries: on the free tier, failed calls (503 overload included) seem to count
        # against the 20-a-day quota, so stop at the first error and resume on a later run.
        try:
            r["judge"] = judge_one(client, r)
        except Exception as e:
            msg = str(e)
            if "429" in msg and "per day" in msg.lower():
                raise DailyQuotaExceeded(f"judge {r['id']}: daily free-tier quota used up; run again tomorrow") from None
            raise StopGrading(f"judge {r['id']}: {msg[:120]}; try again later") from None
        r["judge_model"] = JUDGE_MODEL
        save()
        n = len(r["judge"]["claims"])
        ok = sum(c["supported_by_excerpts"] for c in r["judge"]["claims"])
        print(f"[{i}/{len(todo)}] {r['id']}: {ok}/{n} claims supported, ref={r['judge']['matches_reference']}", flush=True)
        time.sleep(pace)


# ---------- 3. metrics ----------


def is_faithful(j: dict) -> bool:
    return all(c["supported_by_excerpts"] for c in j["claims"])


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    judged = [r for r in rows if "judge" in r]
    claims = [c for r in judged for c in r["judge"]["claims"]]
    judged_ans = [r for r in ans if "judge" in r]
    ref = [r["judge"]["matches_reference"] for r in judged_ans]

    def frac(a, b):
        return a / b if b else 0.0

    return {
        "judge_model": JUDGE_MODEL,
        "answer_model": settings.llm_model,
        "n_answerable": len(ans),
        "n_unanswerable": len(un),
        "answerable_answered": frac(sum(not r["refused"] for r in ans), len(ans)),
        "false_refusals": [f"{r['id']} ({r['refused_by']})" for r in ans if r["refused"]],
        "unanswerable_refused": frac(sum(r["refused"] for r in un), len(un)),
        "unanswerable_refused_by_gate": sum(r["refused_by"] == "gate" for r in un),
        "unanswerable_refused_by_llm": sum(r["refused_by"] == "llm" for r in un),
        "unanswerable_answered": [r["id"] for r in un if not r["refused"]],
        "n_judged": len(judged),
        "n_to_judge": sum(not r["refused"] for r in rows),
        "n_claims": len(claims),
        "claim_support": frac(sum(c["supported_by_excerpts"] for c in claims), len(claims)),
        "citation_support": frac(sum(c["supported_by_cited"] for c in claims), len(claims)),
        "fully_faithful_answers": frac(sum(is_faithful(r["judge"]) for r in judged), len(judged)),
        "unfaithful": [r["id"] for r in judged if not is_faithful(r["judge"])],
        "answers_question": frac(sum(r["judge"]["answers_question"] for r in judged), len(judged)),
        "matches_reference": {k: frac(ref.count(k), len(ref)) for k in ("yes", "partly", "no")},
        "reference_no": [r["id"] for r in judged_ans if r["judge"]["matches_reference"] == "no"],
    }


def report(s: dict) -> str:
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    m = s["matches_reference"]
    lines = [
        f"Answers by `{s['answer_model']}`, judged by `{s['judge_model']}`.",
        "",
        *(
            [
                f"**Partial:** {s['n_judged']} of {s['n_to_judge']} answers graded so far (the free tier is rate-limited "
                "and often overloaded). Grader metrics cover only those; refusal rows cover all questions.",
                "",
            ]
            if s["n_judged"] < s["n_to_judge"]
            else []
        ),
        "| Metric | Value |",
        "|---|---|",
        f"| Answerable questions answered (not refused) | {pct(s['answerable_answered'])} of {s['n_answerable']} |",
        f"| Unanswerable questions refused | {pct(s['unanswerable_refused'])} of {s['n_unanswerable']} "
        f"(gate {s['unanswerable_refused_by_gate']}, LLM {s['unanswerable_refused_by_llm']}) |",
        f"| Claims supported by the retrieved excerpts | {pct(s['claim_support'])} of {s['n_claims']} |",
        f"| Claims supported by the excerpts they cite | {pct(s['citation_support'])} |",
        f"| Answers with every claim supported | {pct(s['fully_faithful_answers'])} of {s['n_judged']} |",
        f"| Answers that address the question | {pct(s['answers_question'])} |",
        f"| Agrees with gold answer: yes / partly / no | {pct(m['yes'])} / {pct(m['partly'])} / {pct(m['no'])} |",
    ]
    for key, label in [
        ("false_refusals", "Answerable but refused"),
        ("unanswerable_answered", "Unanswerable but answered"),
        ("unfaithful", "Has an unsupported claim"),
        ("reference_no", "Disagrees with gold answer"),
    ]:
        if s[key]:
            lines.append(f"\n{label}: {', '.join(s[key])}")
    return "\n".join(lines) + "\n"


# ---------- 4. human agreement ----------


def write_human_sheet(rows: list[dict], path: Path) -> None:
    """Blind sheet: the judge's verdicts are left out so they can't bias the labels.
    Sampled from all answered answerable questions, so it can be labelled before grading finishes."""
    answered = [r for r in rows if not r["refused"] and r["type"] != "unanswerable"]
    sample = random.Random(0).sample(answered, min(HUMAN_SAMPLE, len(answered)))
    with path.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "question": r["question"],
                        "gold_answer": r["gold_answer"],
                        "answer": r["answer"],
                        "excerpts": [f"[{e['tag']}] {e['text']}" for e in r["excerpts"]],
                        "human_faithful": None,  # true if every claim is backed by the excerpts
                        "human_matches_reference": None,  # "yes" | "partly" | "no"
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


_BOX = re.compile(r"^\s*-\s*\[([ xX])\]\s*(.+?)\s*$")
FAITHFUL_OPTS = {"Faithful: yes": True, "Faithful: no": False}
REF_OPTS = {"Matches gold: yes": "yes", "Matches gold: partly": "partly", "Matches gold: no": "no"}


def write_label_md(sheet: Path, path: Path) -> None:
    """The blind sheet as a Markdown form: tick one box per question by typing an x."""
    items = [json.loads(l) for l in sheet.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = [
        "# LectureLens hand labels",
        "",
        "For each answer, put an `x` in one **Faithful** box and one **Matches gold** box, like `- [x]`.",
        "",
        "- **Faithful: yes** = everything the answer says is in the excerpts below. **no** = it says something",
        "  that isn't, even if it's true.",
        "- **Matches gold: yes** = same key points as the gold answer. **partly** = some of them. **no** = missing",
        "  or contradicts it.",
        "",
        "Excerpts the answer cites are shown in full; the rest are folded. Write any comment on the `Note:` line.",
        "",
    ]
    for n, it in enumerate(items, 1):
        cited = {int(m) for m in re.findall(r"\[S(\d+)\]", it["answer"])}
        out += [
            "---",
            "",
            f"## {n}. {it['id']}: {it['question']}",
            "",
            f"**Gold answer:** {it['gold_answer']}",
            "",
            f"**System answer:** {it['answer']}",
            "",
            *(f"- [ ] {o}" for o in FAITHFUL_OPTS),
            "",
            *(f"- [ ] {o}" for o in REF_OPTS),
            "",
            "Note:",
            "",
        ]
        for e in it["excerpts"]:
            tag, text = re.match(r"\[(S\d+)\] (.*)", e, re.S).groups()
            if int(tag[1:]) in cited:
                out += [f"> **{tag} (cited):** {text}", ""]
            else:
                out += [f"<details><summary>{tag} (not cited)</summary>", "", text, "", "</details>", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def read_label_md(path: Path) -> list[dict]:
    labels, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^## \d+\. (\w+):", line):
            cur = {"id": m.group(1), "human_faithful": None, "human_matches_reference": None}
            labels.append(cur)
        elif cur and (m := _BOX.match(line)) and m.group(1).lower() == "x":
            opt = m.group(2)
            if opt in FAITHFUL_OPTS:
                cur["human_faithful"] = FAITHFUL_OPTS[opt]
            elif opt in REF_OPTS:
                cur["human_matches_reference"] = REF_OPTS[opt]
    return labels


def agreement(human_path: Path) -> None:
    """Accepts the labelled -human.jsonl sheet or the -labels.md form."""
    stem = human_path.name.replace("-human.jsonl", "").replace("-labels.md", "")
    judge_path = human_path.with_name(stem + ".json")
    rows = {r["id"]: r for r in json.loads(judge_path.read_text(encoding="utf-8"))["rows"]}
    if human_path.suffix == ".md":
        labels = read_label_md(human_path)
    else:
        labels = [json.loads(l) for l in human_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [h["id"] for h in labels if h["human_faithful"] is None or h["human_matches_reference"] is None]
    if missing:
        print(f"Not fully labelled yet: {', '.join(missing)}")
    for field, judge_value in [
        ("human_faithful", lambda j: is_faithful(j)),
        ("human_matches_reference", lambda j: j["matches_reference"]),
    ]:
        done = [h for h in labels if h[field] is not None and "judge" in rows[h["id"]]]
        if not done:
            print(f"{field}: no labels yet")
            continue
        agree = [h["id"] for h in done if judge_value(rows[h["id"]]["judge"]) == h[field]]
        differ = [h["id"] for h in done if h["id"] not in agree]
        print(f"{field}: judge agrees on {len(agree)}/{len(done)} ({100 * len(agree) / len(done):.0f}%)")
        if differ:
            print(f"  differs on: {', '.join(differ)}")


def main() -> None:
    global JUDGE_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", type=int)
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    ap.add_argument("--answers", help="reuse answers (and any grades) from an earlier judge-*.json")
    ap.add_argument("--rejudge", action="store_true", help="with --answers: drop earlier grades and grade again")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--pace", type=float, default=4.0, help="seconds between judge calls (free-tier rate limit)")
    ap.add_argument("--agreement", help="a labelled judge-*-human.jsonl; report judge/human agreement")
    ap.add_argument("--human-sheet", action="store_true", help="with --answers: only write the blind sheet")
    ap.add_argument("--report-only", action="store_true", help="with --answers: write the report from grades so far, no API calls")
    args = ap.parse_args()

    JUDGE_MODEL = args.judge_model
    if args.agreement:
        agreement(Path(args.agreement))
        return

    if args.answers:
        out = Path(args.answers)
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
        for r in rows:
            if args.rejudge or r.get("judge_model") != JUDGE_MODEL:
                r.pop("judge", None)
    else:
        if args.course is None:
            ap.error("--course is required unless --answers or --agreement is given")
        lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
        questions = [json.loads(l) for l in lines if l.strip() and not l.lstrip().startswith("//")]
        rows = generate(args.course, questions)
        (HERE / "results").mkdir(exist_ok=True)
        out = HERE / "results" / f"judge-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    human = out.with_name(out.stem + "-human.jsonl")
    if args.human_sheet:
        if human.exists():
            ap.error(f"{human.name} already exists; delete it first so labels aren't overwritten")
        write_human_sheet(rows, human)
        print(f"Wrote {human.name}")
        return

    def save():
        out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    save()  # answers are saved before judging, so a judge failure doesn't cost the generation quota
    try:
        if not args.report_only:
            judge_all(rows, args.pace, save)
    except (DailyQuotaExceeded, StopGrading) as e:
        done = sum("judge" in r for r in rows)
        sys.exit(f"{e}. {done} of {sum(not r['refused'] for r in rows)} answers graded so far (saved).")
    summary = summarize(rows)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    md = report(summary)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    if not human.exists():  # never overwrite a sheet that may already hold labels
        write_human_sheet(rows, human)
    print("\n" + md)
    print(f"Saved {out.name} and .md; blind sheet for hand labels: {human.name}")


if __name__ == "__main__":
    main()
