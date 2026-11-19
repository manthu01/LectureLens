# LectureLens demo video: script

Target length: about 2 minutes. Record the live demo (https://lecturelens-app.vercel.app) at 1080p, browser zoom 110 to 125% so text is readable.

**Before recording:** open the demo once and ask any question, so the backend is warm (the first request after a quiet spell is slow). Close other tabs, hide bookmarks, and turn off notifications.

| # | Time | On screen | Say (roughly) |
|---|---|---|---|
| 1 | 0:00–0:12 | Home page, course card | "Lecture videos are long, and the one explanation you need is buried somewhere in hours of them. LectureLens is Ctrl+F for a whole course of lectures." |
| 2 | 0:12–0:25 | Open the course; show the 5 lectures | "This course is five lectures from CampusX's 100 Days of Machine Learning, taught in Hinglish. Whisper transcribed and translated them, and every segment was labelled and indexed." |
| 3 | 0:25–0:50 | Ask: **Why did machine learning only take off after 2010?** Let the answer stream in | "I ask a question, and the answer comes only from what was taught. Every claim cites a lecture and a timestamp." |
| 4 | 0:50–1:02 | Click a citation; the video jumps to that moment and plays a few seconds | "Clicking a citation takes me to the exact second he explains it." |
| 5 | 1:02–1:20 | Ask: **What is the bias-variance trade-off?** Show the refusal and the closest moments | "If the lectures don't cover something, it says so instead of guessing. Bias-variance is only mentioned as a future topic here, so it refuses and shows the closest moments." |
| 6 | 1:20–1:38 | Open a lecture's summary and topic timeline; point at the greyed-out stretch | "Each lecture gets a summary with timestamped key points, and a topic timeline. Off-topic stretches are greyed out." |
| 7 | 1:38–1:50 | Generate a practice quiz; click a question's timestamp | "It also writes practice quizzes from the lecture itself, each question linked to where it was taught." |
| 8 | 1:50–2:05 | README: architecture diagram, then the evaluation table | "Under the hood: hybrid search with a confidence gate, and Gemini for the answers. It's measured on a 66-question eval set: the right moment comes first for 96% of questions." |

## Backup questions

If an answer looks weak on the day, swap in one of these (all from the eval set, answer known to be in the lectures):

- Why is a rule-based spam filter hard to maintain compared to machine learning? (lecture 1)
- Why did expert systems fail? (lecture 2)
- What is machine learning, in simple words? (lecture 1)

Backup refusal: **How do you deploy a machine learning model with Flask?**

## After recording

- Trim the dead air while answers stream (or speed those parts up 1.5x).
- Add captions; many viewers watch muted.
- Upload unlisted to YouTube and link it at the top of the README, next to the live demo link.
