# Read-only LectureLens API: search, answers, summaries and quizzes over a bundled snapshot.
# Build from the backend/ folder after running scripts/make_snapshot.py:
#     docker build -t lecturelens-api .
#     docker run -p 7860:7860 -e GEMINI_API_KEY=... lecturelens-api
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Run as a normal user with a writable home (required by Hugging Face Spaces).
RUN useradd -m -u 1000 app
WORKDIR /home/app/backend
RUN chown app:app /home/app/backend
USER app

ENV UV_COMPILE_BYTECODE=1 \
    FASTEMBED_CACHE_PATH=/home/app/models \
    DATA_DIR=/home/app/data \
    READ_ONLY=true \
    RATE_LIMIT_PER_MIN=10 \
    PORT=7860

COPY --chown=app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-cache

COPY --chown=app app ./app
# Download the embedding and reranker models at build time so the first request is fast.
RUN uv run --no-sync python -c "from app.retrieval import store, search; from app.config import settings; store._dense(); store._sparse(); settings.reranker_model and search._reranker(settings.reranker_model)"

COPY --chown=app snapshot /home/app/data

EXPOSE 7860
CMD ["sh", "-c", "uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
