import json
import time
from collections.abc import Iterator
from functools import lru_cache

from app.config import settings


@lru_cache  # keep one client alive; a temporary one gets closed mid-request when garbage collected
def _client():
    if settings.llm_provider != "gemini":
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=60_000),  # ms; fail instead of hanging
    )


def stream_completion(system: str, prompt: str) -> Iterator[str]:
    """Yield text deltas from the configured LLM provider."""
    from google.genai import types

    stream = _client().models.generate_content_stream(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


class Blocked(RuntimeError):
    """The model returned no text on purpose (e.g. finish reason RECITATION or SAFETY). Retrying
    the same input gives the same result, so this is raised at once."""


def complete_json(system: str, prompt: str, schema: dict, attempts: int = 3, images: list[bytes] = (),
                  model: str = ""):
    """One-shot completion constrained to a JSON schema. Retries on rate limits/overload.

    `images` (JPEG bytes) go before the prompt, each preceded by its number as "[i]";
    `model` overrides LLM_MODEL."""
    from google.genai import types

    contents: list = []
    for i, b in enumerate(images):
        contents += [f"[{i}]", types.Part.from_bytes(data=b, mime_type="image/jpeg")]
    contents.append(prompt)

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
    )
    for attempt in range(attempts):
        try:
            res = _client().models.generate_content(
                model=model or settings.llm_model, contents=contents, config=config
            )
            if res.text is None and res.candidates and res.candidates[0].finish_reason:
                reason = str(res.candidates[0].finish_reason)
                if not reason.endswith(("STOP", "MAX_TOKENS")):
                    raise Blocked(reason)
            return json.loads(res.text)
        except Blocked:
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
