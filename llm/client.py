import hashlib
import json
import os
from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/gemma-4-31b-it:free"
_DEFAULT_JUDGE_MODEL = "google/gemma-4-26b-a4b-it:free"
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Free local model, no API key needed

_client: OpenAI | None = None
_embed_model: SentenceTransformer | None = None
_embed_cache: dict[str, List[float]] = {}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=_OPENROUTER_BASE,
        )
    return _client


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def embed(text: str) -> List[float]:
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    vector = _get_embed_model().encode(text, normalize_embeddings=True).tolist()
    _embed_cache[key] = vector
    return vector


def complete(system_prompt: str, user_message: str, model: str = _DEFAULT_MODEL) -> str:
    response = _get_client().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def judge(instruction: str, context: str, model: str = _DEFAULT_JUDGE_MODEL) -> dict:
    system = instruction + "\nRespond with valid JSON only."
    try:
        raw = complete(system, context, model=model)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except (json.JSONDecodeError, Exception):
        # Retry with stricter prompt
        try:
            system2 = instruction + "\nYou MUST respond with a single valid JSON object and nothing else. No markdown, no explanation."
            raw2 = complete(system2, context, model=model)
            raw2 = raw2.strip()
            if raw2.startswith("```"):
                raw2 = raw2.split("```")[1]
                if raw2.startswith("json"):
                    raw2 = raw2[4:]
            return json.loads(raw2.strip())
        except Exception:
            return {"error": "parse_failed"}
