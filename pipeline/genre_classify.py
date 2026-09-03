"""Batched end-of-scan genre classification via a local text-only LLM call.

The keyword classifier in enrich.py (`classify_genre`) is fast and free but
depends entirely on catalogue subject tags, which are frequently noisy or
absent - about 45% of matched books come back "Uncategorised" this way, even
though the resolved title/author alone is often enough for a person (or a
language model) to guess a sensible genre. Rather than chasing more keyword
rules for individual misses, this asks a model directly: one batched call
covering every still-Uncategorised book in a scan, using the clean resolved
title/author (not the raw noisy OCR), classifying each into the fixed
`enrich.GENRES` vocabulary.

Deliberately separate from providers.py: that module reads book spines out
of photographs (vision, per-tile, parallelized, one call per tile). This is
a single text-only call over a list of titles, made once per scan after
matching is done - a different shape of problem with its own prompt/schema.
"""

import json

from . import config
from .enrich import GENRES, classify_genre

SYSTEM_PROMPT = """\
You are classifying already-identified books into a fixed genre list for a \
home library app. You will be given a numbered list of books (title and \
author, already correctly identified - do not second-guess the title/author \
themselves). For each one, assign exactly one genre from this fixed list:

{genres}

Rules:
- Use your general knowledge of the book, author, or series to judge genre.
- If a book genuinely doesn't fit any of these well, or you don't recognise \
it at all, use "Uncategorised" rather than guessing.
- Use the exact genre strings above, character for character. Never invent \
a new genre name.

Return JSON only.
""".format(genres="\n".join("- " + g for g in GENRES))

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "genre": {"type": "string"},
                },
                "required": ["index", "genre"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

ALLOWED_GENRES = frozenset(GENRES)


def _candidates(records):
    """Records worth sending to the model: the keyword classifier found
    nothing in the catalogue subjects, and there is at least a title to
    classify from."""
    return [r for r in records
            if classify_genre(r.subjects) is None and (r.title or r.raw_title)]


def _build_prompt(records):
    """A numbered title/author list. The number is the index the model must
    echo back in its response, so answers can be matched up without relying
    on title text - which can repeat (multiple copies) or get paraphrased."""
    lines = ["{}. {} - {}".format(i, r.display_title, r.display_author)
             for i, r in enumerate(records)]
    return "\n".join(lines)


def _parse_response(text, count):
    """Turn the model's raw reply into {index: genre}, keeping only
    in-range indices and genres from the fixed vocabulary. Silently drops
    anything malformed rather than raising - a partial or missing
    classification pass should degrade to "no change", never crash a scan
    that had already read every book correctly.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        return {}

    result = {}
    for item in assignments:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        genre = item.get("genre")
        if not isinstance(index, int) or not (0 <= index < count):
            continue
        if genre not in ALLOWED_GENRES:
            continue
        result[index] = genre
    return result


def classify_genres_batch(records, model=None, host=None, timeout=None, progress=None):
    """Classify every still-Uncategorised record in one batched Ollama call.

    Mutates `record.llm_genre` in place for whichever records the model
    confidently classified; leaves everything else untouched (still
    "Uncategorised"). Best-effort: if Ollama isn't reachable or the reply
    doesn't parse, this is a no-op (reported via `progress` if given) rather
    than a failure - a scan that read every book correctly should never
    fail over this optional quality pass. Returns how many were classified.
    """
    candidates = _candidates(records)
    if not candidates:
        return 0

    import requests

    host = (host or config.OLLAMA_HOST).rstrip("/")
    model = model or config.OLLAMA_MODEL
    timeout = timeout or config.OLLAMA_TIMEOUT

    payload = {
        "model": model,
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(candidates)},
        ],
    }

    try:
        response = requests.post(host + "/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        if progress:
            progress("  genre classification skipped: {}".format(exc))
        return 0

    text = (body.get("message") or {}).get("content", "")
    assignments = _parse_response(text, len(candidates))
    for index, genre in assignments.items():
        candidates[index].llm_genre = genre

    if progress:
        progress("  genre classification: {}/{} previously-Uncategorised "
                 "book(s) classified".format(len(assignments), len(candidates)))
    return len(assignments)
