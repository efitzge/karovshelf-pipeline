"""Pluggable vision backends: hosted Claude, or a local model via Ollama.

The point of this seam is that the expensive half of the pipeline should not be
welded to one vendor. Both providers are handed the same tile, the same prompt,
and the same output schema, so a comparison between them measures the models
rather than two different harnesses.
"""

import json

from . import config, enrich

# ---------------------------------------------------------------------------
# Shared prompt and schema - identical for every provider, so the comparison is
# actually about the model.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are cataloguing a private book collection from photographs of its shelves.

Your job is transcription, not recall. Report what is physically printed on each
spine or cover in the image. You are being measured on precision: a book you
omit costs far less than a book you invent.

Rules:

1. Work left to right across the image. Include every book you can see, whether
   its spine is vertical, rotated, or lying in a horizontal stack.
2. Record `spine_text` as the literal characters you can make out, in the order
   they appear, including publisher names and series numbering. Use "?" for
   characters you genuinely cannot resolve. This field is raw transcription.
3. Only then interpret that text into `title` and `author`. If the spine shows
   only a title, return an empty string for `author` - never supply an author
   from memory.
4. An author's name is very often printed smaller than the title, in a
   different font, sideways, or tucked at the top or bottom of the spine -
   it does not have to be prominent to count. Look specifically for a
   personal name (a surname alone is enough - "LAU", "SACKS") separate from
   the title wording, and put it in `author` even when it's the smallest
   text on the spine. Don't confuse this with a publisher or imprint name
   (e.g. "MAGGID", "PENGUIN") - those are not personal names and belong in
   `spine_text` only, never in `author`.
5. Set `legibility` honestly:
   - "clear"   - you can read the text directly and are confident
   - "partial" - some characters are obscured but the reading is well supported
   - "guess"   - you are inferring from fragments, colour, or design
   Do not complete a partially visible title into a book you happen to know
   unless the visible characters genuinely support it; mark such cases "guess".
6. Skip objects that are not books: DVDs, binders, folders, photo frames,
   ornaments.
7. If a shelf edge cuts a book off so that no text is readable, omit it.

Return every book you find, in left-to-right order.
"""

USER_PROMPT = (
    "Transcribe every book visible on this shelf section. This may be one slice "
    "of a wider photograph, so books at the edges may be partially cut off - "
    "include them only if you can read enough to identify them."
)

BOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "books": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spine_text": {
                        "type": "string",
                        "description": "Literal characters visible on the spine.",
                    },
                    "title": {"type": "string"},
                    "author": {
                        "type": "string",
                        "description": "Empty string if no author is printed.",
                    },
                    "legibility": {
                        "type": "string",
                        "enum": ["clear", "partial", "guess"],
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["vertical", "horizontal", "face-out"],
                    },
                },
                "required": [
                    "spine_text", "title", "author", "legibility", "orientation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["books"],
    "additionalProperties": False,
}


class ProviderError(RuntimeError):
    """A tile could not be read. Carries a human-readable reason."""


# ---------------------------------------------------------------------------
# Claude (hosted, paid)
# ---------------------------------------------------------------------------

class ClaudeProvider:
    name = "claude"
    costs_money = True

    def __init__(self, model=None, effort=None):
        import anthropic  # imported lazily so the local path needs no SDK

        self.model = model or config.MODEL
        self.effort = effort or config.DEFAULT_EFFORT
        self.client = anthropic.Anthropic()
        self._errors = anthropic

    def label(self):
        return "{} (effort={})".format(self.model, self.effort)

    def read_tile(self, tile):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": tile.as_base64_png(),
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": BOOK_SCHEMA},
                },
            )
        except self._errors.APIStatusError as exc:
            raise ProviderError("API {}: {}".format(exc.status_code, exc.message)) from exc
        except self._errors.APIConnectionError as exc:
            raise ProviderError("connection error: {}".format(exc)) from exc

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", None) or "no explanation"
            raise ProviderError("model declined this image ({})".format(detail))

        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            raise ProviderError("model returned no text block")

        books, truncated = process_reply(text)
        return (books, response.usage.input_tokens,
                response.usage.output_tokens, truncated)


# ---------------------------------------------------------------------------
# Ollama (local, free)
# ---------------------------------------------------------------------------

class OllamaProvider:
    name = "ollama"
    costs_money = False

    def __init__(self, model=None, host=None, timeout=None):
        import requests  # already a dependency

        self.model = model or config.OLLAMA_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        # Local generation on a consumer GPU is far slower than a hosted call;
        # a dense shelf tile can legitimately take minutes.
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self._requests = requests
        self.session = requests.Session()

    def label(self):
        return "{} (local via Ollama)".format(self.model)

    def read_tile(self, tile):
        payload = {
            "model": self.model,
            "stream": False,
            "format": BOOK_SCHEMA,   # Ollama accepts a JSON schema here
            "options": {
                "temperature": 0,        # transcription, not creativity
                # A dense tile plus this prompt can exceed 8k, and overflowing
                # truncates the reply mid-JSON.
                "num_ctx": config.OLLAMA_NUM_CTX,
                # Tried repeat_penalty as a mitigation for the loop below and it
                # made things WORSE: the model still looped, but got penalised
                # for reusing the title tokens it had already spent on
                # spine_text, so it emitted a correctly-shaped but EMPTY title
                # for every entry instead. Our own filter then silently drops
                # anything with no title, so the failure became invisible - a
                # tile with 5 real, legible books reported 0 with no warning.
                # Removed. num_predict below plus _collapse_repetition() is the
                # real fix: let the loop happen, bound how long it can run, and
                # detect+flag it after the fact rather than trying to suppress
                # it at generation time.
                "num_predict": config.OLLAMA_NUM_PREDICT,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT,
                    "images": [tile.as_base64_png()],
                },
            ],
        }

        try:
            response = self.session.post(self.host + "/api/chat", json=payload,
                                         timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except self._requests.Timeout as exc:
            raise ProviderError(
                "timed out after {}s - the tile may be too large for this "
                "model/GPU".format(self.timeout)) from exc
        except self._requests.RequestException as exc:
            raise ProviderError("ollama request failed: {}".format(exc)) from exc
        except ValueError as exc:
            raise ProviderError("ollama returned non-JSON: {}".format(exc)) from exc

        if "error" in body:
            raise ProviderError("ollama error: {}".format(body["error"]))

        text = (body.get("message") or {}).get("content", "")
        if not text.strip():
            raise ProviderError("model returned empty content")

        books, truncated = process_reply(text)
        return (books, body.get("prompt_eval_count", 0),
                body.get("eval_count", 0), truncated)


# ---------------------------------------------------------------------------

def _salvage_objects(text):
    """Pull every complete {...} object out of a truncated reply.

    Smaller models occasionally run away - repeating entries until they hit the
    context limit and get cut off mid-string. The reply is then invalid JSON,
    but the books listed *before* the truncation are perfectly good. Throwing
    the whole tile away over its last broken entry loses a dozen real books, so
    scan for balanced objects instead and keep what parsed.

    Keeps an object if EITHER `title` or `spine_text` is non-empty - the same
    "whichever field actually holds content" reasoning as `_repetition_key`
    below, for the same reason. Checking `title` alone (as this originally
    did) meant a tile that degenerates into a run of `{"title": ""}` entries -
    real behaviour, seen on IMG_7103.jpeg's r1c1 tile, a crop of loose paper
    folders with no book spines at all - salvaged nothing, even though the
    one entry with real content (`spine_text: "USA"`, an illegible guess) was
    right there. `_finalise_books` still drops the title-less result, so the
    end effect of this fix is a clean "0 books, no error" rather than a hard
    tile failure - the correct outcome for a genuinely bookless crop.
    """
    objects = []
    # Stack of opening-brace positions, so we catch objects nested inside the
    # (unclosed) wrapper as well as top-level ones.
    opens = []
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            opens.append(i)
        elif ch == "}" and opens:
            start = opens.pop()
            try:
                candidate = json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and (candidate.get("title") or candidate.get("spine_text")):
                objects.append(candidate)
    return objects


# A real shelf crop essentially never shows more than 2-3 physical copies of the
# identical title, so any title appearing more than this many times in one
# tile's response is a generation loop, not genuine duplication. Observed twice
# now on real scans: a single tile repeated "The Catcher in the Rye" 218 times
# and "How to Win in a Winner-Take-All World" 202 times, in both cases after a
# handful of correctly-read books at the start of the crop. The loop consumes
# the whole remaining output budget, so anything physically further along that
# shelf section is silently lost - this is a data-loss bug, not a cosmetic one.
MAX_PLAUSIBLE_REPEATS = 3

# Bulk-bought Judaica (a dozen identical Haggadahs/siddurim for Seder or Shabbat
# guests) genuinely clears the generic ceiling above - found 2026-08-26 on a
# real phone-uploaded batch, ~12 real identical copies flagged as a suspected
# loop. Still nowhere near the 202-218 range a real runaway loop produces, so
# raising the ceiling for Judaica-shaped repeats only (see
# `enrich.categorize`) keeps the loop-detection safety net intact while no
# longer false-flagging a household that just bought Haggadahs in bulk.
JUDAICA_MAX_PLAUSIBLE_REPEATS = 20


def _repetition_key(item):
    """The field to compare for repetition - whichever actually holds content.

    `spine_text` is the ground-truth transcription and is what stays populated
    even in the degenerate case where the model gets stuck on a loop and starts
    emitting an empty `title` for every entry (observed when repeat_penalty was
    briefly tried as a fix here: it did not stop the loop, it just penalised
    the model for reusing title tokens it had already spent on spine_text, so
    every entry came back correctly-shaped but title-less). Keying on title
    alone would miss that failure entirely, since "" == "" for every item.
    """
    spine = (item.get("spine_text") or "").strip().lower()
    if spine:
        return spine
    return (item.get("title") or "").strip().lower()


def _collapse_repetition(items):
    """Detect and trim a runaway generation loop within one tile's reply.

    Runs on the RAW parsed items, before anything is dropped for having an
    empty title - that ordering matters, because the failure this exists to
    catch can present as many well-formed-looking entries with real
    `spine_text` but no `title` at all. Filtering those out first would make
    the loop invisible instead of merely unfixable.

    Returns (cleaned_items, truncated) where `truncated` is True if a loop was
    detected. When it fires, everything genuinely read before the loop started
    is kept; the loop itself is collapsed to a single copy (real small
    duplication - two copies of a book sitting together - still survives
    downstream dedupe); and the caller should treat this tile as suspect, since
    whatever came after the loop point in the actual photo was never read.
    """
    counts = {}
    for item in items:
        key = _repetition_key(item)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1

    runaway_keys = set()
    for key, count in counts.items():
        # `key` is the raw spine/title text - already exactly what
        # `categorize` wants for its Hebrew-script/publisher/keyword checks.
        is_judaica = enrich.categorize(key, [], None, spine_text=key) == "Judaica"
        ceiling = JUDAICA_MAX_PLAUSIBLE_REPEATS if is_judaica else MAX_PLAUSIBLE_REPEATS
        if count > ceiling:
            runaway_keys.add(key)
    if not runaway_keys:
        return items, False

    cleaned = []
    seen_runaway = set()
    for item in items:
        key = _repetition_key(item)
        if key in runaway_keys:
            if key in seen_runaway:
                continue
            seen_runaway.add(key)
        cleaned.append(item)
    return cleaned, True


def _parse_books(text):
    """Parse a provider's JSON reply into plain dicts.

    Local models sometimes wrap JSON in prose or a code fence even when handed a
    schema, so recover the outermost object rather than failing the whole tile.

    Deliberately does NOT drop entries with an empty title - that filtering
    happens later, in `_finalise_books`, AFTER repetition detection has had a
    chance to see the raw shape of the response. See `_repetition_key` for why
    that ordering matters.
    """
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]

    items = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Truncated or malformed - keep whatever complete entries survived.
        items = _salvage_objects(text)
        if not items:
            raise ProviderError(
                "unparseable JSON and nothing salvageable ({} chars)".format(len(text))
            ) from None
    else:
        items = payload.get("books")
        if not isinstance(items, list):
            raise ProviderError("reply had no 'books' list")

    books = []
    for item in items:
        if not isinstance(item, dict):
            continue
        legibility = item.get("legibility")
        if legibility not in ("clear", "partial", "guess"):
            legibility = "guess"
        books.append({
            "spine_text": (item.get("spine_text") or "").strip(),
            "title": (item.get("title") or "").strip(),
            "author": (item.get("author") or "").strip() or None,
            "legibility": legibility,
            "orientation": item.get("orientation") or "vertical",
            # Only the manual multi-photo prompt asks for this (see
            # manual.MANUAL_BOOK_SCHEMA) - absent for automated per-tile
            # replies, where it would be redundant with Tile.label.
            "photo_index": item.get("photo_index"),
        })
    return books


def _finalise_books(items):
    """Drop entries with no usable title. The last step, after repetition
    detection has already run on the unfiltered list."""
    return [item for item in items if item["title"]]


def process_reply(text):
    """Full pipeline from raw model text to (books, truncated).

    Order matters: parse (keep everything) -> detect/collapse repetition on the
    raw shape -> only then drop title-less entries. Filtering first would let a
    loop that degenerates into empty titles hide from detection entirely.
    """
    raw = _parse_books(text)
    collapsed, truncated = _collapse_repetition(raw)
    return _finalise_books(collapsed), truncated


PROVIDERS = {"claude": ClaudeProvider, "ollama": OllamaProvider}


def get_provider(name, **kwargs):
    try:
        factory = PROVIDERS[name]
    except KeyError:
        raise ProviderError("unknown provider {!r}; choose from {}".format(
            name, ", ".join(sorted(PROVIDERS)))) from None
    return factory(**kwargs)
