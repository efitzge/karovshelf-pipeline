"""Shared configuration for the shelf-scanning pipeline."""

import os
from pathlib import Path


def _load_dotenv():
    """Read KEY=value pairs from a .env file beside the project root.

    Deliberately dependency-free and non-destructive: a variable already set in
    the real environment always wins, so `.env` is a convenience for local work
    rather than something that can silently override a deployment.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Skip blanks so an unfilled placeholder never shadows a real variable.
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# Claude Opus 5 — the vision model doing the spine reading.
MODEL = "claude-opus-5"

# Claude API pricing, USD per million tokens (Opus 5).
PRICE_IN_PER_MTOK = 5.00
PRICE_OUT_PER_MTOK = 25.00

# Reasoning effort for the extraction call. Spine reading is a perception task
# more than a reasoning one, so this is worth A/B-ing: "low" and "medium" are
# markedly cheaper and faster, "high" is the API default.
DEFAULT_EFFORT = "medium"

# Anthropic caps vision input at roughly 1568px on the long edge before it
# downscales server-side. A whole bookshelf scaled to that is unreadable, which
# is why we tile instead of shrinking the full frame.
MAX_EDGE = 1568

# Fraction of overlap between adjacent tiles, so a book sitting on a seam still
# appears whole in at least one tile.
TILE_OVERLAP = 0.18

# Below this fuzzy-match ratio we refuse to bind an extracted guess to a
# catalogue record, and keep it as unverified instead.
MATCH_THRESHOLD = 0.72

# Author similarity below this is treated as NOT corroborating a title match -
# real scan (2026-09-03): a "Brown" on a Maggid "Ecclesiastes" spine scored
# exactly 0.5 against "William P. Brown", an unrelated author on an unrelated
# book. That is coincidental surname overlap, not agreement. When a candidate
# clears MATCH_THRESHOLD on title but author similarity is None or below
# this, `enrich()` treats it as uncorroborated and applies
# UNCORROBORATED_MATCH_THRESHOLD instead - see its own comment.
AUTHOR_CORROBORATION_FLOOR = 0.6

# The bar an uncorroborated match (no author on the scan, or one that doesn't
# clear AUTHOR_CORROBORATION_FLOOR) must clear on title alone, with the
# prefix-continuation boost turned off. Short/generic titles are exactly
# where a coincidental collision with a real, unrelated book is likely -
# found on real scans, all with no author to catch it: "Isaiah" bound to a
# biography of the philosopher Isaiah Berlin, "Jonah" to a horror novel
# titled "The Jonah", "The Complete Sherlock Holmes: A Concise Companion" to
# the original Conan Doyle story collection. Without a second signal, only
# something close to an exact title match should be trusted.
UNCORROBORATED_MATCH_THRESHOLD = 0.92

# Two extracted books whose normalised titles match this closely are treated as
# the same physical book seen in two overlapping tiles.
DEDUPE_THRESHOLD = 0.88

OPENLIBRARY_SEARCH = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_SEARCH = "https://www.googleapis.com/books/v1/volumes"

# Google Books without a key shares one small anonymous quota with the whole
# internet and returns HTTP 429 most of the time. A free key raises that to
# ~1000 queries/day, which matters here because Google has much better coverage
# of English editions of translated works than Open Library does.
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")

# Be a good citizen of the free Open Library API.
USER_AGENT = os.environ.get(
    "KAROVSHELF_UA", "karovshelf/0.1 (bookshelf cataloguing experiment)"
)
HTTP_TIMEOUT = 20.0

# Both catalogues fail intermittently. Google Books in particular serves 503
# "backendFailed" on a large share of requests at busy times, regardless of the
# query or the key, so retrying is not optional.
HTTP_RETRIES = 4
HTTP_BACKOFF = 0.8

# --- Shared book cache (Supabase) --------------------------------------------
# Optional. Without these the pipeline works exactly as before, just hitting the
# external catalogues every time. With them, a title is looked up once ever and
# served from our own table thereafter - which is what keeps the free Google
# Books quota from being the binding constraint on the whole product.
#
# Writing to `books` requires a key that can insert under RLS. For this local
# admin tool that means the service role key; the web app will use a signed-in
# user's token instead.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")

# --- Local vision provider (Ollama) ------------------------------------------
# A local model costs nothing per scan, which is the whole point: it turns a
# per-photo bill into a one-off hardware requirement.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
# Generous: a dense tile on a consumer GPU can take minutes, and a timeout here
# throws away work that was nearly finished.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
# Context has to hold the image, the prompt, and the whole reply; overflowing it
# truncates the JSON mid-entry. Measured at both 8192 and 16384: accuracy was
# identical (38/42 either way), but 8192 ran 40% faster, so there is no reason
# to pay for the bigger window. The salvage path in providers.py covers the
# truncation case that motivated raising it.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "3072"))
