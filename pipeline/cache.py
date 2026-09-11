"""A shared book cache backed by Supabase.

This is the piece that makes the free catalogue APIs survive real usage. Book
collections overlap enormously - everyone in one community owns some of the same
paperbacks - so a title should be looked up externally exactly once, ever, and
served from our own table every time after that.

The measured need: cataloguing a single 291-book collection exhausted the Google
Books free daily quota twice. Without a cache, the second user of the day gets a
degraded scan through no fault of their own.

The cache is optional. With no credentials configured the pipeline behaves
exactly as before, so nobody needs a database to run a scan.
"""

import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config
from .enrich import categorize, detect_series

# A cached row has to beat this blended title/author score to be served instead
# of a fresh external lookup. Deliberately stricter than the trigram operator's
# own threshold: serving the wrong book from cache would be permanent, whereas a
# miss just costs one API call.
CACHE_HIT_THRESHOLD = 0.72

# Below this many normalised characters, a title without an author is treated as
# too generic to share. Tuned on real misfires: "switch", "culture", "japan".
SHORT_TITLE_CHARS = 12


class BookCache:
    """Read-through / write-behind cache over the `books` table.

    Every method is a no-op when the cache is disabled, so callers never have to
    branch on whether Supabase is configured.
    """

    def __init__(self, url=None, key=None, threshold=CACHE_HIT_THRESHOLD):
        self.url = (url or config.SUPABASE_URL or "").rstrip("/")
        self.key = key or config.SUPABASE_KEY
        self.threshold = threshold
        self.enabled = bool(self.url and self.key)

        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.errors = []

        if not self.enabled:
            self.session = None
            return

        self.session = requests.Session()
        self.session.headers.update({
            "apikey": self.key,
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
            "User-Agent": config.USER_AGENT,
        })
        retry = Retry(total=3, backoff_factor=0.5, backoff_jitter=0.3,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET", "POST"]),
                      raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16, pool_connections=16)
        self.session.mount("https://", adapter)

    # -- internals ---------------------------------------------------------

    def _note_error(self, where, exc):
        """Record a failure without ever letting it break a scan.

        A cache is an optimisation. If Supabase is unreachable the scan must
        still complete against the external APIs.
        """
        message = "{}: {}".format(where, type(exc).__name__)
        with self._lock:
            if message not in self.errors:
                self.errors.append(message)

    def _rest(self, path):
        return "{}/rest/v1/{}".format(self.url, path.lstrip("/"))

    # -- public API --------------------------------------------------------

    def lookup(self, record):
        """Return a cached row for this reading, or None.

        Tries the exact normalised key first - that is the cheap, indexed path
        and covers a re-scan of the same shelf - then falls back to fuzzy title
        matching for readings that differ slightly from what was stored.
        """
        if not self.enabled:
            return None

        key = make_lookup_key(record.raw_title, record.raw_author)

        try:
            response = self.session.get(
                self._rest("books"),
                params={"lookup_key": "eq." + key, "is_placeholder": "eq.false",
                        "limit": 1, "select": "*"},
                timeout=config.HTTP_TIMEOUT)
            if response.status_code == 200:
                rows = response.json()
                if rows:
                    with self._lock:
                        self.hits += 1
                    return rows[0]
        except requests.RequestException as exc:
            self._note_error("lookup", exc)
            return None
        except ValueError as exc:
            self._note_error("lookup-decode", exc)
            return None

        try:
            response = self.session.post(
                self._rest("rpc/match_cached_book"),
                json={"p_title": record.raw_title,
                      "p_author": record.raw_author or "",
                      "p_limit": 3},
                timeout=config.HTTP_TIMEOUT)
            if response.status_code != 200:
                with self._lock:
                    self.misses += 1
                return None
            rows = response.json()
        except requests.RequestException as exc:
            self._note_error("fuzzy", exc)
            return None
        except ValueError as exc:
            self._note_error("fuzzy-decode", exc)
            return None

        best = rows[0] if rows else None
        if best and (best.get("score") or 0) >= self.threshold:
            with self._lock:
                self.hits += 1
            return best

        with self._lock:
            self.misses += 1
        return None

    def store(self, record):
        """Write a freshly-resolved book back so nobody looks it up again.

        Only confident matches are cached. This asymmetry is deliberate: a cache
        miss costs one API call, but a cached *wrong* answer is permanent and is
        served to every other user in the community. Anything a human would be
        asked to review, or that came from a fragmentary spine reading, stays out.
        """
        if not self.enabled or not record.matched:
            return
        if not is_cacheable(record):
            return

        series = detect_series(record.title or record.raw_title)
        # detect_series() needs an explicit "Vol. N" in the title and finds
        # nothing for a series like The Expanse, whose volumes don't share
        # any title wording at all - catalogue_series_title (Open Library's
        # own "series:X" subject tag, when present) is the fallback for
        # exactly that case. No volume number comes with it, so ordering
        # within the group falls back to alphabetical (see
        # karovshelf-web/src/lib/seriesGrouping.ts).
        series_title = series[0] if series else record.catalogue_series_title
        series_volume = series[1] if series else None
        payload = {
            "lookup_key": make_lookup_key(record.raw_title, record.raw_author),
            "title": record.title or record.raw_title,
            "authors": record.authors or ([record.raw_author]
                                          if record.raw_author else []),
            "genre": record.genre,
            "subjects": record.subjects or [],
            "category": categorize(record.title or record.raw_title,
                                   record.authors or ([record.raw_author]
                                                      if record.raw_author else []),
                                   record.genre, record.subjects,
                                   spine_text=record.spine_text,
                                   publisher=record.publisher),
            "first_published": record.first_published,
            "cover_url": record.cover_url,
            "match_source": record.source,
            "match_score": round(float(record.match_score or 0), 3),
            "series_title": series_title,
            "series_volume": series_volume,
        }
        if record.isbn and len(record.isbn) == 13:
            payload["isbn13"] = record.isbn
        elif record.isbn:
            payload["isbn10"] = record.isbn

        try:
            # merge-duplicates: two tiles resolving the same book concurrently
            # must not fail the write. return=representation (not minimal) so
            # the caller - specifically the worker - gets the row's id back
            # to point a library_items.book_id at, without a second round trip.
            response = self.session.post(
                self._rest("books"),
                params={"on_conflict": "lookup_key"},
                headers={"Prefer": "resolution=merge-duplicates,return=representation"},
                json=payload, timeout=config.HTTP_TIMEOUT)
            if response.status_code < 300:
                with self._lock:
                    self.stores += 1
                rows = response.json()
                if rows:
                    record.book_id = rows[0].get("id")
            else:
                self._note_error("store-http{}".format(response.status_code),
                                 RuntimeError(response.text[:120]))
        except requests.RequestException as exc:
            self._note_error("store", exc)
        except ValueError as exc:
            self._note_error("store-decode", exc)

    def apply(self, record, row):
        """Populate a record from a cached row, as if it had been looked up."""
        record.matched = True
        record.match_score = float(row.get("score") or 1.0)
        record.source = "cache"
        record.book_id = row.get("id")
        record.title = row.get("title")
        record.authors = row.get("authors") or []
        record.first_published = row.get("first_published")
        record.isbn = row.get("isbn13") or row.get("isbn10")
        record.cover_url = row.get("cover_url")
        record.subjects = row.get("subjects") or []
        record.cached_genre = row.get("genre")

    @property
    def total(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return (self.hits / self.total) if self.total else 0.0

    def summary(self):
        if not self.enabled:
            return "cache: disabled (no SUPABASE_URL / SUPABASE_KEY)"
        return "cache: {}/{} hits ({:.0f}%), {} new books stored".format(
            self.hits, self.total, self.hit_rate * 100, self.stores)


def is_cacheable(record):
    """Should this resolved book be written to the shared cache?

    The bar is higher than for showing it to its own owner, because a cache row
    is shared with everybody and outlives the scan that created it.
    """
    from .enrich import _is_fragment, normalise

    if not record.matched:
        return False
    if getattr(record, "needs_review", False):
        return False
    if record.legibility == "guess":
        return False

    normalised = normalise(record.raw_title)
    # Very short or dangling readings ("Isaa", "Dungeons of") bind to whatever
    # the catalogue offers first; they are exactly the rows we must not share.
    if len(normalised) < 6 or _is_fragment(normalised):
        return False

    # A short, generic title with no author is ambiguous no matter how good the
    # similarity score looks. "Switch" matched a Sandra Brown thriller instead of
    # the Heath brothers; "Culture" matched "The Culture Code". An author makes
    # such a title identifiable; without one, leave it to a per-user lookup.
    #
    # This deliberately tests `raw_author` - what was actually read off the
    # spine - and NOT `authors`, which the lookup itself fills in. Checking the
    # resolved authors would make the guard vacuous, since a matched record
    # always has one.
    if len(normalised) < SHORT_TITLE_CHARS and not record.raw_author:
        return False
    return True


def make_lookup_key(title, author):
    """Stable key for a book, independent of spine noise.

    Imported lazily to avoid a circular import: enrich needs cache, and this
    needs enrich's normaliser.
    """
    from .enrich import normalise
    return "{}|{}".format(normalise(title), normalise(author or ""))
