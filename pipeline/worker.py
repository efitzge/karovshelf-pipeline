"""The extraction worker: turns one pending scan_batch into a draft library.

Polls Supabase for `scan_batches` in `status='uploaded'`, downloads that
batch's photos from the `shelf-photos` bucket, runs them through the existing
tile/read/dedupe/catalogue-match pipeline, and either:

  - writes draft `library_items` rows and marks the batch `ready` (the owner
    reviews next), or
  - marks it `needs_admin` with a human-readable reason, and writes nothing
    to `library_items` yet - `pipeline/manual.py` handles those.

The routing decision (`decide_route`) is a pure function, deliberately kept
separate from all the Supabase I/O below it, so the thresholds can be tuned
and tested without touching a live database.
"""

import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config, enrich, extract, genre_classify, imageprep, providers
from .cache import BookCache, is_cacheable, make_lookup_key


def resolve_source_photo(seen_in, photo_paths):
    """Map a record's seen_in (tile/photo labels like 'photo00.jpeg [r1c1]'
    or the manual path's plain 'photo00') back to the actual Storage path it
    came from, using the index both labeling schemes share. Returns None if
    unparseable or out of range - the caller then falls back to showing the
    whole batch's photos instead of just one.
    """
    if not seen_in:
        return None
    match = re.match(r"photo(\d+)", seen_in[0])
    if not match:
        return None
    index = int(match.group(1))
    if index >= len(photo_paths):
        return None
    return photo_paths[index]

# Starting thresholds - see MVP_WORKFLOW.md. Tune once real batches land;
# nothing else needs to change when these do, since they're read from here.
MATCH_RATE_FLOOR = 0.65
NEEDS_REVIEW_CEILING = 0.40

# Relaxed floor/ceiling for a batch that's mostly Judaica (see
# `enrich.categorize`). The generic thresholds above were tuned against a
# generic English-library benchmark (MVP_WORKFLOW.md) and over-flag Judaica-
# heavy households by default - found 2026-08-26 on a real phone-uploaded
# batch (Hirsch Chumash, Steinsaltz, Artscroll, Koren editions) that came back
# 48% needs-review purely because specialty Hebrew/Judaica editions match
# Open Library/Google Books structurally worse than mainstream English
# titles, not because the scan itself was unreliable. Accept a lower match
# rate and a higher review ratio for these rather than escalating a shelf
# that's just... a Judaica-heavy household.
JUDAICA_MAJORITY_FLOOR = 0.5
JUDAICA_MATCH_RATE_FLOOR = 0.40
JUDAICA_NEEDS_REVIEW_CEILING = 0.65


def decide_route(stats):
    """Pure: given one batch's pipeline stats, decide auto-approve vs escalate.

    `stats` needs: distinct_books, matched, needs_review, tiles_suspect,
    tiles_failed, judaica_fraction. Returns (status, flagged_reason) - status
    is 'ready' or 'needs_admin'; flagged_reason is None when status is
    'ready'.
    """
    distinct = stats.get("distinct_books", 0)
    if distinct == 0:
        return "needs_admin", "no books were extracted from these photos at all"

    reasons = []
    suspect = stats.get("tiles_suspect", 0)
    if suspect:
        reasons.append("{} tile(s) hit a repetition loop".format(suspect))

    failed = stats.get("tiles_failed", 0)
    if failed:
        reasons.append("{} tile(s) failed to parse".format(failed))

    is_judaica_batch = stats.get("judaica_fraction", 0.0) >= JUDAICA_MAJORITY_FLOOR
    match_floor = JUDAICA_MATCH_RATE_FLOOR if is_judaica_batch else MATCH_RATE_FLOOR
    review_ceiling = JUDAICA_NEEDS_REVIEW_CEILING if is_judaica_batch else NEEDS_REVIEW_CEILING

    match_rate = stats.get("matched", 0) / distinct
    if match_rate < match_floor:
        reasons.append("match rate {:.0%} is below the {:.0%} floor".format(
            match_rate, match_floor))

    review_ratio = stats.get("needs_review", 0) / distinct
    if review_ratio > review_ceiling:
        reasons.append("{:.0%} of books need review (ceiling {:.0%})".format(
            review_ratio, review_ceiling))

    if reasons:
        return "needs_admin", "; ".join(reasons)
    return "ready", None


@dataclass
class PendingBatch:
    id: str
    owner_id: str
    photo_paths: list
    status: str = None


class SupabaseWorkerClient:
    """Thin REST client for the worker's Supabase access.

    Always authenticates with the service-role key - the worker is not a
    signed-in user, it is the system itself, and scan_batches has no
    owner-facing update policy by design (see the migration comment). This
    mirrors the session/retry setup in cache.py; kept separate rather than
    shared so cache.py - already tested and in production use - stays
    untouched by this new module.
    """

    def __init__(self, url=None, key=None):
        self.url = (url or config.SUPABASE_URL or "").rstrip("/")
        self.key = key or config.SUPABASE_KEY
        if not (self.url and self.key):
            raise RuntimeError(
                "worker needs SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")

        self.session = requests.Session()
        self.session.headers.update({
            "apikey": self.key,
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
            "User-Agent": config.USER_AGENT,
        })
        retry = Retry(total=3, backoff_factor=0.5, backoff_jitter=0.3,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET", "POST", "PATCH"]),
                      raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8, pool_connections=8)
        self.session.mount("https://", adapter)

    def _rest(self, path):
        return "{}/rest/v1/{}".format(self.url, path)

    def _storage_object(self, path):
        return "{}/storage/v1/object/{}".format(self.url, path)

    def _admin(self, path):
        return "{}/auth/v1/admin/{}".format(self.url, path)

    def fetch_pending(self, limit=1):
        response = self.session.get(
            self._rest("scan_batches"),
            params={"status": "eq.uploaded", "order": "created_at.asc",
                    "limit": limit, "select": "id,owner_id,photo_paths"},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        return [PendingBatch(id=row["id"], owner_id=row["owner_id"],
                             photo_paths=row["photo_paths"])
                for row in response.json()]

    def get_batch(self, batch_id):
        """Fetch one scan_batches row by id, regardless of status - for
        finishing a batch that already went through a real web upload (the
        manual path's `finish` command), as opposed to `fetch_pending`,
        which only ever looks at 'uploaded' rows for the automated poller."""
        response = self.session.get(
            self._rest("scan_batches"),
            params={"id": "eq." + batch_id,
                    "select": "id,owner_id,photo_paths,status"},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            raise RuntimeError("no scan_batches row with id={!r}".format(batch_id))
        row = rows[0]
        return PendingBatch(id=row["id"], owner_id=row["owner_id"],
                            photo_paths=row["photo_paths"], status=row["status"])

    def claim(self, batch_id):
        """Mark a batch 'processing' before working on it.

        Best-effort only: with a single worker (Elliot's PC) there is no real
        concurrent-claim race to defend against yet. If a second worker is
        ever added, this needs a proper `status='uploaded' -> 'processing'`
        conditional update to avoid two workers processing the same batch.
        """
        response = self.session.patch(
            self._rest("scan_batches"), params={"id": "eq." + batch_id},
            json={"status": "processing"}, timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()

    def download_photo(self, storage_path, dest_path):
        response = self.session.get(
            self._storage_object("shelf-photos/" + storage_path),
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        with open(dest_path, "wb") as handle:
            handle.write(response.content)

    def upload_photo(self, owner_id, batch_id, local_path, index):
        """Push one locally-held photo into the shelf-photos bucket.

        Used by the manual-publish path (pipeline/manual.py's photos never
        went through a web upload form), so the batch still ends up owning a
        real Storage object at the same {owner_id}/{batch_id}/... path shape
        the RLS policies and the automated worker both expect. Returns the
        storage-relative path to store in scan_batches.photo_paths.
        """
        ext = os.path.splitext(local_path)[1] or ".jpg"
        storage_path = "{}/{}/photo{:02d}{}".format(owner_id, batch_id, index, ext)
        content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
        with open(local_path, "rb") as handle:
            data = handle.read()
        response = self.session.post(
            self._storage_object("shelf-photos/" + storage_path),
            data=data, headers={"Content-Type": content_type},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        return storage_path

    def create_batch(self, owner_id, photo_paths):
        """Insert a scan_batches row for photos that arrived outside the
        (not-yet-built) upload web app - e.g. photos Elliot received by email
        or WhatsApp and is publishing through the manual path by hand."""
        response = self.session.post(
            self._rest("scan_batches"),
            json={"owner_id": owner_id, "photo_paths": photo_paths,
                 "status": "processing"},
            headers={"Prefer": "return=representation"},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()[0]["id"]

    def create_tester_account(self, email, pin, display_name, neighbourhood=None):
        """Create one beta tester's account via the Admin API, not `/invite`.

        `/invite` sends a confirmation email and hit Supabase's free-tier rate
        limit the first time this was tried (see MVP_WORKFLOW.md) - fine
        since there's no invite-link landing flow to use anyway, testers get
        their email + PIN directly from Elliot instead. `email_confirm: true`
        skips Supabase's own "confirm your email" step for the same reason.
        `user_metadata` feeds `handle_new_user()`'s COALESCE directly - a real
        display_name at creation time means a tester never sees their raw
        email local-part as their name, even before they've visited Profile.
        Raises for a duplicate email or any other Admin API rejection; the
        caller decides whether that stops a whole batch or just skips one row.
        """
        metadata = {"display_name": display_name}
        if neighbourhood:
            metadata["neighbourhood"] = neighbourhood
        response = self.session.post(
            self._admin("users"),
            json={"email": email, "password": pin, "email_confirm": True,
                 "user_metadata": metadata},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()["id"]

    def resolve_owner_id(self, display_name):
        """Look up a profile by display_name - a convenience for the manual
        CLI so Elliot can type a name instead of copy-pasting a UUID. Exact
        match only; profiles.display_name is set from the account's email
        local-part at invite time (see handle_new_user())."""
        response = self.session.get(
            self._rest("profiles"),
            params={"display_name": "eq." + display_name, "select": "id"},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            raise RuntimeError(
                "no profile with display_name={!r}".format(display_name))
        return rows[0]["id"]

    def finish_batch(self, batch_id, status, stats, flagged_reason=None,
                     processing_source=None):
        # Computed here, not sent as the SQL literal "now()" - that is
        # function-call syntax and is not a valid plain-text timestamp input,
        # so PostgREST would reject it when casting the JSON string.
        now = datetime.now(timezone.utc).isoformat()
        payload = {"status": status, "stats": stats, "processed_at": now}
        if flagged_reason is not None:
            payload["flagged_reason"] = flagged_reason
        if processing_source is not None:
            payload["processing_source"] = processing_source
        if status == "imported":
            payload["imported_at"] = now
        response = self.session.patch(
            self._rest("scan_batches"), params={"id": "eq." + batch_id},
            json=payload, timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()

    def find_owner_placeholder_book(self, owner_id, title, authors):
        """An existing placeholder book this owner already has for (title, authors).

        Placeholders aren't in the shared cache's lookup path (by design -
        see the is_placeholder migration comment), so without this check a
        re-scan of a book that still doesn't clear the cache bar mints a
        brand-new placeholder every time, and upsert_library_item's
        on_conflict (owner_id, book_id) can't find the old row to merge
        into - a real duplicate in the owner's own library. Scoped strictly
        to this owner (via their own library_items) so it can never affect
        the shared catalogue or another owner's books.
        """
        target_key = make_lookup_key(title, ", ".join(authors or []))
        try:
            response = self.session.get(
                self._rest("library_items"),
                params={"owner_id": "eq." + owner_id,
                        "select": "book_id,book:books!inner(title,authors,is_placeholder)"},
                timeout=config.HTTP_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException:
            return None
        for row in response.json():
            book = row.get("book") or {}
            if not book.get("is_placeholder"):
                continue
            existing_key = make_lookup_key(book.get("title"), ", ".join(book.get("authors") or []))
            if existing_key == target_key:
                return row.get("book_id")
        return None

    def insert_placeholder_book(self, owner_id, title, authors, spine_text=None, genre=None):
        """A books row that exists only so a library_item has an FK target.

        Never fuzzy-searchable by another user - see the is_placeholder
        migration comment. lookup_key just needs to satisfy NOT NULL UNIQUE;
        its value is otherwise irrelevant since placeholders are excluded
        from every lookup path.

        Reuses an existing placeholder of this owner's for the same
        (normalised) title/authors instead of minting a new one, so a
        re-scan of an unmatched book updates the same library_items row
        rather than duplicating it.

        `spine_text` matters for `enrich.categorize()` - a Maggid/Koren/
        Artscroll book's cleaned-up TITLE often doesn't carry the publisher
        name that only the spine does (see `categorize`'s own docstring).
        Every caller has this on hand (the record that just failed to match
        a catalogue), so there's no excuse to drop it here - found
        2026-09-03 when a real batch's Maggid/Artscroll Judaica placeholders
        came back uncategorised (`category IS NULL`) purely because this
        parameter didn't exist yet.

        `lookup_key` is deterministic (owner_id + normalised title/authors),
        not a random uuid, and the write uses on_conflict/merge-duplicates -
        the same retry-safety pattern as BookCache.store(). This session's
        own HTTP retry config retries POST on a transient 429/500/502/503/
        504, and a random lookup_key made that retry unsafe: if the first
        attempt's insert actually succeeded server-side but the response was
        lost, the retry minted a second, genuinely duplicate placeholder
        with a different random key - found 2026-09-04 as a real duplicate
        ("My First Book about Physics", two rows from one input record; the
        exact trigger wasn't pinned down, but this gap was real regardless).
        This also gives find_owner_placeholder_book()'s own reuse check a
        database-level backstop instead of being the only guard.

        `genre`, when given, should be `record.genre` (which already folds
        in the batched end-of-scan LLM classification pass, not just the
        catalogue-subjects keyword classifier) - a placeholder has no
        catalogue subjects at all, so without this it always fell back to
        `categorize()`'s title/spine keyword tier alone. Every caller runs
        `genre_classify.classify_genres_batch()` on its records before this
        is called, so `record.genre` is always the best guess available by
        the time we get here. Found 2026-09-04: `category` was being
        computed with a hardcoded `None` genre, silently discarding an LLM
        classification the pipeline had already paid for.
        """
        existing_id = self.find_owner_placeholder_book(owner_id, title, authors)
        if existing_id:
            return existing_id

        series = enrich.detect_series(title)
        lookup_key = "placeholder:" + owner_id + ":" + make_lookup_key(
            title, ", ".join(authors or []))
        payload = {
            "title": title,
            "authors": authors or [],
            "lookup_key": lookup_key,
            "is_placeholder": True,
            "genre": genre,
            "category": enrich.categorize(title, authors, genre, spine_text=spine_text),
            "series_title": series[0] if series else None,
            "series_volume": series[1] if series else None,
        }
        response = self.session.post(
            self._rest("books"), json=payload,
            params={"on_conflict": "lookup_key"},
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()[0]["id"]

    def supersede_owner_placeholder(self, owner_id, book_id, title, authors):
        """Delete this owner's existing placeholder for (title, authors) now
        that a real catalogue match exists for the same book.

        Mirrors find_owner_placeholder_book/insert_placeholder_book's own
        reuse logic, but for the other direction: a book that missed the
        catalogue on one scan (became a placeholder) and then matched on a
        later rescan of the same shelf used to just add a second
        library_items row instead of replacing the first, because only the
        placeholder-to-placeholder path checked for an existing owner copy -
        the placeholder-to-*matched* path had no such check at all. Found
        2026-09-03 as five real duplicate pairs (Contract Law and Theory,
        The Koren Tehillim, etc. - see PROJECT_SUMMARY.md #10) after Elliot
        spotted repeats in the live library; cleaned up by hand in Supabase,
        this closes the code-level gap that produced them.

        No-op if this owner has no matching placeholder - the normal case.
        """
        old_book_id = self.find_owner_placeholder_book(owner_id, title, authors)
        if not old_book_id or old_book_id == book_id:
            return
        response = self.session.delete(
            self._rest("library_items"),
            params={"owner_id": "eq." + owner_id, "book_id": "eq." + old_book_id},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()
        # Placeholders are minted one-per-scan and never shared with another
        # owner (see insert_placeholder_book's docstring) - safe to delete
        # outright now that its one referencing library_item is gone.
        response = self.session.delete(
            self._rest("books"), params={"id": "eq." + old_book_id},
            timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()

    def upsert_library_item(self, owner_id, book_id, scan_spine_text, scan_legibility,
                            scan_batch_id=None, source_photo_path=None):
        """Create (or update, if this owner+book already exists) a draft.

        confirmed=False, shareable=False: nothing here is visible to anyone
        but the owner until they explicitly review and approve it - see
        MVP_WORKFLOW.md's insistence that owner review is identical
        regardless of processing path. `scan_batch_id` lets the review UI
        link a book back to the batch it was scanned from; `source_photo_path`
        narrows that further to the one photo it actually came from, when
        resolvable (see `resolve_source_photo`).
        """
        payload = {
            "owner_id": owner_id, "book_id": book_id,
            "shareable": False, "confirmed": False,
            "scan_spine_text": scan_spine_text, "scan_legibility": scan_legibility,
            "scan_batch_id": scan_batch_id, "source_photo_path": source_photo_path,
        }
        response = self.session.post(
            self._rest("library_items"),
            params={"on_conflict": "owner_id,book_id"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            json=payload, timeout=config.HTTP_TIMEOUT)
        response.raise_for_status()


def process_batch(client, batch, book_cache=None, language="any", progress=None,
                  provider_name="ollama"):
    """Run one PendingBatch through the pipeline and update Supabase.

    `provider_name` defaults to "ollama" (free, local) - that's the automated
    poller's normal path via run_once()/worker_run.py. Pass "claude" for a
    one-off hosted run, e.g. testing a real phone-uploaded batch against the
    Anthropic API; nothing about the default automated behaviour changes.

    Returns the decided status ('ready' or 'needs_admin') for the caller to
    log/report.
    """
    log = progress or (lambda msg: None)
    book_cache = book_cache if book_cache is not None else BookCache()

    with tempfile.TemporaryDirectory(prefix="scanbatch_") as tmp:
        local_paths = []
        for i, storage_path in enumerate(batch.photo_paths):
            ext = os.path.splitext(storage_path)[1] or ".jpg"
            dest = os.path.join(tmp, "photo{:02d}{}".format(i, ext))
            client.download_photo(storage_path, dest)
            local_paths.append(dest)

        log("  downloaded {} photo(s)".format(len(local_paths)))

        tiles = imageprep.collect_tiles(local_paths)
        provider = providers.get_provider(provider_name)
        workers = 4 if provider_name == "claude" else 1
        result = extract.extract_books(tiles, provider, max_workers=workers, progress=log)

        records = enrich.dedupe(result.books)
        lookup_errors = enrich.enrich(records, language=language, book_cache=book_cache)
        genre_classify.classify_genres_batch(records, progress=log)

        matched = sum(1 for r in records if r.matched)
        needs_review = sum(1 for r in records if r.needs_review)
        judaica_count = sum(
            1 for r in records
            if enrich.categorize(r.display_title, r.authors, r.genre,
                                  subjects=r.subjects, spine_text=r.spine_text,
                                  publisher=r.publisher) == "Judaica"
        )
        stats = {
            "provider": result.provider,
            "distinct_books": len(records),
            "raw_readings": len(result.books),
            "matched": matched,
            "needs_review": needs_review,
            "judaica_fraction": judaica_count / len(records) if records else 0.0,
            "tiles_suspect": len(result.tiles_truncated),
            "tiles_failed": len(result.tiles_failed),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": round(result.cost_usd, 4),
            # Persisted so a quota outage is queryable later - see
            # QUOTA_RETRY_SPEC.md.
            "lookup_errors": lookup_errors,
            "quota_exhausted": any("429" in m and "googlebooks" in m for m in lookup_errors),
        }

        status, flagged_reason = decide_route(stats)
        log("  routing decision: {} ({})".format(status, flagged_reason or "clean"))

        if status == "needs_admin":
            client.finish_batch(batch.id, "needs_admin", stats,
                                flagged_reason=flagged_reason)
            return status

        # 'ready': write one library_items draft per distinct book. Every
        # record needs a books.id - a confident match already has one (from
        # a cache hit or a fresh cacheable store); anything else - a match
        # too weak for the shared cache, or no match at all - gets its own
        # private placeholder row instead of polluting the shared cache.
        for record in records:
            book_id = record.book_id
            authors = record.authors or ([record.raw_author]
                                          if record.raw_author else [])
            if not book_id:
                # record.title may already carry a catalogue candidate's title
                # even here - matched=True sets it unconditionally, before
                # is_cacheable() gets a say (see pipeline/cache.py). Found
                # 2026-09-04: "Greatest Battles for Boys: The Vietnam War"
                # matched a real but subtitle-less candidate ("Greatest
                # Battles for Boys"), which failed the cache bar and fell
                # here - but display_title would have used that generic
                # title anyway, silently losing the one thing that
                # distinguished it from an already-owned, unrelated book
                # ("...The Ancient Wars"). raw_title is always what was
                # actually read off the spine - the only safe choice for a
                # private, untrusted placeholder.
                book_id = client.insert_placeholder_book(
                    batch.owner_id, record.raw_title, authors,
                    spine_text=record.spine_text, genre=record.genre)
            else:
                client.supersede_owner_placeholder(
                    batch.owner_id, book_id, record.display_title, authors)
            client.upsert_library_item(
                batch.owner_id, book_id,
                record.spine_text, record.legibility,
                scan_batch_id=batch.id,
                source_photo_path=resolve_source_photo(record.seen_in, batch.photo_paths))

        source = "auto_local" if provider_name == "ollama" else "auto_claude"
        client.finish_batch(batch.id, "ready", stats, processing_source=source)
        return status


def run_once(language="any", progress=None, provider_name="ollama"):
    """Process a single pending batch, if one exists. Returns True if it did."""
    log = progress or print
    client = SupabaseWorkerClient()
    pending = client.fetch_pending(limit=1)
    if not pending:
        return False

    batch = pending[0]
    log("Processing batch {} ({} photo(s))...".format(batch.id, len(batch.photo_paths)))
    client.claim(batch.id)
    status = process_batch(client, batch, language=language, progress=log,
                           provider_name=provider_name)
    log("Batch {} finished: {}".format(batch.id, status))
    return True
