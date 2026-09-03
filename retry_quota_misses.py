"""Nightly retry for books that missed a catalogue match only because
Google Books' daily quota was exhausted at scan time - not because the
model misread the spine or the book is genuinely obscure. See
QUOTA_RETRY_SPEC.md for the full writeup this implements.

Finds library_items whose book is still a private `is_placeholder` row
(no shared-cache match) from a scan_batch that was flagged
`quota_exhausted`, and re-runs catalogue lookup for each. On a fresh
match:
  - if it clears the shared-cache bar (`is_cacheable`): stores it in the
    shared `books` cache and repoints the library_item at that row - the
    same never-mutate-the-shared-cache, always-repoint pattern the "Fix
    title/author" web-app correction flow uses
    (`src/app/library/actions.ts`'s `correctBook`/`bulkCorrectBooks`).
  - if it matches but doesn't clear that bar: updates the existing private
    placeholder row's title/authors/isbn/cover_url in place. No repoint
    needed - it stays private either way.

A book is only retried for `RETRY_WINDOW_DAYS` after its batch ran - the
"give up" mechanism, cheaper than a retry-count column and generous
relative to Google's ~24h quota reset.

Enrichment only calls Open Library / Google Books / Supabase, none of
which needs the local Ollama box, so this can run anywhere (a scheduled
task after Google's midnight-Pacific quota reset is the intended use,
not tied to a machine that has to stay powered on for vision extraction).

    python retry_quota_misses.py
"""

import sys
from datetime import datetime, timedelta, timezone

import requests

from pipeline import config, enrich
from pipeline.cache import BookCache

RETRY_WINDOW_DAYS = 3


def _session():
    session = requests.Session()
    session.headers.update({
        "apikey": config.SUPABASE_KEY,
        "Authorization": "Bearer " + config.SUPABASE_KEY,
        "Content-Type": "application/json",
    })
    return session


def find_candidates(session):
    """library_items whose book is a private placeholder from a batch that
    hit the Google Books quota, within the retry window."""
    base = config.SUPABASE_URL.rstrip("/") + "/rest/v1"

    resp = session.get(base + "/scan_batches", params={
        "select": "id",
        "stats->>quota_exhausted": "eq.true",
    }, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    batch_ids = [row["id"] for row in resp.json()]
    if not batch_ids:
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETRY_WINDOW_DAYS)).isoformat()
    resp = session.get(base + "/library_items", params={
        "select": "id,scan_batch_id,book:books!inner(id,title,authors,is_placeholder,created_at)",
        "scan_batch_id": "in.({})".format(",".join(batch_ids)),
        "book.is_placeholder": "eq.true",
        "book.created_at": "gt." + cutoff,
    }, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def retry_one(record, book_cache):
    """Re-resolve one placeholder's title/author. Returns the record with
    `.matched`/`.book_id` set exactly as a fresh enrichment run would."""
    enrich.enrich([record], language=None, book_cache=book_cache)
    return record


def main():
    session = _session()
    candidates = find_candidates(session)
    print("{} placeholder book(s) eligible for a quota retry".format(len(candidates)))
    if not candidates:
        return 0

    book_cache = BookCache()
    base = config.SUPABASE_URL.rstrip("/") + "/rest/v1"

    recovered_shared = 0
    recovered_private = 0
    still_missing = 0

    for row in candidates:
        book = row["book"]
        authors = book.get("authors") or []
        record = enrich.CatalogueRecord(
            raw_title=book["title"],
            raw_author=authors[0] if authors else None,
            spine_text=book["title"],
            legibility="clear",
            seen_in=["quota-retry"],
        )
        retry_one(record, book_cache)

        if not record.matched:
            still_missing += 1
            continue

        if record.book_id:
            # Cleared the shared-cache bar - repoint at the new shared row,
            # same pattern the web app's correction flow uses. The old
            # placeholder row is left as-is (orphaned), same as every other
            # correction in this app - nothing else ever cleans those up.
            r = session.patch(base + "/library_items", params={"id": "eq." + row["id"]},
                              json={"book_id": record.book_id}, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            recovered_shared += 1
            print("  {!r} -> shared cache row {}".format(record.title, record.book_id))
        else:
            # Matched, but not confident/complete enough for the shared
            # cache bar - update the existing private placeholder in place.
            patch = {"title": record.title, "authors": record.authors}
            if record.isbn:
                key = "isbn13" if len(record.isbn) == 13 else "isbn10"
                patch[key] = record.isbn
            if record.cover_url:
                patch["cover_url"] = record.cover_url
            r = session.patch(base + "/books", params={"id": "eq." + book["id"]},
                              json=patch, timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            recovered_private += 1
            print("  {!r} -> updated in place (private)".format(record.title))

    print()
    print("{} recovered to the shared cache, {} recovered in place, {} still missing"
         .format(recovered_shared, recovered_private, still_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
