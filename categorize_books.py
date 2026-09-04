"""One-off (and re-runnable) backfill: assign books.category to existing rows.

Going forward, `pipeline/cache.py`'s BookCache.store() and
`pipeline/worker.py`'s insert_placeholder_book() both set category
automatically at write time. This script only exists to catch up rows
written before that existed - safe to re-run any time, it only touches rows
where category IS NULL.

For `is_placeholder` rows specifically, also pulls each one's
`library_items.scan_spine_text` (a placeholder maps 1:1 to exactly one
owner's library_items row - see insert_placeholder_book's docstring) and
feeds it to categorize() the same way the live write path now does. Added
2026-09-03 after `insert_placeholder_book()` itself was found dropping
spine_text - Maggid/Koren/Artscroll Judaica whose cleaned-up TITLE doesn't
carry the publisher name were coming back uncategorised. A non-placeholder
row with no category has no single spine_text to attribute (it can be
matched from many owners' different scans), so those keep the original
title/authors/genre/subjects-only call - if categorize() still returns None
for one of those, that's a genuine "no signal" case, not this bug.

2026-09-04: also runs the batched end-of-scan LLM genre pass
(`genre_classify.classify_genres_batch()`) over placeholder rows before
recomputing their category - the same pass `insert_placeholder_book()` now
threads through at write time (see PROJECT_SUMMARY.md), extended here to
reach placeholders written *before* that fix existed, which never got a
`genre` column value at all. A placeholder has no catalogue subjects, so
`classify_genre(subjects)` (the free keyword path) never has anything to
work with - this is the only way most of them will ever get a genre, and by
extension a Fiction/Non-fiction category. Best-effort: if Ollama isn't
reachable, `classify_genres_batch()` degrades to a no-op (as always), and
this script falls back to the title/spine keyword tier alone, same as
before this change.

    python categorize_books.py
"""

import sys

from pipeline import config, enrich, genre_classify
from pipeline.enrich import CatalogueRecord
import requests


def main():
    session = requests.Session()
    session.headers.update({
        "apikey": config.SUPABASE_KEY,
        "Authorization": "Bearer " + config.SUPABASE_KEY,
        "Content-Type": "application/json",
    })
    books_base = config.SUPABASE_URL.rstrip("/") + "/rest/v1/books"
    items_base = config.SUPABASE_URL.rstrip("/") + "/rest/v1/library_items"

    resp = session.get(books_base, params={
        "category": "is.null",
        "select": "id,title,authors,genre,subjects,is_placeholder",
        "limit": 1000,
    }, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()
    print("{} book(s) with no category".format(len(rows)))

    placeholder_ids = [row["id"] for row in rows if row["is_placeholder"]]
    spine_by_book_id = {}
    if placeholder_ids:
        resp = session.get(items_base, params={
            "book_id": "in.({})".format(",".join(placeholder_ids)),
            "select": "book_id,scan_spine_text",
            "limit": 1000,
        }, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        for item in resp.json():
            spine_by_book_id[item["book_id"]] = item["scan_spine_text"]

    # One batched LLM call classifies every placeholder in one shot, same
    # shape as a live scan - build a throwaway CatalogueRecord per row
    # (only the fields classify_genres_batch/categorize actually read are
    # populated) so the exact same code path runs here as at write time.
    record_by_book_id = {}
    for row in rows:
        if not row["is_placeholder"]:
            continue
        authors = row.get("authors") or []
        record_by_book_id[row["id"]] = CatalogueRecord(
            raw_title=row["title"], raw_author=(authors[0] if authors else None),
            spine_text=spine_by_book_id.get(row["id"]) or "", legibility="clear",
            seen_in=["categorize_books_backfill"], title=row["title"], authors=authors,
            subjects=row.get("subjects") or [],
        )
    placeholder_records = list(record_by_book_id.values())
    if placeholder_records:
        genre_classify.classify_genres_batch(placeholder_records, progress=print)

    counts = {"Fiction": 0, "Non-fiction": 0, "Judaica": 0, None: 0}
    for row in rows:
        record = record_by_book_id.get(row["id"])
        spine_text = spine_by_book_id.get(row["id"]) if row["is_placeholder"] else None
        genre = record.genre if record is not None else row.get("genre")
        category = enrich.categorize(row["title"], row.get("authors") or [],
                                     genre, row.get("subjects") or [],
                                     spine_text=spine_text)
        counts[category] += 1
        if category is None:
            continue
        patch = {"category": category}
        if record is not None:
            patch["genre"] = genre
        r = session.patch(books_base, params={"id": "eq." + row["id"]},
                          json=patch, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()

    print("Fiction: {}, Non-fiction: {}, Judaica: {}, left blank: {}".format(
        counts["Fiction"], counts["Non-fiction"], counts["Judaica"], counts[None]))


if __name__ == "__main__":
    sys.exit(main())
