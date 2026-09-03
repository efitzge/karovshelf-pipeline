"""Push an existing scan's resolved books into the shared cache.

Uses the live cache client rather than generated SQL, so the same quality bar
that governs a normal scan governs a bulk import - there is no back door that
writes rows the pipeline itself would have refused.

    python push_cache.py out/local_all.json
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline import cache
from pipeline.enrich import CatalogueRecord


def rebuild(book):
    raw = book["raw"]
    record = CatalogueRecord(
        raw_title=raw["title"] or book["title"],
        raw_author=raw["author"],
        spine_text=raw["spine_text"],
        legibility=raw["legibility"],
        seen_in=raw.get("seen_in", []),
        times_seen=raw.get("times_seen", 1),
    )
    record.matched = book["matched"]
    record.match_score = book["match_score"] or 0
    record.source = book["match_source"]
    record.title = book["title"]
    record.authors = book["authors"]
    record.first_published = book["first_published"]
    record.isbn = book["isbn"]
    record.cover_url = book["cover_url"]
    record.subjects = book["subjects"]
    return record


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/local_all.json"
    with open(path, encoding="utf-8") as handle:
        books = json.load(handle)["books"]

    book_cache = cache.BookCache()
    if not book_cache.enabled:
        print("Cache is not configured - set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        return 2

    records = [rebuild(b) for b in books if b["matched"]]
    eligible = [r for r in records if cache.is_cacheable(r)]

    print("{} matched books; {} pass the cache quality bar ({} withheld)".format(
        len(records), len(eligible), len(records) - len(eligible)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(book_cache.store, r) for r in eligible]
        for future in as_completed(futures):
            future.result()

    print("stored: {}".format(book_cache.stores))
    for message in book_cache.errors:
        print("  problem: {}".format(message))
    return 0


if __name__ == "__main__":
    sys.exit(main())
