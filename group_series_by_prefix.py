"""One-off (and re-runnable) backfill: group existing multi-book sets whose
volumes carry no "Vol. N" marker at all, only a shared series-name prefix
before each volume's own subtitle - e.g. "A Daily Dose of Torah: Daily study
for the weeks of X-Y". detect_series() alone can only tag a volume that
literally says "Vol. N"; this catches the real sets it can't, using
pipeline/enrich.py's cluster_series_by_prefix() (see its docstring and
PROJECT_SUMMARY.md #10 for the full design/reasoning).

Unlike backfill_series.py this isn't wired into the live write path - it
needs to see an owner's *whole* shelf to find a shared prefix, which a
single incoming scan record doesn't have. Safe to re-run any time: it only
writes rows whose series_title actually needs to change.

    python group_series_by_prefix.py
"""

import sys

from pipeline import config, enrich
import requests


def main():
    session = requests.Session()
    session.headers.update({
        "apikey": config.SUPABASE_KEY,
        "Authorization": "Bearer " + config.SUPABASE_KEY,
        "Content-Type": "application/json",
    })
    items_base = config.SUPABASE_URL.rstrip("/") + "/rest/v1/library_items"
    books_base = config.SUPABASE_URL.rstrip("/") + "/rest/v1/books"

    resp = session.get(items_base, params={
        "select": "owner_id,book:books(id,title,series_title)",
        "limit": 1000,
    }, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()

    books = [{
        "id": row["book"]["id"],
        "title": row["book"]["title"],
        "series_title": row["book"]["series_title"],
        "owner_id": row["owner_id"],
    } for row in rows if row.get("book")]
    print("{} library book(s) considered".format(len(books)))

    updates = enrich.cluster_series_by_prefix(books)
    print("{} book(s) need a series_title update".format(len(updates)))

    for book_id, series_title in updates:
        r = session.patch(books_base, params={"id": "eq." + book_id},
                          json={"series_title": series_title},
                          timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        print("  {} -> {!r}".format(book_id, series_title))


if __name__ == "__main__":
    sys.exit(main())
