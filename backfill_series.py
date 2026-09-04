"""One-off (and re-runnable) backfill: assign books.series_title/
series_volume to existing rows.

Going forward, pipeline/cache.py's BookCache.store() and pipeline/worker.py's
insert_placeholder_book() both set these automatically at write time (see
PROJECT_SUMMARY.md #10, "condense N volumes into one card" display grouping).
This script only exists to catch up rows written before that existed - safe
to re-run any time, it only touches rows where series_title IS NULL.

    python backfill_series.py
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
    books_base = config.SUPABASE_URL.rstrip("/") + "/rest/v1/books"

    resp = session.get(books_base, params={
        "series_title": "is.null",
        "select": "id,title",
        "limit": 1000,
    }, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()
    print("{} book(s) with no series_title yet".format(len(rows)))

    updated = 0
    for row in rows:
        series = enrich.detect_series(row["title"])
        if not series:
            continue
        base, volume = series
        r = session.patch(books_base, params={"id": "eq." + row["id"]},
                          json={"series_title": base, "series_volume": volume},
                          timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        updated += 1
        print("  {!r} -> {!r} vol. {}".format(row["title"], base, volume))

    print("{} book(s) tagged with a detected series".format(updated))


if __name__ == "__main__":
    sys.exit(main())
