"""One-off (and re-runnable) backfill: run the batched LLM genre classifier
against existing books the catalogue left Uncategorised.

Going forward, every scan (CLI, the automated worker, and the manual/Claude
path) runs pipeline/genre_classify.py automatically after catalogue
matching. This script only exists to catch up rows written before that
existed - safe to re-run any time, it only touches rows where
genre IS NULL or genre = 'Uncategorised', and only ever fills in a genre,
never overwrites one a previous run (or the keyword classifier) already set.

    python classify_genres_backfill.py
"""

import sys

from pipeline import config, genre_classify
from pipeline.enrich import CatalogueRecord
import requests


def main():
    session = requests.Session()
    session.headers.update({
        "apikey": config.SUPABASE_KEY,
        "Authorization": "Bearer " + config.SUPABASE_KEY,
        "Content-Type": "application/json",
    })
    base = config.SUPABASE_URL.rstrip("/") + "/rest/v1/books"

    resp = session.get(base, params={
        "or": "(genre.is.null,genre.eq.Uncategorised)",
        "select": "id,title,authors,subjects",
        "limit": 1000,
    }, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    rows = [r for r in resp.json() if (r.get("title") or "").strip()]
    print("{} book(s) with no genre".format(len(rows)))
    if not rows:
        return 0

    records = [
        CatalogueRecord(
            raw_title=row["title"], raw_author=None, spine_text=row["title"],
            legibility="clear", seen_in=["backfill"], matched=True,
            title=row["title"], authors=row.get("authors") or [],
            subjects=row.get("subjects") or [],
        )
        for row in rows
    ]

    classified = genre_classify.classify_genres_batch(records, progress=print)
    print("{}/{} classified by the model; {} left Uncategorised".format(
        classified, len(rows), len(rows) - classified))

    for row, record in zip(rows, records):
        if not record.llm_genre:
            continue
        r = session.patch(base, params={"id": "eq." + row["id"]},
                          json={"genre": record.llm_genre}, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        print("  {!r} -> {}".format(row["title"], record.llm_genre))

    return 0


if __name__ == "__main__":
    sys.exit(main())
