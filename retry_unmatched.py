"""Re-run catalogue lookup for the books a scan failed to match.

A scan can leave books unmatched for two very different reasons: the model
misread the spine, or the catalogue was unreachable at that moment. The second
kind is recoverable and costs nothing but a retry - and after a run that
exhausted the Google Books daily quota, most of them are the second kind.

Only the unmatched books are re-queried, so this does not burn quota re-checking
books that are already correct.

    python retry_unmatched.py out/local_all.json
"""

import json
import shutil
import sys

from pipeline import enrich, report
from pipeline.enrich import CatalogueRecord


def rebuild(book):
    """Reconstruct a CatalogueRecord from a saved scan entry."""
    raw = book["raw"]
    return CatalogueRecord(
        raw_title=raw["title"] or book["title"],
        raw_author=raw["author"],
        spine_text=raw["spine_text"],
        legibility=raw["legibility"],
        seen_in=raw.get("seen_in", []),
        times_seen=raw.get("times_seen", 1),
    )


def restore(book):
    """Rebuild a record that already matched, without re-querying it."""
    record = rebuild(book)
    record.matched = book["matched"]
    record.match_score = book["match_score"]
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
        payload = json.load(handle)

    books = payload["books"]
    stats = payload.get("stats", {})

    matched_before = [b for b in books if b["matched"]]
    unmatched = [b for b in books if not b["matched"]]

    print("{} books total: {} matched, {} unmatched".format(
        len(books), len(matched_before), len(unmatched)))
    if not unmatched:
        print("Nothing to retry.")
        return 0

    print("Re-querying only the {} unmatched...".format(len(unmatched)))
    retry_records = [rebuild(b) for b in unmatched]
    errors = enrich.enrich(retry_records, language=None, progress=None)

    recovered = [r for r in retry_records if r.matched]
    print()
    print("RECOVERED: {}/{} ({:.0f}% of the previously unmatched)".format(
        len(recovered), len(retry_records),
        len(recovered) / len(retry_records) * 100))
    if errors:
        print("Lookup errors this time: {}".format(errors))

    for record in recovered[:15]:
        print("   {:<42} -> {}".format(record.raw_title[:41], (record.title or "")[:32]))
    if len(recovered) > 15:
        print("   ... and {} more".format(len(recovered) - 15))

    still_missing = len(retry_records) - len(recovered)
    total_matched = len(matched_before) + len(recovered)
    print()
    print("MATCH RATE: {}/{} ({:.0f}%)  <- was {}/{} ({:.0f}%)".format(
        total_matched, len(books), total_matched / len(books) * 100,
        len(matched_before), len(books), len(matched_before) / len(books) * 100))
    print("Still unmatched: {} - these are genuine misreads or obscure editions."
          .format(still_missing))

    # Write the improved catalogue back, keeping a copy of the original.
    records = [restore(b) for b in matched_before] + retry_records
    stats["matched"] = total_matched
    stats["needs_review"] = sum(1 for r in records if r.needs_review)
    stats["retried"] = len(retry_records)

    shutil.copyfile(path, path + ".before-retry")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report.to_json(records, stats))
    html_path = path.replace(".json", ".html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(report.to_html(records, stats))

    print()
    print("Updated {} and {} (original kept as {}.before-retry)".format(
        path, html_path, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
