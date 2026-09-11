"""Run one shelf-scan batch through Claude/ChatGPT by hand.

Usable right now, with no web app and no scan_batches table yet - point it at
a folder of photos, paste the generated prompt into Claude.ai or ChatGPT along
with the listed photos, then feed the reply back in to get the same catalogue
JSON/HTML report the automated pipeline produces.

    python manual_batch.py prepare samples/ --batch-id shelf1
        -> prints the copy-paste packet

    python manual_batch.py ingest samples/ --batch-id shelf1 --reply reply.txt
        -> writes out/shelf1.json / out/shelf1.html, same shape as any scan,
           but ONLY as local files - nothing is written to Supabase yet.

    python manual_batch.py publish samples/ --batch-id shelf1 --reply reply.txt \\
            --owner-name neighbour1
        -> does everything 'ingest' does, PLUS uploads the photos to that
           owner's Storage prefix and writes real (draft) library_items rows,
           same as the automated worker would. Use this for photos that came
           in outside the (not-yet-built) upload web app - e.g. someone
           emailed or WhatsApped their shelf photos to you.

    python manual_batch.py download <real-batch-id> --dest tmp/shelf1
        -> for a batch that already exists in Supabase (a real owner uploaded
           it through the actual web app) - downloads its photos locally and
           prints the same copy-paste packet 'prepare' would, so you (or
           Claude, in a session with vision) can read them directly instead
           of running the automated worker.

    python manual_batch.py finish <real-batch-id> --reply reply.txt
        -> the 'publish' equivalent for a batch 'download' already fetched:
           writes real (draft) library_items rows against that SAME existing
           scan_batches row (no new batch, no re-upload) and marks it
           'imported'. Use this instead of 'publish' whenever the batch
           already exists - 'publish' always creates a brand-new batch and
           would leave the real one stuck at 'uploaded' forever.
"""

import argparse
import glob
import os
import sys

from pipeline import cache, enrich, genre_classify, report
from pipeline.imageprep import IMAGE_SUFFIXES
from pipeline.manual import build_manual_packet, books_to_extracted, ingest_manual_reply
from pipeline.worker import SupabaseWorkerClient, resolve_source_photo

# A batch is safe to 'finish' if it's sitting exactly where the automated
# worker would find it (never processed) or where it left off after a crash
# mid-processing. Anything past that already has a real outcome recorded.
_FINISHABLE_STATUSES = {"uploaded", "processing"}


def _find_photos(path):
    if os.path.isdir(path):
        found = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES:
                found.append(full)
        return found
    return [path]


def cmd_prepare(args):
    photos = _find_photos(args.photos)
    if not photos:
        print("No photos found in {!r}".format(args.photos))
        return 1
    packet = build_manual_packet(args.batch_id, photos)
    print(packet.render())
    print("-" * 70, file=sys.stderr)
    print("Paste everything above into Claude.ai or ChatGPT, attach the "
         "{} photo(s) listed, send it, then run:".format(len(photos)),
         file=sys.stderr)
    print("    python manual_batch.py ingest {} --batch-id {} --reply <file>"
         .format(args.photos, args.batch_id), file=sys.stderr)
    return 0


def _parse_and_enrich(reply_path, batch_id, language, no_cache):
    """Shared by 'ingest' and 'publish': reply text -> deduped, enriched
    records + a stats dict in the same shape scan_batches.stats expects."""
    with open(reply_path, encoding="utf-8") as handle:
        raw = handle.read()

    books, truncated = ingest_manual_reply(raw)
    if not books:
        return None, None, truncated

    extracted = books_to_extracted(books, "manual:" + batch_id)
    print("{} books parsed from the manual reply".format(len(extracted)))
    if truncated:
        print("Note: the pasted reply looked like it hit a repetition loop - "
             "double-check nothing is missing.")

    records = enrich.dedupe(extracted)
    print("{} distinct after dedupe".format(len(records)))

    book_cache = None if no_cache else cache.BookCache()
    if book_cache is not None and not book_cache.enabled:
        book_cache = None

    # defer_store=True: classify_genres_batch() below sets record.llm_genre for
    # records the keyword classifier couldn't place - storing to the shared
    # cache before that runs would permanently cache "Uncategorised" for those
    # books (found 2026-09-11). Store explicitly once classification is done,
    # same fix as pipeline/worker.py's process_batch().
    lookup_errors = enrich.enrich(records, language=language, book_cache=book_cache,
                                  defer_store=True)
    matched = sum(1 for r in records if r.matched)
    needs_review = sum(1 for r in records if r.needs_review)
    print("{}/{} matched a catalogue record".format(matched, len(records)))
    if lookup_errors:
        print("lookup errors:", lookup_errors)

    genre_classify.classify_genres_batch(records, progress=print)
    if book_cache is not None:
        for record in records:
            if record.source != "cache":
                book_cache.store(record)
        print(book_cache.summary())

    stats = {
        "provider": "manual (admin-processed via chat UI)",
        "distinct_books": len(records),
        "raw_readings": len(extracted),
        "tiles_processed": 0,
        "tiles_failed": 0,
        "tiles_suspect": 1 if truncated else 0,
        "tiles_truncated": 1 if truncated else 0,
        "matched": matched,
        "needs_review": needs_review,
        "cost_usd": 0.0,
        "elapsed": 0.0,
        # Persisted so a quota outage is queryable later, not just visible in
        # whoever's terminal happened to run this - see QUOTA_RETRY_SPEC.md.
        "lookup_errors": lookup_errors,
        "quota_exhausted": any("429" in m and "googlebooks" in m for m in lookup_errors),
    }
    return records, stats, truncated


def _write_local_report(records, stats, batch_id):
    out_path = os.path.join("out", batch_id)
    os.makedirs("out", exist_ok=True)
    with open(out_path + ".json", "w", encoding="utf-8") as handle:
        handle.write(report.to_json(records, stats))
    with open(out_path + ".html", "w", encoding="utf-8") as handle:
        handle.write(report.to_html(records, stats))
    print()
    print("wrote {}.json / {}.html".format(out_path, out_path))


def cmd_ingest(args):
    records, stats, truncated = _parse_and_enrich(
        args.reply, args.batch_id, args.language, args.no_cache)
    if records is None:
        print("No books parsed from the reply. Check it actually contains "
             "the JSON block.")
        return 1
    _write_local_report(records, stats, args.batch_id)
    return 0


def cmd_publish(args):
    records, stats, truncated = _parse_and_enrich(
        args.reply, args.batch_id, args.language, args.no_cache)
    if records is None:
        print("No books parsed from the reply. Check it actually contains "
             "the JSON block.")
        return 1
    _write_local_report(records, stats, args.batch_id)

    photos = _find_photos(args.photos)
    if not photos:
        print("No photos found in {!r} - can't publish without the source "
             "photos (they still need to land in Storage).".format(args.photos))
        return 1

    client = SupabaseWorkerClient()
    owner_id = args.owner_id or client.resolve_owner_id(args.owner_name)

    print("uploading {} photo(s) to Storage for owner {}...".format(
        len(photos), owner_id))
    storage_paths = [client.upload_photo(owner_id, args.batch_id, path, i)
                     for i, path in enumerate(photos)]

    batch_db_id = client.create_batch(owner_id, storage_paths)
    print("created scan_batches row {}".format(batch_db_id))

    for record in records:
        book_id = record.book_id
        authors = record.authors or ([record.raw_author] if record.raw_author else [])
        if not book_id:
            # raw_title, not display_title - a rejected-for-caching match can
            # still have overwritten record.title with a generic candidate
            # title (matched=True sets it before is_cacheable() runs). See
            # pipeline/worker.py's process_batch() for the full incident.
            book_id = client.insert_placeholder_book(
                owner_id, record.raw_title, authors,
                spine_text=record.spine_text, genre=record.genre)
        else:
            client.supersede_owner_placeholder(
                owner_id, book_id, record.display_title, authors)
        client.upsert_library_item(owner_id, book_id, record.spine_text, record.legibility,
                                   scan_batch_id=batch_db_id,
                                   source_photo_path=resolve_source_photo(
                                       record.seen_in, storage_paths))

    client.finish_batch(batch_db_id, "imported", stats,
                        processing_source="admin_manual")
    print("published {} book(s) as private drafts for owner {}"
         .format(len(records), owner_id))
    print("(shareable=false, confirmed=false - the owner still needs to "
         "review and approve each one; nothing is visible to neighbours yet)")
    return 0


def cmd_download(args):
    client = SupabaseWorkerClient()
    batch = client.get_batch(args.batch_id)
    if batch.status not in _FINISHABLE_STATUSES:
        print("Note: batch {} is already '{}', not 'uploaded' - it was "
             "presumably already processed once (see scan_batches.stats). "
             "Downloading again is harmless, but 'finish' will refuse to "
             "write a second time without --force.".format(batch.id, batch.status),
             file=sys.stderr)

    os.makedirs(args.dest, exist_ok=True)
    local_paths = []
    for i, storage_path in enumerate(batch.photo_paths):
        ext = os.path.splitext(storage_path)[1] or ".jpg"
        dest = os.path.join(args.dest, "photo{:02d}{}".format(i, ext))
        client.download_photo(storage_path, dest)
        local_paths.append(dest)
    print("downloaded {} photo(s) to {}\n".format(len(local_paths), args.dest), file=sys.stderr)

    packet = build_manual_packet(args.batch_id, local_paths)
    print(packet.render())
    print("-" * 70, file=sys.stderr)
    print("Read the photos above (or paste this packet into Claude.ai/ChatGPT "
         "with them attached), save the JSON reply to a file, then run:",
         file=sys.stderr)
    print("    python manual_batch.py finish {} --reply <file>"
         .format(args.batch_id), file=sys.stderr)
    return 0


def cmd_finish(args):
    client = SupabaseWorkerClient()
    batch = client.get_batch(args.batch_id)
    if batch.status not in _FINISHABLE_STATUSES and not args.force:
        print("Batch {} is already '{}', not 'uploaded'/'processing' - looks "
             "already finished. Pass --force to write anyway (e.g. you're "
             "deliberately reprocessing it).".format(args.batch_id, batch.status),
             file=sys.stderr)
        return 1

    records, stats, truncated = _parse_and_enrich(
        args.reply, args.batch_id, args.language, args.no_cache)
    if records is None:
        print("No books parsed from the reply. Check it actually contains "
             "the JSON block.")
        return 1
    _write_local_report(records, stats, args.batch_id)

    client.claim(args.batch_id)
    for record in records:
        book_id = record.book_id
        authors = record.authors or ([record.raw_author] if record.raw_author else [])
        if not book_id:
            # raw_title, not display_title - see process_batch() in
            # pipeline/worker.py for the full incident this guards against.
            book_id = client.insert_placeholder_book(
                batch.owner_id, record.raw_title, authors,
                spine_text=record.spine_text, genre=record.genre)
        else:
            client.supersede_owner_placeholder(
                batch.owner_id, book_id, record.display_title, authors)
        client.upsert_library_item(batch.owner_id, book_id, record.spine_text, record.legibility,
                                   scan_batch_id=args.batch_id,
                                   source_photo_path=resolve_source_photo(
                                       record.seen_in, batch.photo_paths))

    client.finish_batch(args.batch_id, "imported", stats, processing_source="admin_manual")
    print("finished batch {} for owner {} - {} book(s) written as private drafts"
         .format(args.batch_id, batch.owner_id, len(records)))
    print("(shareable=false, confirmed=false - the owner still needs to "
         "review and approve each one; nothing is visible to neighbours yet)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("prepare", help="Generate the copy-paste packet for a batch.")
    p1.add_argument("photos", help="A folder of photos, or a single file.")
    p1.add_argument("--batch-id", required=True)

    p2 = sub.add_parser("ingest", help="Parse a pasted chat reply into a catalogue.")
    p2.add_argument("photos", help="Same folder given to 'prepare' (used only for the batch id/report naming).")
    p2.add_argument("--batch-id", required=True)
    p2.add_argument("--reply", required=True, help="Text file containing the pasted chat reply.")
    p2.add_argument("--language", default="eng")
    p2.add_argument("--no-cache", action="store_true")

    p3 = sub.add_parser("publish", help="Ingest, then write real (draft) "
                        "library_items rows for a specific owner.")
    p3.add_argument("photos", help="Folder (or file) of the actual photos - "
                    "these get uploaded to that owner's Storage prefix.")
    p3.add_argument("--batch-id", required=True)
    p3.add_argument("--reply", required=True, help="Text file containing the pasted chat reply.")
    p3.add_argument("--language", default="eng")
    p3.add_argument("--no-cache", action="store_true")
    owner = p3.add_mutually_exclusive_group(required=True)
    owner.add_argument("--owner-id", help="The owner's profile UUID.")
    owner.add_argument("--owner-name", help="The owner's profiles.display_name "
                       "(looked up for you - convenient but exact-match only).")

    p4 = sub.add_parser("download", help="Fetch an existing batch's photos "
                        "locally and print the same packet 'prepare' would.")
    p4.add_argument("batch_id", help="A real scan_batches.id (e.g. one sitting "
                    "at status='uploaded' from a real web upload).")
    p4.add_argument("--dest", required=True, help="Local folder to save photos into.")

    p5 = sub.add_parser("finish", help="Ingest, then write real (draft) "
                        "library_items against an EXISTING batch (from "
                        "'download') and mark it done - unlike 'publish', "
                        "does not create a new batch or re-upload photos.")
    p5.add_argument("batch_id", help="The same real scan_batches.id given to 'download'.")
    p5.add_argument("--reply", required=True, help="Text file containing the pasted chat reply.")
    p5.add_argument("--language", default="eng")
    p5.add_argument("--no-cache", action="store_true")
    p5.add_argument("--force", action="store_true",
                    help="Write anyway even if the batch looks already finished.")

    args = parser.parse_args()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "publish":
        return cmd_publish(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "finish":
        return cmd_finish(args)
    return cmd_ingest(args)


if __name__ == "__main__":
    sys.exit(main())
