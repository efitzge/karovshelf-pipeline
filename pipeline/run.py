"""CLI entry point: photos of a bookshelf in, catalogue out.

    python -m pipeline.run samples/ --provider ollama --out out/local
    python -m pipeline.run samples/ --provider claude --out out/claude

Writes a JSON catalogue and an HTML accuracy report next to each other.
"""

import argparse
import os
import sys
import time

from . import cache, config, enrich, extract, genre_classify, imageprep, providers, report


def _log(message):
    print(message, flush=True)


def _force_utf8_output():
    """Windows consoles default to cp1252, which cannot print most book titles.

    Without this, a single accented or non-Latin title crashes the whole run
    with a UnicodeEncodeError at the very end - after the work is done.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pipeline.run",
        description="Catalogue a bookshelf from photos or video using a vision model.",
    )
    parser.add_argument("inputs", nargs="+",
                        help="Image files, video files, or directories of them.")
    parser.add_argument("--out", default="out/scan",
                        help="Output path prefix (default: out/scan).")
    parser.add_argument("--provider", default="ollama", choices=sorted(providers.PROVIDERS),
                        help="Vision backend. 'ollama' runs locally and is free; "
                             "'claude' is hosted and billed per token "
                             "(default: %(default)s).")
    parser.add_argument("--model", default=None,
                        help="Override the provider's model name.")
    parser.add_argument("--tiles", type=int, default=None,
                        help="Cap the tile grid. Default: derived from the source "
                             "resolution. Fewer tiles is cheaper and faster but "
                             "less legible.")
    parser.add_argument("--effort", default=config.DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Claude only: reasoning effort (default: %(default)s).")
    parser.add_argument("--every-seconds", type=float, default=1.5,
                        help="For video input, seconds between sampled frames.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel tile reads. Defaults to 4 for Claude and 1 "
                             "for Ollama, which is GPU-bound and gains nothing "
                             "from concurrency.")
    parser.add_argument("--language", default="eng",
                        help="Prefer catalogue records in this language "
                             "(default: eng). Use 'any' to disable - necessary if "
                             "the shelves include non-English books.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip the shared Supabase book cache and query the "
                             "external catalogues directly.")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip catalogue lookup; show raw model output only.")
    parser.add_argument("--no-genre-classify", action="store_true",
                        help="Skip the batched end-of-scan LLM genre classification "
                             "pass for books the catalogue left Uncategorised.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any checkpoint and re-read every tile.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N tiles. Useful for a quick "
                             "sanity check before committing to a full run.")
    return parser


def _build_provider(args):
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.provider == "claude":
        kwargs["effort"] = args.effort
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            _log("ANTHROPIC_API_KEY is not set, and --provider claude needs it.")
            _log("Either add it to .env, or use the free local backend:")
            _log("    python -m pipeline.run samples/ --provider ollama")
            return None
    return providers.get_provider(args.provider, **kwargs)


def main(argv=None):
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    language = None if args.language.lower() in ("any", "none", "") else args.language

    try:
        provider = _build_provider(args)
    except providers.ProviderError as exc:
        _log("Error: {}".format(exc))
        return 2
    if provider is None:
        return 2

    started = time.time()

    _log("Preparing images...")
    try:
        tiles = imageprep.collect_tiles(
            args.inputs, tile_count=args.tiles, every_seconds=args.every_seconds
        )
    except RuntimeError as exc:
        _log("Error: {}".format(exc))
        return 1

    if not tiles:
        _log("No images or videos found in: " + ", ".join(args.inputs))
        _log("Supported: " + ", ".join(sorted(imageprep.IMAGE_SUFFIXES)))
        return 1

    if args.limit:
        tiles = tiles[:args.limit]

    est_tokens = sum(t.estimated_tokens() for t in tiles)
    _log("  {} tiles from {} source(s)".format(
        len(tiles), len({t.source for t in tiles})))
    if provider.costs_money:
        _log("  ~{:,} input tokens (~${:.3f} before output)".format(
            est_tokens, est_tokens / 1_000_000 * config.PRICE_IN_PER_MTOK))

    workers = args.workers
    if workers is None:
        workers = 4 if args.provider == "claude" else 1

    # Long local runs get interrupted - sleeping laptops, closed lids, thermal
    # shutdowns. Save each tile as it lands so none of that work is lost.
    checkpoint = None if args.no_resume else args.out + ".progress.jsonl"

    _log("Reading spines with {}...".format(provider.label()))
    result = extract.extract_books(tiles, provider, max_workers=workers,
                                   progress=_log, checkpoint=checkpoint)

    if result.tiles_failed:
        _log("  {} tile(s) failed:".format(len(result.tiles_failed)))
        for label, reason in result.tiles_failed:
            _log("    {} - {}".format(label, reason))

    if result.tiles_truncated:
        _log("  {} tile(s) hit a repetition loop and may be missing books:"
             .format(len(result.tiles_truncated)))
        for label in result.tiles_truncated:
            _log("    " + label)
        _log("  These tiles read some books correctly before the model got")
        _log("  stuck repeating one title. Whatever was physically further")
        _log("  along that shelf section was not read. Consider re-photographing")
        _log("  those regions, or re-running with --no-resume to retry them.")

    if not result.books:
        _log("No books extracted. Check that the images show book spines, and "
             "that the model is actually a vision model.")
        return 1

    _log("  {} raw readings".format(len(result.books)))

    records = enrich.dedupe(result.books)
    _log("  {} distinct books after merging overlapping tiles".format(len(records)))

    if args.no_enrich:
        _log("Skipping catalogue lookup (--no-enrich).")
    else:
        book_cache = None if args.no_cache else cache.BookCache()
        if book_cache is not None and book_cache.enabled:
            _log("Matching (shared cache first, then Open Library / Google Books)...")
        else:
            _log("Matching against Open Library / Google Books...")
            book_cache = None

        lookup_errors = enrich.enrich(records, language=language, progress=_log,
                                      book_cache=book_cache)
        matched = sum(1 for r in records if r.matched)
        _log("  {}/{} matched a catalogue record".format(matched, len(records)))
        if book_cache is not None:
            _log("  " + book_cache.summary())
            for message in book_cache.errors:
                _log("    cache problem: " + message)

        if lookup_errors:
            _log("  catalogue lookup problems (these depress the match rate and")
            _log("  are NOT the model misreading spines):")
            for message, count in sorted(lookup_errors.items(), key=lambda kv: -kv[1]):
                _log("    {}x  {}".format(count, message))
            if any("429" in m and "googlebooks" in m for m in lookup_errors):
                _log("  Google Books is rate-limiting. A free key raises the")
                _log("  ceiling to ~1000/day - see README.")

        if not args.no_genre_classify:
            genre_classify.classify_genres_batch(records, progress=_log)

    elapsed = time.time() - started
    result.elapsed = elapsed
    stats = {
        "provider": result.provider,
        "tiles_processed": result.tiles_processed,
        "tiles_failed": len(result.tiles_failed),
        "tiles_truncated": len(result.tiles_truncated),
        "raw_readings": len(result.books),
        "distinct_books": len(records),
        "matched": sum(1 for r in records if r.matched),
        "needs_review": sum(1 for r in records if r.needs_review),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 4),
        "elapsed": round(elapsed, 1),
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    json_path = args.out + ".json"
    html_path = args.out + ".html"
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(report.to_json(records, stats))
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(report.to_html(records, stats))

    money = "${:.3f} spent".format(result.cost_usd) if result.priced \
        else "free (ran locally)"
    _log("")
    _log("Done in {:.0f}s - {} books, {} flagged for review, {}.".format(
        elapsed, len(records), stats["needs_review"], money))
    _log("  " + json_path)
    _log("  " + html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
