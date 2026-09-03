"""Drive a vision provider over the tiles of a shelf photo.

This module owns the orchestration - parallelism, error collection, cost
accounting - and knows nothing about which model is doing the reading. The
model-specific parts live in `providers.py`.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import config
from .providers import ProviderError


@dataclass
class ExtractedBook:
    spine_text: str
    title: str
    author: str | None
    legibility: str
    orientation: str
    source: str  # which tile it was first seen in
    seen_in: list = field(default_factory=list)


@dataclass
class ExtractionResult:
    books: list
    input_tokens: int
    output_tokens: int
    tiles_processed: int
    tiles_failed: list
    tiles_truncated: list
    provider: str
    elapsed: float = 0.0
    priced: bool = True

    @property
    def cost_usd(self):
        """Zero for a local provider - that is the entire point of having one."""
        if not self.priced:
            return 0.0
        return (
            self.input_tokens / 1_000_000 * config.PRICE_IN_PER_MTOK
            + self.output_tokens / 1_000_000 * config.PRICE_OUT_PER_MTOK
        )


def load_checkpoint(path):
    """Read an in-progress run's saved tiles, keyed by tile label."""
    done = {}
    if not path or not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written final line after a hard kill
            if "tile" in entry:
                done[entry["tile"]] = entry
    return done


def _append_checkpoint(path, entry):
    """Append one tile's result and flush it to disk immediately.

    Flushing per tile is the whole point: a scan of a large collection runs for
    tens of minutes, and a sleeping laptop or a closed lid must not throw away
    work that was already paid for in time and heat.
    """
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_books(tiles, provider, max_workers=4, progress=None, checkpoint=None):
    """Read every tile with `provider`, in parallel.

    A local model is usually GPU-bound and serialises anyway, so callers should
    pass max_workers=1 for Ollama; parallelism there just fights for VRAM.

    If `checkpoint` is given, each tile's result is written to that file as soon
    as it lands, and any tile already recorded there is skipped - so an
    interrupted run resumes instead of starting over.
    """
    all_books = []
    failures = []
    total_in = 0
    total_out = 0

    already = load_checkpoint(checkpoint)
    pending = []
    resumed_truncated = []
    for tile in tiles:
        entry = already.get(tile.label)
        if entry is None:
            pending.append(tile)
            continue
        for item in entry.get("books", []):
            all_books.append(ExtractedBook(
                spine_text=item.get("spine_text", ""),
                title=item.get("title", ""),
                author=item.get("author"),
                legibility=item.get("legibility", "guess"),
                orientation=item.get("orientation", "vertical"),
                source=tile.label,
                seen_in=[tile.label],
            ))
        total_in += entry.get("input_tokens", 0)
        total_out += entry.get("output_tokens", 0)
        if entry.get("truncated"):
            resumed_truncated.append(tile.label)

    resumed = len(tiles) - len(pending)
    if resumed and progress:
        progress("  resuming: {} tile(s) already done, {} to go".format(
            resumed, len(pending)))

    truncated_tiles = list(resumed_truncated)

    def work(tile):
        return tile, provider.read_tile(tile)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(work, tile): tile for tile in pending}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                _, (raw_books, tokens_in, tokens_out, truncated) = future.result()
            except ProviderError as exc:
                failures.append((tile.label, str(exc)))
                # Record the failure too, so a resume does not retry a tile the
                # model reliably chokes on - and so the count stays honest.
                _append_checkpoint(checkpoint, {
                    "tile": tile.label, "books": [], "error": str(exc),
                    "input_tokens": 0, "output_tokens": 0,
                })
                if progress:
                    progress("  {}: FAILED ({})".format(tile.label, exc))
                continue

            _append_checkpoint(checkpoint, {
                "tile": tile.label,
                "books": raw_books,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "truncated": truncated,
            })

            for item in raw_books:
                all_books.append(ExtractedBook(
                    spine_text=item["spine_text"],
                    title=item["title"],
                    author=item["author"],
                    legibility=item["legibility"],
                    orientation=item["orientation"],
                    source=tile.label,
                    seen_in=[tile.label],
                ))
            total_in += tokens_in
            total_out += tokens_out
            if truncated:
                truncated_tiles.append(tile.label)
            if progress:
                label = "  {}: {} books".format(tile.label, len(raw_books))
                if truncated:
                    label += "  [REPETITION LOOP DETECTED - may be missing books past this point]"
                progress(label)

    prior_failures = [(label, entry["error"])
                      for label, entry in already.items() if entry.get("error")]
    failures = prior_failures + failures

    return ExtractionResult(
        books=all_books,
        input_tokens=total_in,
        output_tokens=total_out,
        tiles_processed=len(tiles) - len(failures),
        tiles_failed=failures,
        tiles_truncated=truncated_tiles,
        provider=provider.label(),
        priced=provider.costs_money,
    )
