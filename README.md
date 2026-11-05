# KarovShelf shelf-scan pipeline

Photos of a bookshelf in, a catalogue out. This Python project supports the
live KarovShelf neighbourhood book-sharing app; it is now the operational
cataloguing tool and fallback benchmark, not a prerequisite for starting the
web product.

The companion web app is live at
[karovshelf.vercel.app](https://karovshelf.vercel.app).
It accepts private shelf-photo uploads, queues batches, lets owners review and
share books, supports neighbourhood search, and handles borrow requests.
For the current operating model and verified system status, start with
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) and [`MVP_WORKFLOW.md`](MVP_WORKFLOW.md).

## Run it

```bash
python -m pip install -r requirements.txt
```

There are two vision backends. They read the same tiles with the same prompt and
the same output schema, so swapping between them measures the models rather than
two different harnesses.

### Free: a local model on your own GPU

Install [Ollama](https://ollama.com), then pull a vision model once:

```bash
ollama pull qwen2.5vl:7b
```

```bash
python -m pipeline.run samples/ --provider ollama --out out/local
```

No API key, no per-scan cost, no data leaving the machine. Needs a GPU with
enough VRAM (this model wants ~8GB) and is much slower per tile than the hosted
option. Accuracy is the open question - run both and compare.

### Paid: Claude

Needs pay-as-you-go API credits. **A Claude Pro subscription does not cover
this** - the API bills separately.

```bash
python -m pipeline.run samples/ --provider claude --out out/claude
```

### Either way

Drop shelf photos in `samples/` first. Open the resulting `.html` to see every
book found next to the raw spine transcription, so misreads are obvious.

Use `--limit 2` for a quick sanity check before committing to a full run, and
`--language any` if the shelves include non-English books.

Verify the non-AI half without any model at all:

```bash
python selftest.py
```

## How it works

```
photos ──► tile ──► Claude Opus 5 vision ──► dedupe ──► catalogue match ──► report
```

1. **Tile** (`imageprep.py`). The most important step. A phone photo of a full
   bookcase is ~4000px wide, but vision input works at ~1568px on the long edge,
   so sending the whole frame throws away exactly the resolution spine text
   lives in. Instead the image is cut into an overlapping **grid** sized from
   the source resolution (a 4032x3024 photo becomes 3x2), each tile sent at
   close to native resolution. Overlap is 18% so a book on a seam appears whole
   in at least one tile. Sizing the grid by aspect ratio instead of resolution
   is a trap: it calls a 4:3 photo "square" and sends it as one squashed tile.

2. **Read** (`extract.py`). Each tile goes to `claude-opus-5` with a structured
   output schema. The model returns, per book, both a literal `spine_text`
   transcription *and* its interpreted `title`/`author`, plus a self-reported
   `legibility`. Keeping transcription separate from interpretation is what
   makes hallucination visible: if `spine_text` reads `TH3 REM4INS` and the
   title comes back as a real book, that is a good inference; if `spine_text` is
   `?????` and the title is confident, that is a fabrication.

3. **Dedupe** (`enrich.py`). Overlapping tiles see the same book twice. Readings
   merge on fuzzy title match, keeping the most legible version.

4. **Match** (`enrich.py`). Each reading is looked up in Open Library, then
   Google Books. This is what turns a fuzzy guess into a real record with a
   correct title, author, ISBN, cover, and subject tags. A reading that matches
   *nothing* is a signal, not a failure — it goes to the review queue.

5. **Report** (`report.py`). Writes `scan.json` (the catalogue) and `scan.html`
   (the accuracy audit).

## Options

| Flag | Default | Notes |
|---|---|---|
| `--provider` | `ollama` | `ollama` (free, local) or `claude` (paid, hosted) |
| `--model` | per provider | Override the model name |
| `--limit N` | off | Read only the first N tiles |
| `--tiles N` | from source resolution | Caps the grid. Fewer tiles = cheaper, less legible |
| `--effort` | `medium` | `low`/`medium`/`high`/`xhigh`/`max` — worth A/B-ing |
| `--language` | `eng` | `any` to disable |
| `--workers` | 4 | Parallel API calls |
| `--no-enrich` | off | Raw model output only, no catalogue lookup |

Video input works if you `pip install opencv-python`, but **photos read
noticeably better** — phone video is compressed and motion-blurred. Worth
knowing before the UI promises a video-first flow.

## What a scan actually costs

Output tokens dominate, not the image. A 4032x3024 phone photo becomes a 3x2
grid of 6 tiles:

| | tokens | cost |
|---|---|---|
| Input (6 tiles of image) | ~18,000 | ~$0.09 |
| Output (JSON + adaptive thinking) | ~15,000-24,000 | ~$0.38-$0.60 |
| **Per photo** | | **~$0.45-$0.70** |

Input is cheap and exactly predictable; output is 5x the price per token and
includes thinking, which is the part that varies. Budget roughly **$3-5 to
catalogue a 300-book collection** at default settings.

Three levers, in order of how much they help:

- `--effort low` — less thinking, the single biggest reduction.
- **The Batch API halves it.** Scanning is inherently asynchronous: someone
  uploads shelf photos and can happily wait a few minutes. This is close to a
  free 50% discount and the pipeline should use it before launch.
- **Try Haiku 4.5** (5x cheaper per token). Spine reading is closer to OCR than
  to reasoning, so it may hold up fine — worth a head-to-head on the same photo
  before assuming Opus is needed.

## Known limitations (measured, not guessed)

- **Google Books rate-limits anonymous traffic.** Without a key you will see
  `429 quota exceeded` and the match rate will look artificially bad. A free key
  raises it to ~1000/day and materially improves coverage of translated works:
  enable the Books API in Google Cloud, then set `GOOGLE_BOOKS_API_KEY`. The CLI
  now reports lookup errors separately from model errors, so an outage can't
  masquerade as a misread.

- **Open Library files translated works under the original language.** "Kafka on
  the Shore" exists there only as 海辺のカフカ, so an English spine finds nothing.
  Google Books handles this correctly, which is the main reason it is worth a key.

- **Genre is derived from catalogue subject tags**, which are noisy. Subjects are
  folded onto a fixed vocabulary in tiers (specific genre → fiction → topical) so
  a historical novel doesn't get filed under History. It works, but if genre
  quality matters, the better fix is one extra batched model call to classify
  everything at the end.

- **`spine_text` is self-reported.** It makes hallucination visible; it does not
  prevent it. Real accuracy numbers need a real shelf with a known ground truth.

- **Dedupe used to let a cluster's identity drift.** Merging a book into an
  existing cluster upgraded that cluster's comparison title to the more
  legible reading, so its match target could wander across tiles it was never
  actually similar to at the start. On a real 759-reading scan this collapsed
  9 genuinely different titles into 3, and in the worst real case, 218
  unrelated readings into one book. Fixed by freezing each cluster's matching
  anchor (`match_title`) at creation - see `check_dedupe_chaining()` in
  `selftest.py`.

- **A small local model can get stuck in a generation loop mid-tile**, repeating
  one title dozens of times until it exhausts its output budget - silently
  losing whatever the shelf actually showed after that point. `repeat_penalty`
  looked like an obvious fix and made things *worse*: the loop still happened,
  but the model got penalised for reusing title tokens it had already spent on
  `spine_text`, so every entry came back with a correctly-shaped but EMPTY
  title - which the pipeline's own filter then silently dropped, turning a
  5-book tile into a **silent** 0 with no warning at all.

  The real fix: `repeat_penalty` was removed; detection now runs on
  `spine_text` (which stays populated in both failure shapes) *before* the
  empty-title filter, not on `title` after it. A detected loop is collapsed to
  one copy, the correctly-read books before the loop are kept, and the tile is
  flagged - both in the CLI output and as a "tiles suspect" count in the HTML
  report - so a rescan can be targeted rather than the collection silently
  missing books. See `check_repetition_guard()` in `selftest.py`.

- **Dedupe only covers duplicates within one scan, plus rescans of an
  already-*unmatched* book - not a placeholder that later gets a real
  catalogue match.** Found 2026-09-03: a book that missed the catalogue on
  one scan (became a private placeholder) and was later rescanned and
  correctly matched used to create a *second* `library_items` row instead
  of replacing the first, because only the placeholder-to-placeholder
  rescan path checked for an existing owner copy - the placeholder-to-
  matched path had no check at all. Produced 5 real duplicates in
  production (e.g. "The Koren Tehillim" appearing twice - once as an
  unmatched placeholder, once correctly matched) before anyone noticed.
  Fixed with `SupabaseWorkerClient.supersede_owner_placeholder()`
  (`pipeline/worker.py`), wired into every write path that can produce a
  matched `book_id`. This is a live-HTTP-calling method with no unit test
  (same convention as `insert_placeholder_book`/`find_owner_placeholder_book`)
  and hasn't yet been exercised against a real production rescan - **worth
  watching the next time an owner's already-placeholder'd book gets
  rescanned and matches**, to confirm it actually supersedes instead of
  duplicating. See `PROJECT_SUMMARY.md` §10 for the full incident.

## The shared book cache

The binding constraint on this product is **not** model accuracy - it is
catalogue quota. Cataloguing one 291-book collection exhausted the Google Books
free daily allowance twice. The second user of the day would get a degraded scan
through no fault of their own.

The fix is that book collections overlap enormously: everyone in one community
owns some of the same paperbacks. So a title is looked up externally **once,
ever**, and served from our own `books` table thereafter.

Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env` and it turns on
automatically; without them the pipeline behaves exactly as before. `--no-cache`
disables it for a run.

Lookup is two-stage: an exact hit on a normalised `title|author` key, then a
trigram fuzzy match via the `match_cached_book` RPC for readings that differ
slightly from what was stored ("Children 0f Dune" still finds "Children of Dune").

### What is deliberately NOT cached

A cache miss costs one API call. A cached *wrong* answer is permanent and is
served to every other user in the community. So the write bar is much higher
than the display bar - `is_cacheable()` rejects:

- anything already flagged for review, or read with `legibility: "guess"`
- **fragments** - a spine read only as far as "Dungeons of" was binding to
  "The Dungeons of Nakasero" at 0.95 confidence
- **short generic titles with no author** - "Switch" matched a Sandra Brown
  thriller instead of the Heath brothers; "Culture" matched "The Culture Code"

That drops roughly 18% of otherwise-matched books from the cache. They still
appear in their owner's library; they just get looked up per-user rather than
being shared as fact.

## Scope boundary

This repository does not provide the browser experience; that is the separate
Next.js app in `E:\Code\karovshelf`. The pipeline does write catalogue results
to the same Supabase system used by the web app, including `scan_batches`,
`books`, and `library_items`.

For real user batches, use the **manual-via-Claude** flow in
`manual_batch.py` by default. The Ollama worker remains useful for fallback
processing and benchmarking, but it is not the default production workflow.
