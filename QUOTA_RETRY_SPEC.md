# Google Books quota: retry spec + alternate source

Context for a fresh session: written after reading the real pipeline code
(not just designed in the abstract) — file/line references below are
accurate as of 2026-08-26.

**Parts 1 and 2 implemented and verified 2026-08-25/26 — see "Not yet
done" at the bottom for exactly what's still open (3 and 4).**

## The problem

Google Books' free key is ~1000 queries/day, per-key, not controllable in
code. Cataloguing one real 291-book shelf exhausted it twice in one day.
`pipeline/cache.py`'s shared Supabase cache helps a lot in steady state (a
re-scan of the same 291-book collection made only 115 external calls, 60%
served from cache) but does nothing for a brand-new community's first scan,
or the first time a genuinely uncommon title shows up.

`pipeline/enrich.py::_enrich_one` (line 521) already tries Open Library before
Google Books, so there's no per-book call waste to trim — Google Books is
already reserved for OL misses only.

## Important finding: the "obvious" fix target is the wrong file

Per `MVP_WORKFLOW.md` line 15 ("Operating default changed, 2026-08-25"),
**`worker.py`'s automated Ollama pass is no longer the default path.** Every
real batch now goes through `manual_batch.py publish` (line 132) →
`_parse_and_enrich()` → `enrich.enrich()` → cache → Open Library → Google
Books. Any fix has to land there, not in `worker.py`.

Two concrete gaps found in that live path:

1. **The quota diagnostic is computed and then discarded.**
   `_parse_and_enrich` (manual_batch.py:85) captures `lookup_errors` from
   `enrich.enrich()` but only `print()`s it (line 92) — the `stats` dict
   written to `scan_batches` (lines 94–106) has no field for it. Same gap in
   `worker.py::process_batch` (line 308 — return value of `enrich.enrich()`
   discarded entirely). Today, once a batch finishes, there is no way to
   later tell "quota-hit" apart from "genuinely obscure books" just by
   querying Supabase — the evidence only ever existed in Elliot's terminal.
2. **`cmd_publish` never calls `decide_route`.** It hardcodes
   `status="imported"` (manual_batch.py:169). The escalation logic that
   exists in `worker.py` (low match rate → `needs_admin`) isn't wired into
   the path that's actually used.

## Spec

### 1. Persist the signal (no migration — `scan_batches.stats` is jsonb)

In `_parse_and_enrich` (manual_batch.py:94), add to the `stats` dict:

```python
"lookup_errors": lookup_errors,
"quota_exhausted": any("429" in m and "googlebooks" in m for m in lookup_errors),
```

Same two lines in `worker.py::process_batch`'s stats dict (line 312),
capturing the currently-discarded return value of `enrich.enrich(...)` at
line 308.

### 2. Finding retry candidates — no new column needed

A quota-missed book is already a `books` row with `is_placeholder=true`
(written by `insert_placeholder_book`, manual_batch.py:161), and its
`library_items.scan_batch_id` points at the batch. One join finds them:

```sql
select li.id as library_item_id, b.id as book_id, b.title, b.authors
from library_items li
join books b on b.id = li.book_id
join scan_batches sb on sb.id = li.scan_batch_id
where b.is_placeholder = true
  and (sb.stats->>'quota_exhausted')::boolean = true
  and b.created_at > now() - interval '3 days';
```

The 3-day cutoff is the "give up" mechanism — cheaper than a retry-count
column, generous relative to Google's 24h quota reset.

### 3. `retry_quota_misses.py` (new script, sibling to `retry_unmatched.py`)

- Pull the query above.
- Rebuild a `CatalogueRecord` per row from the placeholder's `title`/`authors`
  (the same data `insert_placeholder_book` wrote — nothing missing).
- Re-run `enrich._try_openlibrary` / `_try_google_books` via a `BookCache`.
- On a fresh confident match:
  - If `is_cacheable(record)` passes: `book_cache.store(record)` (creates or
    reuses the shared row), then `PATCH library_items.book_id` to point at
    it — the same repoint pattern the existing "Fix title/author" correction
    flow uses (`src/app/library/actions.ts`'s `correctBook` /
    `bulkCorrectBooks`), just automatic instead of owner-initiated.
  - If it matches but doesn't clear the shared-cache bar: `UPDATE` the
    existing placeholder row's `title`/`authors`/`isbn`/`cover_url` in place.
    No repoint needed — it stays private either way.

### 4. Scheduling

Enrichment only calls Open Library / Google Books / Supabase — none of it
needs the local Ollama box, so it doesn't have to live on "Elliot's PC" the
way vision extraction does. A nightly scheduled task (after Google's
midnight-Pacific quota reset) running `retry_quota_misses.py` directly
against Supabase is simpler than a Windows Task Scheduler entry tied to a
machine that has to be powered on.

## A fourth catalogue source

Skip Amazon's Product Advertising API (requires an active Associates account
with qualifying sales just to keep access — too much compliance overhead for
this size project) and WorldCat (member-library gated).

**Recommended add: Wikidata's SPARQL endpoint.** No API key, no daily quota
(politely rate-limited, not hard-capped), structured title/author/ISBN/cover
(via Commons) data. Coverage of random personal-shelf genre fiction is
patchier than Google Books, so it slots in as a fourth rung — cache → Open
Library → Google Books → Wikidata — tried only when the first three miss.
Same "weaker coverage, zero quota" relationship Open Library already has to
Google Books.

If that proves too thin once real usage data exists, **ISBNdb**'s cheap paid
tier is the next escalation — consistent with the "don't spend API money
before demand is proven" principle already stated in `MVP_WORKFLOW.md` line
355.

## Status

**Done, 2026-08-25/26:**

1. **Stats-dict edits** — `manual_batch.py::_parse_and_enrich` and
   `worker.py::process_batch` both now write `lookup_errors` and
   `quota_exhausted` (the `"429" in m and "googlebooks" in m` check, exactly
   as specced) into the `stats` dict that lands in `scan_batches.stats`.
   `worker.py` previously discarded `enrich.enrich()`'s return value
   entirely; now captured.
2. **`retry_quota_misses.py`** — built essentially as specced, with one
   simplification: it calls the existing public `enrich.enrich()` (which
   already does cache → Open Library → Google Books, in order) rather than
   reaching into the private `_try_openlibrary`/`_try_google_books`
   functions directly — same result, less surface area. Both outcomes from
   the spec are implemented: a cacheable fresh match repoints the
   `library_item` at a new shared cache row (via `BookCache.store()`,
   which already no-ops for a non-cacheable match); a non-cacheable match
   updates the existing placeholder row's title/authors/isbn/cover_url in
   place.

   **Verified against real production, not just read through:** built a
   throwaway test fixture directly in Supabase (a `scan_batches` row
   flagged `quota_exhausted`, one placeholder book with a findable real
   title — "The Hobbit" — and one with an unfindable nonsense title,
   both linked via `library_items`), ran the actual script against it
   twice. First run: "The Hobbit" correctly repointed to a brand-new real
   shared-cache row (`match_source: openlibrary, match_score: 1.000,
   is_placeholder: false`), the nonsense one correctly left untouched.
   Second run: confirmed idempotent — the recovered book no longer shows
   up as a candidate (it's not a placeholder anymore), the nonsense one
   correctly still fails to match, no errors, no duplicate rows created.
   All test data deleted afterward; verified `books`/`library_items`/
   `scan_batches` counts back to their pre-test baseline (60/33/2).

**Also done, 2026-08-26:**

3. **The nightly schedule.** Elliot chose Windows Task Scheduler on his own
   PC. Task `"KarovShelf Quota Retry"`, daily trigger at **11:00 AM Israel
   time** (~1 hour after Google's midnight-Pacific quota reset, which
   currently lands around 10:00 AM Israel time while both regions are in
   DST — the hour of buffer absorbs the small drift on the few days a year
   the two DST calendars don't line up exactly). Runs
   `python retry_quota_misses.py` via `cmd.exe` (so stdout/stderr can be
   redirected), working directory `E:\Code\karovshelf-pipeline`, output appended to
   `out/quota_retry.log`. Registered with `-StartWhenAvailable` so a
   powered-off/asleep PC catches up next time it's on, and under the
   current user with no stored password (`Run only when user is logged
   on`, same trust model as the Ollama Startup-folder shortcut already
   uses — no credential prompt needed). **Verified for real**: manually
   fired the task once (`Start-ScheduledTask`) rather than waiting for
   11am — `LastTaskResult: 0` (success), and the log file shows the
   correct real output.

**Evaluated and NOT recommended, 2026-08-26 — see "Wikidata evaluation"
below for the real test results.** Part 4 is closed as "don't build this,"
not "not yet done."

## Wikidata evaluation (2026-08-26) — real test, not just theory

The recommendation in "A fourth catalogue source" above was written in
the abstract, without actually querying Wikidata. Tested it for real
before building anything, against two sets of real titles.

**Set 1 - the 6 genuinely obscure/self-published books currently sitting
as placeholders in the live library** (exactly the case a 4th source
needs to help with, since anything mainstream is already found by Open
Library/Google Books): **0/6 found.** Several had zero search hits at
all (`wbsearchentities` for "The Battle for Hjalgar", "The Occidental
Empire", "The Sleeping Mother", "Ulpan Ivrit", "Spy the Lie" all came back
empty).

**Set 2 - famous control titles, to sanity-check the query mechanism
itself wasn't just buggy** (Dune, Sapiens, Kafka on the Shore, Robert's
Rules of Order, The Thursday Murder Club, "THE DAWNING"): mixed on
whether a book-like entity was found at all (Wikidata often ranks a film
adaptation or an unrelated same-titled concept above the actual novel in
search results - Dune's top hit was the 2021 film, not Frank Herbert's
book; "Sapiens" and "Geometry" similarly matched non-book concepts
first), but the finding that actually matters: **every single successful
book-like match had no ISBN (`P212`/`P957` both empty), and no cover
image in 3 of 4 cases** - this held across every genre/era tested, not
just the obscure ones.

**Verdict: don't build this.** Two independent, compounding problems, not
one fixable bug:
1. Wikidata's book coverage skews toward "notable" (Wikipedia-notability-
   bar) works, close to the opposite of what a 4th tier needs - mainstream
   books already match fine via Open Library/Google Books, so the only
   case a 4th tier matters for is exactly the obscure/self-published
   titles Wikidata doesn't have either.
2. Even where Wikidata *does* have an entry, it's frequently missing ISBN
   and cover image - the two fields the pipeline most wants a catalogue
   match for in the first place. A "match" with neither field populated
   isn't much better than the private placeholder it would be replacing.

Disambiguation could likely be improved with a proper SPARQL query
(filtering by `wdt:P31/wdt:P279*` book-like classes plus an author-name
match, instead of the simple `wbsearchentities` + top-5-candidates
approach used for this test) - but that would only address problem 2's
symptom (finding the right entity), not problem 1 (the entity often
doesn't exist) or the ISBN/cover gap. Not worth the implementation cost
for what it would actually fix.

**What would actually help the underlying problem instead**, if it
resurfaces: a second Google Books API key under a separate Google Cloud
project (mechanical, no code change, just another `~1000/day` ceiling to
rotate to); or simply accepting that a genuinely obscure/self-published
title stays a private placeholder forever, which the app already handles
gracefully (fully usable in its owner's own library, just not
contributed to the shared cache) - not every book needs to clear that
bar.
