"""Exercise everything except the paid vision call.

Feeds deliberately noisy spine readings - the kind of OCR damage a real photo
produces - through dedupe, catalogue matching, and reporting. The question it
answers: does the catalogue lookup actually rescue a misread, or does garbage
in stay garbage out?
"""

from PIL import Image

from pipeline import enrich, genre_classify, imageprep, report
from pipeline.extract import ExtractedBook

# Each tuple is (spine_text, title-as-read, author-as-read, legibility).
# The damage is modelled on what actually goes wrong: dropped and swapped
# characters, missing authors, subtitle noise, and one pure hallucination.
NOISY_READINGS = [
    ("KAFKA ON THE SHORE  MURAKAMI  VINTAGE", "Kafka on the Shore", "Murakami", "clear"),
    ("KAFKA 0N THE SH0RE MURAKAM1", "Kafka on the Shore", "Murakami", "partial"),
    ("TH3 REM4INS OF THE DAY  ISHIGURO", "The Remains of the Day", "Ishiguro", "partial"),
    # Expected to come back UNMATCHED since AUTHOR_CORROBORATION_FLOOR was
    # added (2026-09-03) - not a regression, a disclosed tradeoff. A bare
    # surname against a full name ("Harari" vs "Yuval Noah Harari") scores
    # 0.522 by similarity() - almost identical to a real, confirmed
    # COINCIDENTAL surname collision found the same day ("Brown" vs
    # "William P. Brown", an unrelated author on an unrelated book, scored
    # 0.5). No threshold value can accept one and reject the other; nothing
    # in the title or author text distinguishes "same person, abbreviated"
    # from "different person, same common surname" without outside
    # knowledge of which surnames are actually rare. Given the project's own
    # standing rule (a wrong match is worse than an unmatched one), this
    # sides with rejecting both - see PROJECT_SUMMARY.md Sec10, 2026-09-03,
    # for the full false-match investigation this came out of.
    ("SAP1ENS  A BRIEF H1STORY 0F HUMANKIND  HARARI",
     "Sapiens: A Brief History of Humankind", "Harari", "partial"),
    ("WOLF HALL   MANTEL", "Wolf Hall", "Hilary Mantel", "clear"),
    ("THE GOLDF?NCH  D?NNA TARTT", "The Goldfinch", "Donna Tartt", "partial"),
    ("N0RMAL PE0PLE", "Normal People", None, "partial"),
    ("?????  ??????", "Selected Poems", None, "guess"),
    ("EDUCATED  WESTOVER", "Educated", "Westover", "clear"),
    ("A GENTLEMAN IN M0SC0W  T0WLES", "A Gentleman in Moscow", "Towles", "partial"),
    ("PACHINK0  MIN JIN LEE", "Pachinko", "Min Jin Lee", "clear"),
    ("THE THURSDAY MURDER CLUB  OSMAN", "The Thursday Murder Club", "Richard Osman", "clear"),
    ("QRZTL FLMBX ZZQ", "Qrztl Flmbx", None, "guess"),
]


def check_tiling():
    print("== tiling ==")
    # A typical wide bookcase photo.
    shelf = Image.new("RGB", (4032, 1200), "white")
    tiles = imageprep.tile_image(shelf, source="fake-shelf.jpg")
    print("  4032x1200 shelf -> {} tiles".format(len(tiles)))
    for tile in tiles:
        print("     {} at {} (~{:,} tokens)".format(
            tile.label, tile.image.size, tile.estimated_tokens()))

    # Verify the overlap actually covers the full width.
    assert len(tiles) >= 3, "a wide shelf should produce several tiles"
    assert all(max(t.image.size) <= 1568 for t in tiles), "tile exceeds max edge"

    # A single tall shelf should stay whole.
    tall = Image.new("RGB", (900, 1400), "white")
    one = imageprep.tile_image(tall, source="tall.jpg")
    assert len(one) == 1, "a portrait shelf should not be sliced"
    print("  900x1400 shelf -> 1 tile (correct, no slicing)")
    print()


def check_dedupe_chaining():
    """Guard against clusters drifting across the title space.

    Greedy fuzzy merging is single-linkage: A merges with B and B with C, so A
    and C can end up together even though they are unrelated. It gets far worse
    if the cluster adopts each new title, because its comparison anchor then
    wanders. On a real 759-reading run that produced two clusters holding 218
    and 202 unrelated books - silently destroying most of the catalogue.
    """
    print("== dedupe chain resistance ==")
    chain = ["The Art of War", "The Art of Wart", "The Art of Warts",
             "The Part of Warts", "The Part of Hearts", "The Park of Hearts",
             "The Dark of Hearts", "The Dark of Hearth", "The Dark of Earth"]
    books = [
        ExtractedBook(spine_text="", title=title, author=None,
                      legibility="clear" if i else "guess",
                      orientation="vertical", source="t", seen_in=["t"])
        for i, title in enumerate(chain)
    ]
    merged = enrich.dedupe(books)
    worst = max(m.times_seen for m in merged)
    print("  {} chained titles -> {} clusters, largest holds {}".format(
        len(chain), len(merged), worst))
    assert worst <= 3, "a cluster ran away and absorbed {} books".format(worst)
    assert len(merged) >= 4, "chain collapsed too far"
    print("  no runaway cluster")
    print()


def check_volume_number_guard():
    """Different volumes of the same series must never merge, even though
    their titles are otherwise near-identical - found 2026-09-03 on real
    scans: "Kitzur Shulchan Aruch I" vs "...II" (similarity 0.979) and four
    Sandman volumes ("Vol. 7" through "Vol. 10", pairwise 0.889-0.923) both
    cleared DEDUPE_THRESHOLD (0.88) and silently collapsed to one survivor
    each, losing real books. `_trailing_volume_marker` in pipeline/enrich.py
    blocks a merge whenever both titles end in a bare volume/part number
    (arabic or roman) and those numbers disagree.
    """
    print("== volume-number dedupe guard ==")

    def entry(title, author="Author"):
        return ExtractedBook(spine_text="", title=title, author=author,
                             legibility="clear", orientation="vertical",
                             source="t", seen_in=["t"])

    kitzur = enrich.dedupe([entry("Kitzur Shulchan Aruch I"), entry("Kitzur Shulchan Aruch II")])
    assert len(kitzur) == 2, "Vol. I and Vol. II must stay separate, not merge into one"
    print("  Kitzur Shulchan Aruch I / II -> 2 distinct records")

    sandman = enrich.dedupe([entry("The Sandman, Vol. 10", "Neil Gaiman"),
                             entry("The Sandman, Vol. 9", "Neil Gaiman"),
                             entry("The Sandman, Vol. 8", "Neil Gaiman"),
                             entry("The Sandman, Vol. 7", "Neil Gaiman")])
    assert len(sandman) == 4, "all four Sandman volumes must survive as distinct records"
    print("  Sandman Vol. 7/8/9/10 -> 4 distinct records")

    # Must NOT block a genuine repeat read of the exact same volume.
    same_vol = enrich.dedupe([entry("The Sandman, Vol. 10", "Neil Gaiman"),
                              entry("The Sandman, Vol. 10", "Neil Gaiman")])
    assert len(same_vol) == 1, "two readings of the SAME volume must still merge"
    print("  two readings of the same volume -> still merges to 1")

    # The classic Biblical/Judaica split-book convention puts the numeral
    # FIRST, not last ("I Kings" vs "II Kings") - almost merged for real on
    # 2026-09-03 (similarity 0.933) before this leading-token check was added.
    kings = enrich.dedupe([entry("I Kings"), entry("II Kings")])
    assert len(kings) == 2, "I Kings and II Kings must stay separate, not merge into one"
    print("  I Kings / II Kings -> 2 distinct records")
    print()


def check_series_detection():
    """detect_series() powers the web app's "condense N volumes into one
    card" display grouping (My Library / Search) - it must find the same
    base title/volume number the volume-number dedupe guard above already
    protects, plus the "Vol. N: Subtitle" shape that guard doesn't need to
    handle (a subtitle after the number never blocks a dedupe merge, since
    dedupe compares whole titles, but grouping needs the bare number).
    """
    print("== series detection (for display grouping) ==")

    assert enrich.detect_series("The Sandman, Vol. 7") == ("The Sandman", 7)
    assert enrich.detect_series("The Sandman Vol. 10: The Wake") == ("The Sandman", 10)
    print("  'The Sandman, Vol. 7' / 'Vol. 10: The Wake' -> base + volume number")

    assert enrich.detect_series("Kitzur Shulchan Aruch, Vol. II") == ("Kitzur Shulchan Aruch", 2)
    print("  'Vol.' + bare roman numeral -> detected")

    assert enrich.detect_series("The Goldfinch") is None
    assert enrich.detect_series("A Gentleman in Moscow") is None
    print("  ordinary titles with no volume marker -> None")

    # Real false positives found the first time this ran against the live
    # library, 2026-09-03 - all from the bare leading/trailing-numeral
    # fallback this function deliberately does NOT use (unlike the dedupe
    # guard above, which is safe to use it because it only ever blocks a
    # merge, never actively groups). Requiring an explicit "Vol."/"Volume"/
    # "Part" keyword avoids every one of these.
    assert enrich.detect_series("I Believe") is None
    assert enrich.detect_series("I asked for wonder") is None
    assert enrich.detect_series("Fahrenheit 451") is None
    assert enrich.detect_series("60 Days Away") is None
    assert enrich.detect_series("I Kings") is None
    assert enrich.detect_series("The ABCs of the UCC, Articles 3 and 4") is None
    print("  real false-positive cases from the 2026-09-03 backfill -> all None")
    print()


def check_series_prefix_clustering():
    """cluster_series_by_prefix() catches a real multi-book set with no
    "Vol. N" marker at all - just a shared series-name prefix before each
    volume's own subtitle. Found for real 2026-09-03: 14 "A Daily Dose of
    Torah: ..." books in production, only 2 of which detect_series() alone
    could tag (the two that happened to also say "Vol. N").
    """
    print("== series grouping by shared title prefix ==")

    def book(id_, title, series_title=None, owner="owner-1"):
        return {"id": id_, "title": title, "series_title": series_title, "owner_id": owner}

    daily_dose = [
        book("b1", "A Daily Dose of Torah, Vol. 11: Weeks of Matos-Vaeschanan", "A Daily Dose of Torah"),
        book("b2", "A daily dose of Torah: Daily study for the weeks of Acharei-Bechukosai. 2007"),
        book("b3", "A Daily Dose of Torah: Shemos-Beshalach"),
    ]
    updates = dict(enrich.cluster_series_by_prefix(daily_dose, min_group_size=3))
    assert "b1" not in updates, "already-correct canonical title shouldn't get rewritten"
    assert updates["b2"] == "A Daily Dose of Torah"
    assert updates["b3"] == "A Daily Dose of Torah"
    print("  14-volume-style set, only 1 pre-tagged -> the other 2 inherit its title")

    # Too few members - must NOT group (same MIN_GROUP_SIZE convention as
    # the display grouping this feeds).
    pair = [book("c1", "Covenant & Conversation: Genesis"),
            book("c2", "Covenant & Conversation: Exodus")]
    assert enrich.cluster_series_by_prefix(pair, min_group_size=3) == []
    print("  only 2 shared-prefix books -> left alone")

    # Different owners with the same prefix are NOT one set.
    cross_owner = [
        book("d1", "The Hirsch Chumash: Bereishis", owner="alice"),
        book("d2", "The Hirsch Chumash: Vayikra", owner="alice"),
        book("d3", "The Hirsch Chumash: Bemidbar", owner="bob"),
    ]
    assert enrich.cluster_series_by_prefix(cross_owner, min_group_size=3) == []
    print("  same prefix, different owners -> not grouped together")

    # A short shared prefix before the comma/colon must never qualify, even
    # with 3+ members - "I" alone is not a meaningful series signal.
    assert enrich._title_prefix("I: Book One") is None
    assert enrich._title_prefix("A Daily Dose of Torah: Weeks of X") == "A Daily Dose of Torah"
    short_prefix = [book("e1", "I: Book One"), book("e2", "I: Book Two"), book("e3", "I: Book Three")]
    assert enrich.cluster_series_by_prefix(short_prefix, min_group_size=3) == []
    print("  short shared prefix ('I') -> not grouped even with 3+ members")
    print()


def check_series_from_catalogue_subjects():
    """catalogue_series_title (_series_from_subjects()) covers the one real
    gap neither detect_series() nor cluster_series_by_prefix() can: a series
    like The Expanse ("Leviathan Wakes", "Caliban's War", "Abaddon's Gate", ...)
    whose volumes share no common title wording at all - Open Library tags
    some editions' subjects with "series:<name>", the only real signal
    available for this case. Added 2026-09-09."""
    print("== series name from catalogue subjects (differently-titled series) ==")

    assert enrich._series_from_subjects(
        ["Space warfare", "Fiction", "series:The Expanse"]
    ) == "The Expanse"
    print("  'series:The Expanse' subject tag -> 'The Expanse'")

    assert enrich._series_from_subjects(["Fiction", "American literature"]) is None
    assert enrich._series_from_subjects([]) is None
    assert enrich._series_from_subjects(None) is None
    print("  no series-tagged subject -> None (most books, most editions)")

    # _clean_subjects() (the genre-classifier's subjects list) must not keep
    # the raw "series:X" tag around - it reads like noise to a genre
    # classifier, and _series_from_subjects() is the one place that cares
    # about it, working from the raw list before cleaning.
    cleaned = enrich._clean_subjects(["Fiction", "series:The Expanse", "Space warfare"])
    assert "series:The Expanse" not in cleaned
    assert cleaned == ["Fiction", "Space warfare"]
    print("  'series:X' tag stripped from the cleaned genre-subjects list")
    print()


def check_categorize_publisher_signal():
    """categorize()'s publisher parameter - added 2026-09-11 after validating
    the Judaica-routing fix against a real shelf found it under-firing.
    "Jeremiah" (Binyamin Lau's Maggid Tanakh commentary) matched a real
    catalogue record whose spine reading only captured the author's surname
    ("LAU"), never "MAGGID" - but Open Library's own `publisher` field for
    that exact match says "Maggid Books". Catalogue publisher is a far more
    reliable Judaica-publisher signal than hoping OCR caught a small spine
    imprint."""
    print("== categorize() catalogue-publisher signal (Judaica) ==")

    assert enrich.categorize("Jeremiah", ["Binyamin Lau"], None, spine_text="LAU",
                             publisher="Maggid Books") == "Judaica", (
        "a Judaica publisher from the catalogue must be caught even when "
        "neither the title nor the OCR'd spine text carries any Judaica signal"
    )
    print("  catalogue publisher 'Maggid Books' -> Judaica, even with no other signal")

    assert enrich.categorize("Jeremiah", ["Binyamin Lau"], None, spine_text="LAU") is None, (
        "without the publisher, this exact case must stay unresolved - "
        "confirms the fix is additive, not a change to the existing rules"
    )
    print("  same book, no publisher available -> still None (unchanged behaviour)")

    assert enrich.categorize("SPQR", ["Mary Beard"], None,
                             publisher="Penguin Random House") is None
    print("  an ordinary publisher -> no false Judaica match")
    print()


def check_catalogue_match_disambiguation():
    """Guard against binding a short/generic title to a completely unrelated
    real book, when there's no author on the scan to catch it.

    Found on real scans 2026-09-03, all with no (or only a coincidentally-
    similar) author on the spine to cross-check: "Isaiah" (a Maggid Tanakh
    commentary) matched a biography titled "Isaiah Berlin"; "Jonah" matched
    a horror novel called "The Jonah" (an exact match once `normalise()`
    strips the leading "The"); "Ecclesiastes" matched an unrelated
    commentary-series book whose author ("William P. Brown") happened to
    share a surname (0.5 similarity - a real coincidence, not agreement);
    "A Citizen's Guide to the Economy" matched an Armenian government
    budget guide. `_score_candidate` (pipeline/enrich.py) exists to reject
    all of these while still accepting genuine short-title matches (both
    with and without an author on the scan).
    """
    print("== catalogue match disambiguation ==")
    from pipeline import config
    from pipeline.enrich import CatalogueRecord, _score_candidate

    def rec(title, author=None):
        return CatalogueRecord(raw_title=title, raw_author=author, spine_text="",
                               legibility="clear", seen_in=["t"])

    false_matches = [
        ("Jonah (no author)", rec("Jonah"), "The Jonah", ["James Herbert"]),
        ("Isaiah (no author)", rec("Isaiah"), "Isaiah Berlin", ["Michael Ignatieff"]),
        ("Ecclesiastes/Brown vs unrelated William P. Brown book",
         rec("Ecclesiastes", "Brown"),
         "Ecclesiastes (Interpretation, a Bible Commentary for Teaching and Preaching)",
         ["William P. Brown"]),
        ("A Citizen's Guide to the Economy (no author)", rec("A Citizen's Guide to the Economy"),
         "A citizen's guide to the budget", ["Armenia Finance Ministry"]),
    ]
    for label, record, cand_title, cand_authors in false_matches:
        score = _score_candidate(record, cand_title, cand_authors)
        assert score is None, "{} must be rejected, got score {}".format(label, score)
        print("  {} -> correctly rejected".format(label))

    legit_matches = [
        ("Sapiens/Harari vs full-subtitle catalogue entry, same author",
         rec("Sapiens", "Yuval Noah Harari"),
         "Sapiens: A Brief History of Humankind", ["Yuval Noah Harari"]),
        ("Dune (no author) vs exact same-length title", rec("Dune"), "Dune", ["Frank Herbert"]),
    ]
    for label, record, cand_title, cand_authors in legit_matches:
        score = _score_candidate(record, cand_title, cand_authors)
        assert score is not None and score >= config.MATCH_THRESHOLD, \
            "{} should still match, got {}".format(label, score)
        print("  {} -> still matches ({:.3f})".format(label, score))
    print()


def check_repetition_guard():
    """Guard against a runaway generation loop silently eating a tile.

    Observed on real scans in two shapes. The first time: the model gets stuck
    repeating one well-formed title (74x-218x in separate cases) until its
    output budget runs out, after reading a handful of real books correctly.
    The second time, after adding repeat_penalty as what seemed like an obvious
    mitigation: the loop still happened, but the model got penalised for
    reusing title tokens it had already spent on spine_text, so it emitted a
    correctly-shaped but EMPTY title for every entry - which our own filter
    then silently dropped, turning a detectable failure into an invisible one
    (a tile with 5 real, legible books reported 0, with no warning at all).
    repeat_penalty was removed; detection now runs on spine_text, which stays
    populated in both failure shapes, and runs BEFORE the empty-title filter.

    Either shape must strip the loop down to one copy and flag the tile - and
    neither check may fire on a shelf that genuinely holds two physical copies
    of the same book sitting together.
    """
    print("== repetition-loop guard ==")
    import json
    from pipeline.providers import process_reply

    def entry(spine, title="", author=None):
        return {"spine_text": spine, "title": title, "author": author,
                "legibility": "clear", "orientation": "vertical"}

    # Shape 1: well-formed repeated titles (the original failure).
    well_formed = json.dumps({"books": [entry("A", "Real Book One", "X")]
                              + [entry("B", "Looping Title", "Y")] * 74})
    books1, truncated1 = process_reply(well_formed)
    assert truncated1, "a 74x well-formed repeat must be detected as a loop"
    assert len(books1) == 2, "the loop should collapse to a single copy"
    print("  74x well-formed repeat -> collapsed to {}, flagged".format(len(books1)))

    # Shape 2: the empty-title degenerate case that repeat_penalty produced.
    # If detection ran on `title` instead of `spine_text`, or ran after the
    # empty-title filter instead of before, this would silently report 0 books
    # with no warning - exactly the regression that shipped once already.
    degenerate = json.dumps({"books": [entry("LADIES AND GENTLEMEN OF THE JURY")]
                             + [entry("A GUIDE TO THE MODEL CODE")] * 30})
    books2, truncated2 = process_reply(degenerate)
    assert truncated2, ("an empty-title loop must still be flagged - this is "
                        "the exact regression a repeat_penalty mitigation caused")
    print("  30x empty-title repeat -> flagged (not silently reported as 0)")

    # Must not fire on a shelf that genuinely holds 2 copies of one title -
    # these have different spine_text (different physical crops of the spine).
    legit = json.dumps({"books": [
        entry("SIDE1", "Where the Sidewalk Ends", "Silverstein"),
        entry("ATTIC", "A Light in the Attic", "Silverstein"),
        entry("SIDE2", "Where the Sidewalk Ends", "Silverstein"),
    ]})
    books3, truncated3 = process_reply(legit)
    assert not truncated3, "two genuine copies must not be flagged as a loop"
    assert len(books3) == 3, "genuine duplicates must survive untouched"
    print("  2 genuine duplicate copies -> left alone, not flagged")

    # Judaica-shaped repeats get a relaxed ceiling (20, not 3) - found
    # 2026-08-26 on a real phone-uploaded batch with a dozen identical
    # Haggadahs bought in bulk for Seder guests, false-flagged as a loop.
    haggadahs = json.dumps({"books":
        [entry("Haggadah", "Haggadah", "Various")] * 12})
    books4, truncated4 = process_reply(haggadahs)
    assert not truncated4, "12 real bulk-bought Haggadahs must not be flagged as a loop"
    assert len(books4) == 12, "all 12 must survive uncollapsed - dedupe happens later, in enrich.dedupe"
    print("  12x genuine bulk Haggadahs -> left alone, not flagged")

    # But the safety net must still catch an actual runaway loop even when the
    # looping text happens to be Judaica-shaped - 20 is a relaxed ceiling, not
    # no ceiling at all.
    haggadah_loop = json.dumps({"books":
        [entry("Haggadah", "Haggadah", "Various")] * 60})
    books5, truncated5 = process_reply(haggadah_loop)
    assert truncated5, "a 60x Judaica-shaped repeat is still a runaway loop, not bulk buying"
    print("  60x Haggadah repeat -> still flagged as a loop")
    print()


def check_salvage_keeps_spine_only_entries():
    """Guard against a truncated reply with no usable *title* being thrown
    away entirely, when it still had usable *spine_text*.

    Real failure, root-caused 2026-08-25 on IMG_7103.jpeg's r1c1 tile - not a
    bookshelf at all, a stack of loose paper folders. The model read one
    illegible label into spine_text ("USA"), then degenerated into repeating
    `{"spine_text": "", "title": ""}` for the rest of its output budget and
    got cut off mid-string (num_predict truncation), so the whole reply was
    invalid JSON. `_salvage_objects` fell back to scanning for balanced
    `{...}` objects - reasonable - but only kept ones with a non-empty
    `title`, so it discarded every single one, including the one with real
    content in `spine_text`, and raised "unparseable JSON and nothing
    salvageable". `_finalise_books` would have dropped the title-less entry
    anyway, so the correct end result is "0 books, no error" (a genuinely
    bookless crop), not a hard tile failure demanding a human look at it.
    """
    print("== salvage keeps spine-only entries ==")
    from pipeline.providers import process_reply, ProviderError

    # A hand-built truncated reply in the same shape as the real capture:
    # one entry with real spine_text but no title, then filler with neither,
    # cut off mid-string before the object ever closes.
    truncated_reply = (
        '{\n  "books": [\n'
        '    {"spine_text": "USA", "title": "", "author": "", '
        '"legibility": "guess", "orientation": "vertical"},\n'
        '    {"spine_text": "", "title": "", "author": "", '
        '"legibility": "guess", "orientation": "vertical"},\n'
        '    {"spine_text": "", "title": "", "author": "", '
        '"legibility": "guess", "orientation": "vertical"},\n'
        '    {\n      "spine_text": "'
    )

    try:
        books, truncated = process_reply(truncated_reply)
    except ProviderError as exc:
        raise AssertionError(
            "a bookless-crop truncation should resolve to 0 books, not raise: {}".format(exc)
        ) from exc

    assert books == [], "a title-less salvage should still end up empty after _finalise_books"
    print("  truncated reply with spine-only content -> 0 books, no error")
    print()


def check_genre_classify():
    """Guard the batched end-of-scan genre classification pass's pure logic -
    which books get sent to the model, how the prompt numbers them, and how
    a reply gets validated back - without making a real Ollama call (that
    stays untested here, same as every other live model call in this file).

    Real problem this exists to fix: ~45% of matched books come back
    "Uncategorised" because catalogue subject tags are noisy or absent, even
    when the resolved title/author alone would be enough to guess a genre.
    """
    print("== genre classification (batched LLM pass) ==")
    from pipeline.enrich import CatalogueRecord

    has_subjects = CatalogueRecord(
        raw_title="Dune", raw_author="Herbert", spine_text="DUNE HERBERT",
        legibility="clear", seen_in=["t"], title="Dune", authors=["Frank Herbert"],
        subjects=["science fiction", "adventure"],
    )
    needs_llm = CatalogueRecord(
        raw_title="Educated", raw_author="Westover", spine_text="EDUCATED WESTOVER",
        legibility="clear", seen_in=["t"], title="Educated", authors=["Tara Westover"],
        subjects=[],
    )
    nothing_to_classify = CatalogueRecord(
        raw_title="", raw_author=None, spine_text="???",
        legibility="guess", seen_in=["t"], subjects=[],
    )

    records = [has_subjects, needs_llm, nothing_to_classify]
    candidates = genre_classify._candidates(records)
    assert candidates == [needs_llm], (
        "only the record with no keyword-classifiable subjects (and a real "
        "title) should be sent to the model"
    )
    print("  {} of {} records need the LLM pass".format(len(candidates), len(records)))

    prompt = genre_classify._build_prompt(candidates)
    assert prompt == "0. Educated - Tara Westover", "prompt must number from 0, matching response indices"
    print("  prompt numbering: {!r}".format(prompt))

    import json
    good = json.dumps({"assignments": [{"index": 0, "genre": "Biography & History"}]})
    assert genre_classify._parse_response(good, 1) == {0: "Biography & History"}

    out_of_range = json.dumps({"assignments": [{"index": 5, "genre": "Romance"}]})
    assert genre_classify._parse_response(out_of_range, 1) == {}, "an out-of-range index must be dropped"

    invented_genre = json.dumps({"assignments": [{"index": 0, "genre": "Self-Help"}]})
    assert genre_classify._parse_response(invented_genre, 1) == {}, (
        "a genre outside the fixed vocabulary must be dropped, not passed through"
    )

    truncated = '{"assignments": [{"index": 0, "genre": "Rom'
    assert genre_classify._parse_response(truncated, 1) == {}, (
        "malformed/truncated JSON must degrade to no classifications, not raise"
    )
    print("  response validation: in-range+allowed kept, everything else dropped")

    # End-to-end on the mutable field itself, bypassing the network call.
    needs_llm.llm_genre = "Biography & History"
    assert needs_llm.genre == "Biography & History", "llm_genre must win over the empty keyword result"
    assert has_subjects.genre == "Sci-Fi & Fantasy", "a record the keyword classifier already handled is untouched"
    print("  llm_genre correctly takes priority only when the keyword classifier found nothing")
    print()


def check_categorize_title_fallback():
    """categorize()'s last-resort title/spine keyword fallback, added
    2026-09-04 so placeholders (no catalogue subjects, so classify_genre()
    never runs) aren't stuck "Uncategorised" forever just because they never
    matched anything. Must stay narrow: only phrases naming the book's own
    form, not topic words a novel and a history book could both use."""
    print("== categorize() title/spine fallback (subject-less books) ==")
    from pipeline.enrich import categorize

    assert categorize("Klara and the Sun: A Novel", ["Kazuo Ishiguro"], None) == "Fiction", (
        "'a novel' in the title is a strong fiction signal with no subjects to fall back on"
    )
    assert categorize("Educated: A Memoir", ["Tara Westover"], None) == "Non-fiction", (
        "'a memoir' in the title is a strong non-fiction signal"
    )
    assert categorize("A Brief History of Time", ["Stephen Hawking"], None) is None, (
        "'history' alone must NOT trigger the fallback - it's a topic word, not a form marker, "
        "and this particular book is Science, not History"
    )
    assert categorize("Untitled Reading", [], None) is None, (
        "no signal at all must still stay None - conservative by design"
    )

    # A real catalogue-derived genre still wins outright; the title fallback
    # never even gets consulted when subjects already answered the question.
    assert categorize("Some Novel", [], "Biography & History", subjects=["memoir"]) == "Non-fiction", (
        "genre from real subjects must take priority over any title wording"
    )
    print("  'a novel'/'a memoir' style phrasing -> categorized; bare topic words -> left None")
    print()


def check_worker_routing():
    """The auto-vs-escalate decision must catch each known bad signal alone,
    and must not escalate a genuinely clean batch. This is a product decision
    (thresholds in pipeline/worker.py), so it deserves its own explicit test
    rather than being inferred from the pipeline tests above."""
    print("== worker routing decision ==")
    from pipeline.worker import decide_route

    clean = {"distinct_books": 20, "matched": 18, "needs_review": 2,
            "tiles_suspect": 0, "tiles_failed": 0}
    status, reason = decide_route(clean)
    assert status == "ready" and reason is None
    print("  clean batch -> ready")

    for flaw, patch in [
        ("repetition loop", {"tiles_suspect": 1}),
        ("hard-failed tile", {"tiles_failed": 1}),
        ("low match rate", {"matched": 10}),
        ("high needs_review", {"needs_review": 9}),
    ]:
        bad = dict(clean, **patch)
        status, reason = decide_route(bad)
        assert status == "needs_admin", "{} should have escalated".format(flaw)
        print("  {} -> needs_admin ({})".format(flaw, reason))

    status, reason = decide_route({"distinct_books": 0, "matched": 0,
                                   "needs_review": 0, "tiles_suspect": 0,
                                   "tiles_failed": 0})
    assert status == "needs_admin"
    print("  nothing extracted -> needs_admin")

    # Judaica-majority batches get a relaxed match-rate floor and needs_review
    # ceiling (see JUDAICA_MATCH_RATE_FLOOR/JUDAICA_NEEDS_REVIEW_CEILING) -
    # found 2026-08-26 on a real Hirsch/Steinsaltz/Artscroll/Koren-heavy batch
    # that hit 48% needs-review and would have escalated under the generic
    # 40% ceiling for no reason other than being a Judaica-heavy household.
    judaica_rough = {"distinct_books": 20, "matched": 10, "needs_review": 9,
                     "tiles_suspect": 0, "tiles_failed": 0, "judaica_fraction": 0.8}
    status, reason = decide_route(dict(judaica_rough, matched=10))  # 50% match rate
    assert status == "ready", "50% match rate should clear the relaxed 40% Judaica floor"
    print("  Judaica-majority batch, 50% match / 45% needs_review -> ready")

    # Same shape, generic thresholds - proves the relaxation is real, not a no-op.
    status, reason = decide_route(dict(judaica_rough, matched=10, judaica_fraction=0.0))
    assert status == "needs_admin", "the same stats without judaica_fraction must still escalate"
    print("  identical stats without judaica_fraction -> needs_admin (relaxation is Judaica-specific)")

    # A Judaica batch can still be genuinely bad - the floor is relaxed, not removed.
    status, reason = decide_route(dict(judaica_rough, matched=5, judaica_fraction=0.8))  # 25%
    assert status == "needs_admin", "25% match rate is below even the relaxed Judaica floor"
    print("  Judaica-majority batch, 25% match rate -> still needs_admin")
    print()


def check_manual_ingestion():
    """The manual (admin-via-chat-UI) path must produce the same shape as
    the automated providers, including tolerating a prose-wrapped reply -
    that's exactly the format a chat UI actually returns."""
    print("== manual ingestion path ==")
    from pipeline.manual import build_manual_packet, books_to_extracted, ingest_manual_reply

    packet = build_manual_packet("t1", ["a.jpg", "b.jpg"])
    rendered = packet.render()
    assert "a.jpg" in rendered and "b.jpg" in rendered
    assert '"books"' in rendered
    print("  packet includes photo list and schema")

    pasted = ('Sure, here you go:\n```json\n'
             '{"books": [{"spine_text": "DUNE", "title": "Dune", '
             '"author": "Frank Herbert", "legibility": "clear", '
             '"orientation": "vertical"}]}\n```\nAnything else?')
    books, truncated = ingest_manual_reply(pasted)
    assert not truncated and len(books) == 1 and books[0]["title"] == "Dune"
    extracted = books_to_extracted(books, "manual:t1")
    assert extracted[0].title == "Dune"
    print("  prose-wrapped reply parses correctly, matches ExtractedBook shape")
    print()


def check_cache_apply_restores_publisher_and_genre():
    """BookCache.apply() (pure logic, no HTTP - the cached row is just a
    dict) must restore both `publisher` and the original resolved genre
    onto a cache-hit record. Before 2026-09-11 it did neither: `publisher`
    was never set at all (so a cache-hit Judaica book lost the publisher
    signal categorize() needs - see the "catalogue-publisher signal"
    fix), and the cached genre was written to a dead `cached_genre`
    attribute the `genre` property never reads, instead of `llm_genre`,
    the one it actually prefers."""
    print("== BookCache.apply() restores publisher and genre from a cache hit ==")
    from pipeline.cache import BookCache
    from pipeline.enrich import CatalogueRecord

    record = CatalogueRecord(raw_title="Jeremiah", raw_author="Binyamin Lau",
                             spine_text="LAU JEREMIAH MAGGID", legibility="clear",
                             seen_in=["t"])
    row = {
        "id": "fake-book-id", "title": "Jeremiah", "authors": ["Binyamin Lau"],
        "genre": "Philosophy & Religion", "subjects": ["Bible", "Commentaries"],
        "publisher": "Maggid Books",
    }
    # apply() only touches its arguments - safe to call on an unconfigured
    # (no SUPABASE_URL) instance, same as this file's other pure-logic checks.
    BookCache.apply(object.__new__(BookCache), record, row)

    assert record.publisher == "Maggid Books", "publisher must be restored from the cached row"
    assert record.llm_genre == "Philosophy & Religion", (
        "the cached genre must land in llm_genre (what CatalogueRecord.genre reads), "
        "not a dead attribute"
    )
    assert record.genre == "Philosophy & Religion"
    print("  cache hit -> record.publisher and record.genre both restored correctly")

    from pipeline.enrich import categorize
    assert categorize(record.title, record.authors, record.genre,
                      subjects=record.subjects, spine_text=record.spine_text,
                      publisher=record.publisher) == "Judaica", (
        "the restored publisher must be enough on its own to re-derive Judaica on a cache hit"
    )
    print("  restored publisher alone is enough to re-derive category Judaica")
    print()


def check_defer_store():
    """defer_store=True must suppress BookCache.store() during enrich() -
    added 2026-09-11 after finding worker.py's process_batch() and
    manual_batch.py's _parse_and_enrich() both called book_cache.store() (via
    _enrich_one, mid-match) *before* genre_classify.classify_genres_batch()
    ran afterward. record.llm_genre is unset at store-time, so any book the
    keyword classifier alone couldn't place got permanently cached as
    "Uncategorised" - fixed by deferring the store and having each caller
    store explicitly once classification has run.

    Uses a real, live catalogue match (a well-known title, not mocked) with a
    lightweight fake BookCache double - only lookup()/apply()/store() are
    faked, matching this file's convention of never mocking the live HTTP
    calls themselves.
    """
    print("== defer_store suppresses the cache write until the caller is ready ==")
    from pipeline.enrich import CatalogueRecord

    class FakeCache:
        def __init__(self):
            self.store_calls = 0

        def lookup(self, record):
            return None  # force a live match, never a cache hit

        def apply(self, record, row):
            raise AssertionError("lookup() always misses here, apply() should never run")

        def store(self, record):
            self.store_calls += 1

    record = CatalogueRecord(raw_title="Dune", raw_author="Frank Herbert",
                             spine_text="DUNE HERBERT", legibility="clear", seen_in=["t"])
    fake = FakeCache()
    enrich.enrich([record], book_cache=fake, defer_store=True)
    assert record.matched, "a real, well-known title must still match live"
    assert fake.store_calls == 0, "defer_store=True must suppress store() during matching"
    print("  defer_store=True -> matched live against the real catalogue, store() not called yet")

    fake.store(record)
    assert fake.store_calls == 1
    print("  caller's own store() call after classification runs normally")
    print()


def check_pipeline():
    print("== dedupe + catalogue matching ==")
    readings = [
        ExtractedBook(spine_text=spine, title=title, author=author,
                      legibility=leg, orientation="vertical",
                      source="tile 1", seen_in=["tile 1"])
        for spine, title, author, leg in NOISY_READINGS
    ]
    print("  {} raw readings in".format(len(readings)))

    records = enrich.dedupe(readings)
    print("  {} after merging duplicates".format(len(records)))
    assert len(records) == len(readings) - 1, "the two Kafka readings should merge"

    print("  querying Open Library / Google Books...")
    errors = enrich.enrich(records)
    if errors:
        print("  lookup errors:", errors)

    print()
    print("  {:<38} {:<22} {:>6}  {}".format("AS READ", "MATCHED TO", "SCORE", "GENRE"))
    print("  " + "-" * 92)
    for record in records:
        print("  {:<38} {:<22} {:>6}  {}".format(
            record.raw_title[:37],
            (record.title or "- no match -")[:21],
            "{:.2f}".format(record.match_score) if record.matched else "-",
            record.genre[:26],
        ))

    matched = sum(1 for r in records if r.matched)
    review = sum(1 for r in records if r.needs_review)
    print()
    print("  matched {}/{}, {} flagged for review".format(matched, len(records), review))

    # The nonsense reading must NOT match anything - that is the guard against
    # the catalogue inventing a book to satisfy a bad OCR result.
    nonsense = next(r for r in records if r.raw_title.startswith("Qrztl"))
    assert not nonsense.matched, "nonsense should not match a real book"
    print("  nonsense reading correctly rejected")

    stats = {"tiles_processed": 4, "cost_usd": 0.0, "elapsed": 0.0}
    with open("out/selftest.html", "w", encoding="utf-8") as handle:
        handle.write(report.to_html(records, stats))
    with open("out/selftest.json", "w", encoding="utf-8") as handle:
        handle.write(report.to_json(records, stats))
    print("  wrote out/selftest.html and out/selftest.json")
    return records


def check_create_testers_parsing():
    """The batch tester-creation CLI's pure parsing logic - no live Supabase
    call here, same as every other Admin-API-touching path in this file."""
    print("== create_testers.py tester parsing ==")
    from create_testers import generate_pin, parse_tester

    name, email, neighbourhood = parse_tester("Sarah Cohen <sarah@example.com>")
    assert (name, email, neighbourhood) == ("Sarah Cohen", "sarah@example.com", None)
    print("  'Name <email>' -> name/email parsed, neighbourhood None")

    name, email, neighbourhood = parse_tester("David Levy <david@example.com> @Anafa")
    assert (name, email, neighbourhood) == ("David Levy", "david@example.com", "Anafa")
    print("  'Name <email> @Neighbourhood' -> all three parsed")

    try:
        parse_tester("no angle brackets here")
        raise AssertionError("a string with no <email> must be rejected")
    except ValueError:
        print("  malformed input (no <email>) -> rejected")

    pin = generate_pin()
    assert len(pin) == 6 and pin.isdigit(), "PIN must be 6 digits - meets Supabase's password minimum"
    print("  generated PIN is 6 digits: {}".format(pin))
    print()


if __name__ == "__main__":
    check_tiling()
    check_dedupe_chaining()
    check_volume_number_guard()
    check_series_detection()
    check_series_prefix_clustering()
    check_series_from_catalogue_subjects()
    check_categorize_publisher_signal()
    check_catalogue_match_disambiguation()
    check_repetition_guard()
    check_salvage_keeps_spine_only_entries()
    check_genre_classify()
    check_categorize_title_fallback()
    check_manual_ingestion()
    check_worker_routing()
    check_create_testers_parsing()
    check_defer_store()
    check_cache_apply_restores_publisher_and_genre()
    check_pipeline()
    print()
    print("All checks passed.")
