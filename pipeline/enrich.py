"""Turn raw spine readings into canonical catalogue records.

This is the half of the pipeline that makes the AI half trustworthy. A spine
reading like "MURAKAM1 KAFKA ON SHORE" is useless on its own, but matched
against a real bibliographic database it becomes a record with a correct title,
a correct author, an ISBN, a cover image, and subject tags we can use as genres.

Just as importantly, a reading that matches *nothing* is a signal: it is either
a rare book or a misread, and either way it belongs in the human review queue
rather than silently in the catalogue.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

_ARTICLES = ("the ", "a ", "an ")
_NOISE = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalise(text):
    """Reduce a title to a comparable core: lowercase, no punctuation, no article."""
    if not text:
        return ""
    text = _NOISE.sub(" ", text.lower())
    text = _SPACES.sub(" ", text).strip()
    for article in _ARTICLES:
        if text.startswith(article):
            text = text[len(article):]
            break
    return text


def similarity(a, b):
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def _word_count(text):
    """Word count of the RAW title (article included, not normalised) - a
    cheap, explicit guard against the one collision no amount of fuzzy-ratio
    tuning fixes: normalising strips a leading article, so "Jonah" and "The
    Jonah" become the SAME string and score a perfect 1.0 against each other
    even though they are different books (real scan, 2026-09-03). Comparing
    raw word counts catches exactly that - 1 word vs 2 - without touching
    `normalise()` itself, which plenty of other, correct matching relies on."""
    return len((text or "").split())


# A reading that ends in one of these is a truncated fragment, not a title.
# Without this guard the prefix rule below turns "Dungeons of" into a 0.95 match
# for "The Dungeons of Nakasero" - a completely different book.
_DANGLING_WORDS = {
    "of", "the", "a", "an", "and", "in", "on", "to", "for", "with", "from",
    "by", "at", "or", "de", "la", "le", "el", "und", "des", "du",
}


def _is_fragment(text):
    """True if a normalised title looks cut off rather than complete."""
    words = text.split()
    return bool(words) and words[-1] in _DANGLING_WORDS


def title_similarity(a, b, allow_prefix_boost=True):
    """Compare titles with tolerance for missing subtitles.

    Spines very often carry only the main title while the catalogue record
    carries the full one - "Sapiens" on the shelf, "Sapiens: A Brief History of
    Humankind" in the database. Treating one title as a prefix of the other as a
    near-match fixes a whole class of false misses.

    Two guards keep that from over-firing: a length floor, so a short title does
    not match its own sequel ("Dune" vs "Dune Messiah"); and a fragment check, so
    a spine read only as far as "Dungeons of" does not confidently bind to the
    first catalogue entry that happens to continue those words.

    `allow_prefix_boost=False` disables the boost entirely, returning the raw
    ratio - for the no-author-to-corroborate matching path in `enrich()`
    (see its own comment): the boost is exactly the mechanism that let
    "Isaiah" bind to a biography titled "Isaiah Berlin" and "Ecclesiastes"
    bind to an unrelated commentary series, both on real scans with no
    author on the spine to catch it (2026-09-03).
    """
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    base = SequenceMatcher(None, na, nb).ratio()
    if not allow_prefix_boost:
        return base
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 6 and longer.startswith(shorter) and not _is_fragment(shorter):
        return max(base, 0.95)
    return base


# Subject lists from both catalogues are long, noisy, and in no useful order -
# taken raw they produce "genres" like "sex" or "goldfinch". Mapping them onto a
# small fixed vocabulary is what makes browsing by genre actually work.
#
# The rules are tiered rather than a flat list because topical subjects would
# otherwise beat the genre. A novel tagged {Fiction, Family Life, Historical}
# hits the word "historical" and lands under History unless specific fiction
# genres are checked first.

# Condensed to 10 browsable genres (from an earlier 21-value version - too
# many to be useful in a dropdown). Six fiction, four non-fiction; kept in
# the same tiered shape as before so a specific hit always beats a generic
# "this is fiction" marker, which in turn beats a topical non-fiction guess.
#
# Tier 1: specific genres, fiction and non-fiction alike, that are unambiguous
# whenever their keywords appear at all.
SPECIFIC_GENRES = [
    ("Sci-Fi & Fantasy", ("science fiction", "sci-fi", "dystopia", "space opera",
                          "time travel", "fantasy", "dragons", "wizards",
                          "mythical", "sword and sorcery")),
    ("Mystery, Thriller & Horror", ("mystery", "detective", "crime", "thriller",
                                    "suspense", "noir", "murder", "espionage",
                                    "horror", "ghost stories", "vampires")),
    ("Romance", ("romance", "love stories")),
    ("Poetry, Drama & Graphic Novels", ("comic", "graphic novel", "manga",
                                        "poetry", "poems", "verse", "drama",
                                        "plays", "theater", "theatre",
                                        "shakespeare")),
    ("Children's & Young Adult", ("juvenile literature", "picture books",
                                  "children's stories", "young adult",
                                  "juvenile fiction", "teenage")),
    ("Biography & History", ("biography", "memoir", "autobiography",
                             "personal narratives", "anecdotes", "correspondence")),
    ("Society, Business & Lifestyle", ("cooking", "cookery", "recipes", "travel",
                                       "voyages", "description and travel")),
]

# Tier 2: markers that say "this is a novel". Checked before any topical
# subject, so a historical novel does not get filed under Biography &
# History. Unlike the old scheme, historical and literary fiction share one
# consolidated bucket now, so there is no need to branch on which marker
# matched - either one lands the same place.
FICTION_MARKERS = ("fiction", "novel", "literature", "literary", "stories",
                   "historical", "history")

# Tier 3: topical non-fiction, only consulted once fiction has been ruled out.
TOPICAL_GENRES = [
    ("Biography & History", ("history", "historical", "civilization", "war",
                             "ancient", "revolution")),
    ("Science & Nature", ("science", "physics", "biology", "astronomy", "evolution",
                          "nature", "mathematics", "medicine", "technology")),
    ("Philosophy & Religion", ("philosophy", "religion", "theology", "ethics",
                               "spiritual", "judaism", "bible", "islam",
                               "christianity")),
    ("Society, Business & Lifestyle", ("political", "politics", "sociology",
                                       "social", "economics", "government", "law",
                                       "business", "management", "self-help",
                                       "success", "psychology", "leadership",
                                       "finance", "art", "painting", "music",
                                       "photography", "architecture")),
]


def classify_genre(subjects):
    """Fold a noisy subject list down to one of 10 browsable genres."""
    haystack = " | ".join(subjects or []).lower()
    if not haystack:
        return None

    for genre, keywords in SPECIFIC_GENRES:
        if any(keyword in haystack for keyword in keywords):
            return genre

    if any(marker in haystack for marker in FICTION_MARKERS):
        return "Literary & Historical Fiction"

    for genre, keywords in TOPICAL_GENRES:
        if any(keyword in haystack for keyword in keywords):
            return genre
    return None


# --- Coarse top-level category (Fiction / Non-fiction / Judaica) ------------
#
# Separate from genre on purpose. General catalogues (Open Library, Google
# Books) essentially never tag "Judaica" - a Koren Talmud volume or an
# ArtScroll halacha guide comes back with noisy or empty subjects, which is
# exactly why so many of these end up "Uncategorised" by genre. Judaica gets
# its own detection pass, checked BEFORE genre, using signals genre ignores
# entirely: known publishers/imprints, subject-matter keywords, and Hebrew
# script itself - any of the three is a strong, low-false-positive signal.
JUDAICA_PUBLISHERS = (
    "koren", "maggid", "artscroll", "mesorah", "feldheim", "yeshiva university",
    "ktav", "targum press", "jewish publication society", "mosaica press",
    "metsudah",
)
JUDAICA_KEYWORDS = (
    "judaica", "torah", "talmud", "mishnah", "mishnayot", "halacha", "halachos",
    "halakhah", "mussar", "parasha", "parshah", "midrash", "tanakh", "siddur",
    "haggadah", "niddah", "kashrus", "kashrut", "brochos", "brachos",
    "shemoneh esrei", "gemara", "kabbalah", "chumash", "hebrew", "yeshiva",
    "shabbos", "shabbat", "kosher", "shulchan aruch", "rashi",
)
HEBREW_SCRIPT_RE = re.compile(r"[֐-׿]")

# The single source of truth for the 10 browsable genres - same order as
# `karovshelf-web/src/app/library/genres.ts`'s dropdown, which must stay in
# sync by hand since the two are different languages/repos. FICTION_GENRES/
# NONFICTION_GENRES below are just named slices of this, so a genre can never
# exist in one list but not the other.
GENRES = (
    "Sci-Fi & Fantasy",
    "Mystery, Thriller & Horror",
    "Romance",
    "Literary & Historical Fiction",
    "Children's & Young Adult",
    "Poetry, Drama & Graphic Novels",
    "Biography & History",
    "Science & Nature",
    "Philosophy & Religion",
    "Society, Business & Lifestyle",
)
FICTION_GENRES = set(GENRES[:6])
NONFICTION_GENRES = set(GENRES[6:])


# Tier 4 (categorize() only, last resort): title/spine phrasing that names
# its own form directly. A placeholder (unmatched book) has no catalogue
# `subjects` at all, so classify_genre() never gets a chance to run - it was
# landing in "Uncategorised" forever regardless of how obviously fictional
# the book was (found 2026-09-04, since placeholders are exactly the books
# most likely to have no cover either, so they're the ones that most need a
# category to pick a fallback cover tile). Deliberately narrow compared to
# FICTION_MARKERS/TOPICAL_GENRES above: those are tuned for noisy but
# reliable Library-of-Congress subject strings, where "history" or "war"
# really does mean the book is about that. In a TITLE the same word is as
# likely to belong to a novel as a history book, so only phrases that name
# the book's own form (not just a topic it touches) are safe here.
TITLE_FICTION_MARKERS = ("a novel", "a novella")
TITLE_NONFICTION_MARKERS = (
    "a memoir", "an autobiography", "a biography of", "a history of",
    "a guide to", "field guide", "the true story", "an introduction to",
    "encyclopedia of", "dictionary of", "cookbook", "for dummies",
)


def categorize(title, authors, genre, subjects=None, spine_text=None):
    """Fiction / Non-fiction / Judaica, or None when genuinely unsure.

    Deliberately conservative: this stays None rather than guess when there's
    no real signal either way, matching how the rest of the app treats
    uncertainty (`is_placeholder`, `legibility: guess`) - a wrong category is
    worse than an unlabelled one. `spine_text`, when available, often carries
    the publisher name that the cleaned-up `title` doesn't (e.g. a Koren/
    Maggid book's spine literally says "MAGGID" - the resolved title won't).
    """
    haystack = " | ".join(
        filter(None, [title, spine_text] + list(authors or []) + list(subjects or []))
    ).lower()

    if HEBREW_SCRIPT_RE.search((title or "") + (spine_text or "")):
        return "Judaica"
    if any(pub in haystack for pub in JUDAICA_PUBLISHERS):
        return "Judaica"
    if any(keyword in haystack for keyword in JUDAICA_KEYWORDS):
        return "Judaica"

    if genre in FICTION_GENRES:
        return "Fiction"
    if genre in NONFICTION_GENRES:
        return "Non-fiction"

    if any(marker in haystack for marker in TITLE_NONFICTION_MARKERS):
        return "Non-fiction"
    if any(marker in haystack for marker in TITLE_FICTION_MARKERS):
        return "Fiction"
    return None


@dataclass
class CatalogueRecord:
    """A book after enrichment - what would actually go in the database."""

    raw_title: str
    raw_author: str | None
    spine_text: str
    legibility: str
    seen_in: list
    times_seen: int = 1

    # The title this cluster was created with, frozen. Dedupe compares against
    # THIS, never against `raw_title`, which may be upgraded to a more legible
    # reading as copies merge in. Letting the comparison key drift turns greedy
    # merging into a random walk across the title space: on a real 759-reading
    # run, two clusters swallowed 218 and 202 unrelated books that way.
    match_title: str | None = None

    def __post_init__(self):
        if self.match_title is None:
            self.match_title = self.raw_title

    matched: bool = False
    match_score: float = 0.0
    source: str | None = None  # which catalogue answered

    title: str | None = None
    authors: list = field(default_factory=list)
    first_published: int | None = None
    isbn: str | None = None
    cover_url: str | None = None
    subjects: list = field(default_factory=list)
    lookup_errors: list = field(default_factory=list)

    # A series name straight from the catalogue's own "series:X"-tagged
    # subject (Open Library only - see _series_from_subjects), for series
    # whose volumes don't share a common title/prefix at all (The Expanse:
    # "Leviathan Wakes", "Caliban's War", ... - detect_series() needs an
    # explicit "Vol. N" and cluster_series_by_prefix() needs a shared
    # prefix, so differently-titled series like this were a known,
    # documented gap neither one covers). No volume number comes with it -
    # only used as a fallback series_title when detect_series() finds
    # nothing (see cache.py's store()).
    catalogue_series_title: str | None = None

    # Set by the optional batched end-of-scan LLM classification pass
    # (genre_classify.classify_genres_batch), for records whose catalogue
    # subjects gave classify_genre() nothing to work with. None otherwise.
    llm_genre: str | None = None

    # The Supabase `books.id` this record resolved to, if any - set by
    # BookCache on a cache hit or a fresh cacheable store. None for a CLI-only
    # run with no cache configured, or a match that didn't clear the cache's
    # confidence bar. The worker (pipeline/worker.py) uses this to know
    # whether it still needs to create a books row before it can write a
    # library_items FK.
    book_id: str | None = None

    @property
    def display_title(self):
        return self.title or self.raw_title

    @property
    def display_author(self):
        if self.authors:
            return ", ".join(self.authors)
        return self.raw_author or "Unknown"

    @property
    def genre(self):
        """A single coarse genre.

        Preference order: the batched end-of-scan LLM classification pass
        (`llm_genre`, only ever set when the catalogue subjects gave the
        keyword classifier nothing), then the keyword classifier itself,
        then "Uncategorised" when neither found anything.
        """
        return self.llm_genre or classify_genre(self.subjects) or "Uncategorised"

    @property
    def needs_review(self):
        """Should a human confirm this before it enters the catalogue?"""
        return (not self.matched) or self.legibility == "guess" or self.match_score < 0.85


# Volumes of the same work often produce titles that differ from each other
# by only ONE number token, leading or trailing - "Kitzur Shulchan Aruch I"
# vs "...II", "The Sandman, Vol. 9" vs "...Vol. 10", or the classic Biblical/
# Judaica split-book convention "I Kings" vs "II Kings" (also I/II Samuel,
# Chronicles, Corinthians, Timothy...) - and that one-token difference is
# nowhere near enough to drop `similarity()` below DEDUPE_THRESHOLD (real
# scans: 0.889-0.979 for the trailing case, 0.933 for "I Kings"/"II Kings").
# Left unguarded, dedupe silently swallows every volume after the first into
# one record - found 2026-09-03 on real scans, losing "Kitzur Shulchan Aruch
# II", three Sandman volumes (7, 8, 9), and nearly losing "II Kings" into "I
# Kings" the same way before this was caught. A whitelist of the actual
# numbers/numerals people use for volumes (not a general roman-numeral regex,
# which would also match ordinary words like "MIX" or "DIM") is enough: if
# both titles carry a marker in the SAME position (both leading, or both
# trailing) and the values differ, they are different volumes - never merge,
# no matter how similar the rest of the title reads. Two titles that both
# happen to start with an ordinary word matching a numeral (e.g. two
# different books both starting with "I") are unaffected - equal markers
# never block a merge, only differing ones do.
_ROMAN_NUMERALS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}


def _volume_marker_token(token):
    """An arabic integer or roman numeral (I-XX) value for one bare token,
    or None if the token isn't a bare volume/part number."""
    token = token.strip(".,:;")
    if token.isdigit():
        return int(token)
    return _ROMAN_NUMERALS.get(token.upper())


def _leading_volume_marker(title):
    """The title's first token, if it's a bare volume/part number."""
    if not title:
        return None
    return _volume_marker_token(title.strip().split()[0])


def _trailing_volume_marker(title):
    """The title's last token, if it's a bare volume/part number."""
    if not title:
        return None
    return _volume_marker_token(title.strip().split()[-1])


_VOLUME_WORD_PATTERN = re.compile(
    r"^(?P<base>.+?)[,:]?\s+(?:vol(?:ume)?|part|pt)\.?\s*(?P<num>[0-9]+|[ivxlcdm]+)\b",
    re.IGNORECASE,
)


def detect_series(title):
    """(base_title, volume_number) for an obviously-numbered volume in a
    multi-part set, or None if the title doesn't carry one.

    Built for display grouping ("A Daily Dose of Torah, Vol. 1-13" as one
    collapsed card instead of 13 separate entries - see PROJECT_SUMMARY.md
    #10), not for matching or dedupe - a false positive here just fails to
    group two books together, it can never wrongly merge them the way a
    dedupe false positive would (that guard, in dedupe() below, is separate
    and deliberately stricter).

    Deliberately requires an explicit "Vol./Volume/Part N" marker - it does
    NOT fall back to _leading_volume_marker/_trailing_volume_marker's bare-
    numeral check the way the dedupe guard does. That bare check is safe
    there because it only ever *blocks* an already-near-identical-title
    merge; reused here as a fallback it produced real false positives on
    the actual library the first time this ran - "I Believe" and "I asked
    for wonder" read as "Vol. 1" (bare leading "I"), "Fahrenheit 451" as
    "Vol. 451" and "60 Days Away" as "Vol. 60" (bare trailing numeral), "The
    ABCs of the UCC, Articles 3 and 4" as "Vol. 4". The cost of dropping it
    is missing a few keyword-less cases like "I Kings"/"II Kings" - an
    acceptable trade for not mistagging ordinary titles as volume 1 of a
    series that doesn't exist.
    """
    if not title:
        return None
    m = _VOLUME_WORD_PATTERN.match(title.strip())
    if not m:
        return None
    num = _volume_marker_token(m.group("num"))
    if num is None:
        return None
    base = m.group("base").strip().rstrip(",:")
    return (base, num) if base else None


_PREFIX_SPLIT = re.compile(r"[,:]")


def _title_prefix(title, min_length=12):
    """Text before the title's first comma or colon, or None if that's
    shorter than `min_length` - too short to be a meaningful series-name
    signal on its own (a bare "The" matching half the library isn't one).
    """
    if not title:
        return None
    part = _PREFIX_SPLIT.split(title, maxsplit=1)[0].strip()
    return part if len(part) >= min_length else None


def cluster_series_by_prefix(books, min_group_size=3, min_prefix_length=12):
    """Existing-library backfill pass, complementary to detect_series():
    catches a real multi-book set whose volumes carry no "Vol. N" marker at
    all, just a shared series-name prefix before the per-volume subtitle -
    "A Daily Dose of Torah: Daily study for the weeks of X-Y" (14 real
    volumes in production, none individually detectable by detect_series())
    is the case that prompted this, at Elliot's request 2026-09-03.

    `books`: an iterable of dicts with `id`, `title`, `series_title`,
    `owner_id`. Returns a list of (book_id, canonical_series_title) pairs
    to write - only for books whose series_title needs to change, so a
    re-run is a no-op once applied.

    Clustering is scoped per-owner (the same prefix from two different
    owners' shelves is not one set), and requires 3+ real members sharing
    the prefix - a lone match or a pair is left alone, same MIN_GROUP_SIZE
    convention as the display grouping this feeds
    (karovshelf-web/src/lib/seriesGrouping.ts). Purely additive: a member
    that already has a series_title from detect_series() (an explicit
    "Vol. N") keeps contributing its real volume number - this only fills
    in series_title for its prefix-matched siblings, using the majority
    already-assigned title as canonical when one exists, so "A Daily Dose
    of Torah, Vol. 11" and its 12 unnumbered siblings end up in the exact
    same display group rather than two separate ones.
    """
    from collections import Counter, defaultdict

    by_key = defaultdict(list)
    for book in books:
        prefix = _title_prefix(book["title"], min_prefix_length)
        if not prefix:
            continue
        by_key[(book["owner_id"], prefix.lower())].append(book)

    updates = []
    for members in by_key.values():
        if len(members) < min_group_size:
            continue
        existing_titles = [m["series_title"] for m in members if m.get("series_title")]
        if existing_titles:
            canonical = Counter(existing_titles).most_common(1)[0][0]
        else:
            prefixes = [_title_prefix(m["title"], min_prefix_length) for m in members]
            canonical = Counter(prefixes).most_common(1)[0][0]
        for member in members:
            if member.get("series_title") != canonical:
                updates.append((member["id"], canonical))
    return updates


def dedupe(books):
    """Collapse the same physical book seen in several overlapping tiles.

    Two readings merge when their normalised titles are close enough AND their
    authors do not actively contradict each other - so two different books in a
    series keep their separate entries, but "Kafka on the Shore" seen twice
    becomes one record.
    """
    merged = []
    for book in books:
        target = None
        best_score = 0.0
        for candidate in merged:
            # Compare against the cluster's frozen anchor, not its current
            # display title, so a cluster can only ever absorb books similar to
            # what it started as.
            score = similarity(book.title, candidate.match_title)
            if score < config.DEDUPE_THRESHOLD:
                continue
            both_named = book.author and candidate.raw_author
            if both_named and similarity(book.author, candidate.raw_author) < 0.6:
                continue
            this_lead = _leading_volume_marker(book.title)
            cand_lead = _leading_volume_marker(candidate.match_title)
            if this_lead is not None and cand_lead is not None and this_lead != cand_lead:
                continue
            this_vol = _trailing_volume_marker(book.title)
            cand_vol = _trailing_volume_marker(candidate.match_title)
            if this_vol is not None and cand_vol is not None and this_vol != cand_vol:
                continue
            # Take the best match rather than the first one found; with many
            # similar titles, "first" is an accident of ordering.
            if score > best_score:
                target, best_score = candidate, score

        if target is None:
            merged.append(
                CatalogueRecord(
                    raw_title=book.title,
                    raw_author=book.author,
                    spine_text=book.spine_text,
                    legibility=book.legibility,
                    seen_in=list(book.seen_in),
                )
            )
        else:
            target.times_seen += 1
            target.seen_in.extend(book.seen_in)
            # Keep the most confident reading of the pair.
            ranking = {"clear": 0, "partial": 1, "guess": 2}
            if ranking.get(book.legibility, 2) < ranking.get(target.legibility, 2):
                target.legibility = book.legibility
                target.raw_title = book.title
                target.spine_text = book.spine_text
            if book.author and not target.raw_author:
                target.raw_author = book.author
    return merged


def _session():
    """A session that retries transient catalogue failures.

    Both free catalogues fail intermittently under load - Google Books in
    particular returns 503 `backendFailed` on a large fraction of requests at
    busy times, unrelated to the query or the key. Without retries a single
    blip silently drops a book from someone's library forever, which is a much
    worse outcome than waiting a couple of seconds.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})

    retry = Retry(
        total=config.HTTP_RETRIES,
        backoff_factor=config.HTTP_BACKOFF,
        backoff_jitter=0.3,          # de-synchronise our own parallel workers
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,       # let our own code see the final status
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16, pool_connections=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _clean_subjects(subjects):
    """Open Library subjects are noisy; keep the ones that read like genres."""
    skip = ("accessible book", "protected daisy", "in library", "internet archive",
            "overdrive", "large type", "reading level")
    out = []
    for subject in subjects or []:
        low = subject.lower()
        if low.startswith("series:"):
            continue
        if any(bad in low for bad in skip):
            continue
        if len(subject) > 40 or subject.isdigit():
            continue
        if subject not in out:
            out.append(subject)
        if len(out) >= 6:
            break
    return out


def _series_from_subjects(subjects):
    """Open Library tags some editions' subjects with "series:<name>" -
    e.g. "series:The Expanse". The only real signal this pipeline has for a
    series whose volumes share no common title wording at all (see
    CatalogueRecord.catalogue_series_title). Takes the first one found;
    real books rarely carry more than one."""
    for subject in subjects or []:
        if subject.lower().startswith("series:"):
            name = subject[len("series:"):].strip()
            if name:
                return name
    return None


OL_FIELDS = "title,author_name,first_publish_year,isbn,cover_i,subject"


def _score_candidate(record, candidate_title, candidate_authors):
    """Score one catalogue candidate against a scanned record, or None to
    reject it outright. Shared by both `_try_openlibrary` and
    `_try_google_books` so the two catalogues can never drift apart on this.

    Two paths, chosen by whether the author actually corroborates:
    - Corroborated (author present and clears AUTHOR_CORROBORATION_FLOOR):
      the original blended score - title weighted more heavily, since a
      right title with a misread author should still match.
    - Uncorroborated (no author on the scan, or one that doesn't clear the
      floor): title alone decides, but with the prefix-continuation boost
      OFF and a much higher bar (UNCORROBORATED_MATCH_THRESHOLD), plus an
      exact word-count match required. See config.py's comments for the
      real false matches (Isaiah/Isaiah Berlin, Jonah/The Jonah, Ecclesiastes
      and others, 2026-09-03) that this path exists to block - all of them
      had no author on the spine and rode in on the boost or on article-
      stripping alone.
    """
    author_score = None
    if record.raw_author and candidate_authors:
        author_score = max(similarity(record.raw_author, name) for name in candidate_authors)

    if author_score is not None and author_score >= config.AUTHOR_CORROBORATION_FLOOR:
        title_score = title_similarity(record.raw_title, candidate_title)
        return title_score * 0.7 + author_score * 0.3

    if _word_count(record.raw_title) != _word_count(candidate_title):
        return None
    title_score = title_similarity(record.raw_title, candidate_title, allow_prefix_boost=False)
    if title_score < config.UNCORROBORATED_MATCH_THRESHOLD:
        return None
    return title_score


def _try_openlibrary(session, record, language=None):
    # Open Library's structured title=&author= search is an AND that is far too
    # strict for OCR input - "Kafka on the Shore" + "Murakami" returns zero
    # results. The free-text q= search handles the same query fine, so combine
    # the two fields there instead.
    query = record.raw_title
    if record.raw_author:
        query = query + " " + record.raw_author

    params = {"q": query, "limit": 8, "fields": OL_FIELDS}
    if language:
        # Without this, a search for an English title happily returns the
        # original-language edition (Kafka on the Shore comes back as
        # the Japanese-language record), which then fails to match.
        params["language"] = language

    response = session.get(config.OPENLIBRARY_SEARCH, params=params,
                           timeout=config.HTTP_TIMEOUT)
    response.raise_for_status()
    docs = response.json().get("docs", [])

    best = None
    best_score = 0.0
    for doc in docs:
        score = _score_candidate(record, doc.get("title", ""), doc.get("author_name"))
        if score is not None and score > best_score:
            best, best_score = doc, score

    if not best or best_score < config.MATCH_THRESHOLD:
        return False

    record.matched = True
    record.match_score = round(best_score, 3)
    record.source = "openlibrary"
    record.title = best.get("title")
    record.authors = best.get("author_name", [])[:3]
    record.first_published = best.get("first_publish_year")
    isbns = best.get("isbn") or []
    record.isbn = isbns[0] if isbns else None
    if best.get("cover_i"):
        record.cover_url = "https://covers.openlibrary.org/b/id/{}-M.jpg".format(
            best["cover_i"])
    record.subjects = _clean_subjects(best.get("subject"))
    record.catalogue_series_title = _series_from_subjects(best.get("subject"))
    return True


def _try_google_books(session, record, language=None):
    # Quoted intitle:/inauthor: is likewise too strict for noisy readings; a
    # plain keyword query recalls far better and we do our own scoring anyway.
    query = record.raw_title
    if record.raw_author:
        query = query + " " + record.raw_author

    params = {"q": query, "maxResults": 8}
    if language:
        params["langRestrict"] = language[:2]
    if config.GOOGLE_BOOKS_API_KEY:
        params["key"] = config.GOOGLE_BOOKS_API_KEY

    response = session.get(config.GOOGLE_BOOKS_SEARCH, params=params,
                           timeout=config.HTTP_TIMEOUT)
    response.raise_for_status()
    items = response.json().get("items", [])

    best = None
    best_score = 0.0
    for item in items:
        info = item.get("volumeInfo", {})
        score = _score_candidate(record, info.get("title", ""), info.get("authors"))
        if score is not None and score > best_score:
            best, best_score = info, score

    if not best or best_score < config.MATCH_THRESHOLD:
        return False

    record.matched = True
    record.match_score = round(best_score, 3)
    record.source = "googlebooks"
    record.title = best.get("title")
    record.authors = best.get("authors", [])[:3]
    year = (best.get("publishedDate") or "")[:4]
    record.first_published = int(year) if year.isdigit() else None
    for ident in best.get("industryIdentifiers", []):
        if ident.get("type") in ("ISBN_13", "ISBN_10"):
            record.isbn = ident.get("identifier")
            break
    links = best.get("imageLinks") or {}
    record.cover_url = links.get("thumbnail") or links.get("smallThumbnail")
    record.subjects = _clean_subjects(best.get("categories"))
    return True


def _describe_http_error(exc):
    """Turn a requests exception into something a human can act on."""
    response = getattr(exc, "response", None)
    if response is None:
        return "network: " + type(exc).__name__
    if response.status_code == 429:
        return "429 quota exceeded"
    return "HTTP " + str(response.status_code)


def _enrich_one(session, record, language=None, book_cache=None):
    """Try each catalogue in turn. Never let a lookup kill the run.

    The shared cache is consulted first. A hit costs one indexed query against
    our own database instead of a call to an external API with a daily quota -
    which is the difference between the product working for the tenth user of
    the day and not.

    A lookup that *errors* is recorded rather than swallowed. That distinction
    matters more than it looks: if Google Books is rate-limiting, every book
    silently comes back unmatched and the whole scan looks like a model failure
    when it is really an outage in the enrichment layer.
    """
    if book_cache is not None:
        cached = book_cache.lookup(record)
        if cached:
            book_cache.apply(record, cached)
            return record

    lookups = [("openlibrary", _try_openlibrary), ("googlebooks", _try_google_books)]

    for name, lookup in lookups:
        try:
            if lookup(session, record, language=language):
                if book_cache is not None:
                    book_cache.store(record)
                return record
        except requests.RequestException as exc:
            record.lookup_errors.append(name + ": " + _describe_http_error(exc))
        except (ValueError, KeyError) as exc:
            record.lookup_errors.append(name + ": bad response (" + type(exc).__name__ + ")")
        time.sleep(0.1)  # gentle on the free APIs

    # A language-filtered search can miss a book the unfiltered one finds, so
    # fall back to searching everything before declaring no match.
    if language:
        try:
            if _try_openlibrary(session, record, language=None) and book_cache:
                book_cache.store(record)
        except (requests.RequestException, ValueError, KeyError):
            pass
    return record


def enrich(records, language="eng", max_workers=6, progress=None, book_cache=None):
    """Look every deduped reading up in a real bibliographic catalogue.

    Returns a dict of per-source error counts so the caller can tell a genuine
    "this book is not in the catalogue" from "the catalogue was unreachable".
    """
    session = _session()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_enrich_one, session, record, language, book_cache)
                   for record in records]
        for future in as_completed(futures):
            future.result()
            done += 1
            if progress and done % 10 == 0:
                progress("  looked up {}/{}".format(done, len(records)))

    errors = {}
    for record in records:
        for message in record.lookup_errors:
            errors[message] = errors.get(message, 0) + 1
    return errors
