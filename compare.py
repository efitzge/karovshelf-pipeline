"""Score a scan against a human-checked reading of the same shelf.

Counting books is not evidence: two scans can both report 42 and disagree about
every one. This matches title-by-title so we can see what was actually found,
what was missed, and - the expensive kind of error - what was invented.

    python compare.py out/local_shelf1.json
"""

import json
import sys

from pipeline.enrich import normalise, title_similarity
from realshelf import READINGS

# Two titles count as the same book at or above this similarity. Loose enough
# to forgive "Black Holes and Time Warps" vs "Black Holes & Time Warps".
SAME_BOOK = 0.80


def load_scan(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    out = []
    for book in payload["books"]:
        out.append({
            "title": book["raw"]["title"] or book["title"],
            "author": (book["raw"]["author"] or ""),
            "spine": book["raw"]["spine_text"],
            "matched": book["matched"],
        })
    return out, payload.get("stats", {})


def truth():
    return [{"title": t, "author": a or "", "spine": s} for s, t, a, _ in READINGS]


def pair_up(found, expected):
    """Greedily pair each expected book with its best unused scan result."""
    unused = list(range(len(found)))
    hits, misses = [], []
    for want in expected:
        best_i, best_score = None, 0.0
        for i in unused:
            score = title_similarity(want["title"], found[i]["title"])
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= SAME_BOOK:
            unused.remove(best_i)
            hits.append((want, found[best_i], best_score))
        else:
            misses.append(want)
    extras = [found[i] for i in unused]
    return hits, misses, extras


def author_ok(want, got):
    if not want["author"]:
        return True
    if not got["author"]:
        return None  # not claimed - neither right nor wrong
    return title_similarity(want["author"], got["author"]) >= 0.6


def score(path):
    """Score one scan against the ground truth. Returns a plain dict."""
    found, stats = load_scan(path)
    expected = truth()
    hits, misses, extras = pair_up(found, expected)

    wrong_author = [(w, g) for w, g, _ in hits if author_ok(w, g) is False]
    no_author = [(w, g) for w, g, _ in hits if author_ok(w, g) is None]

    return {
        "provider": stats.get("provider", "?"),
        "elapsed": stats.get("elapsed", 0),
        "cost_usd": stats.get("cost_usd", 0),
        "expected": len(expected),
        "reported": len(found),
        "found": len(hits),
        "missed": len(misses),
        "invented": len(extras),
        "right_author": len(hits) - len(wrong_author) - len(no_author),
        "wrong_author": len(wrong_author),
        "no_author": len(no_author),
        "catalogued": sum(1 for _, g, _ in hits if g["matched"]),
        "recall": len(hits) / len(expected) if expected else 0.0,
        "miss_titles": [w["title"] for w in misses],
        "invented_titles": [g["title"] for g in extras],
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/local_shelf1.json"
    found, stats = load_scan(path)
    expected = truth()

    hits, misses, extras = pair_up(found, expected)

    print("=" * 78)
    print("SCAN:  {}".format(path))
    print("MODEL: {}".format(stats.get("provider", "?")))
    print("TIME:  {}s   COST: {}".format(
        stats.get("elapsed", "?"),
        "free" if not stats.get("cost_usd") else "${:.3f}".format(stats["cost_usd"])))
    print("=" * 78)
    print()
    print("Books on the shelf (human-checked): {}".format(len(expected)))
    print("Books the scan reported:            {}".format(len(found)))
    print()
    print("FOUND:   {:>2}/{}  ({:.0f}%)".format(
        len(hits), len(expected), len(hits) / len(expected) * 100))
    print("MISSED:  {:>2}".format(len(misses)))
    print("INVENTED:{:>2}   <- books reported that are not on the shelf".format(
        len(extras)))
    print()

    wrong_author = [(w, g) for w, g, _ in hits if author_ok(w, g) is False]
    no_author = [(w, g) for w, g, _ in hits if author_ok(w, g) is None]
    print("Of the {} found:".format(len(hits)))
    print("  correct author : {}".format(len(hits) - len(wrong_author) - len(no_author)))
    print("  wrong author   : {}   <- hallucination risk".format(len(wrong_author)))
    print("  no author given: {}".format(len(no_author)))
    print()

    if misses:
        print("-" * 78)
        print("MISSED ({}):".format(len(misses)))
        for want in misses:
            print("   {}".format(want["title"][:70]))
        print()

    if extras:
        print("-" * 78)
        print("INVENTED / SPURIOUS ({}):".format(len(extras)))
        for got in extras:
            print("   {:<44} spine: {}".format(
                got["title"][:43], (got["spine"] or "-")[:28]))
        print()

    if wrong_author:
        print("-" * 78)
        print("WRONG AUTHOR ({}):".format(len(wrong_author)))
        for want, got in wrong_author:
            print("   {:<38} said {:<20} (really {})".format(
                want["title"][:37], got["author"][:19], want["author"][:20]))
        print()

    catalogued = sum(1 for _, g, _ in hits if g["matched"])
    print("=" * 78)
    print("END TO END: {}/{} books on this shelf ended up correctly catalogued "
          "({:.0f}%)".format(catalogued, len(expected),
                             catalogued / len(expected) * 100))
    print("=" * 78)


if __name__ == "__main__":
    main()
