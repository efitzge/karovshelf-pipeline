"""Render the scan results so a human can judge accuracy at a glance.

The point of this report is not to look pretty - it is to answer one question
fast: how often did the model get it right? So every row shows the raw spine
transcription next to the catalogue match, which makes misreads obvious.
"""

import html
import json


def to_json(records, stats):
    payload = {
        "stats": stats,
        "books": [
            {
                "title": r.display_title,
                "authors": r.authors or ([r.raw_author] if r.raw_author else []),
                "genre": r.genre,
                "subjects": r.subjects,
                "isbn": r.isbn,
                "first_published": r.first_published,
                "cover_url": r.cover_url,
                "matched": r.matched,
                "match_score": r.match_score,
                "match_source": r.source,
                "needs_review": r.needs_review,
                "raw": {
                    "spine_text": r.spine_text,
                    "title": r.raw_title,
                    "author": r.raw_author,
                    "legibility": r.legibility,
                    "times_seen": r.times_seen,
                    "seen_in": r.seen_in,
                },
            }
            for r in records
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _esc(value):
    return html.escape(str(value)) if value is not None else ""


def _row(record):
    if not record.matched:
        badge = '<span class="badge bad">no match</span>'
    elif record.needs_review:
        badge = '<span class="badge warn">review ({:.0%})</span>'.format(record.match_score)
    else:
        badge = '<span class="badge good">{:.0%}</span>'.format(record.match_score)

    cover = (
        '<img src="{}" alt="" loading="lazy">'.format(_esc(record.cover_url))
        if record.cover_url
        else '<div class="nocover">no cover</div>'
    )

    meta = []
    if record.first_published:
        meta.append(_esc(record.first_published))
    if record.isbn:
        meta.append("ISBN " + _esc(record.isbn))
    if record.source:
        meta.append(_esc(record.source))
    if record.times_seen > 1:
        meta.append("seen {}x".format(record.times_seen))

    return """
    <tr class="{cls}">
      <td class="cover">{cover}</td>
      <td>
        <div class="title">{title}</div>
        <div class="author">{author}</div>
        <div class="meta">{meta}</div>
      </td>
      <td class="genre">{genre}</td>
      <td class="raw">
        <div class="spine">{spine}</div>
        <div class="asread">read as: {rawtitle}{rawauthor}</div>
        <div class="leg leg-{leg}">{leg}</div>
      </td>
      <td class="score">{badge}</td>
    </tr>
    """.format(
        cls="review" if record.needs_review else "",
        cover=cover,
        title=_esc(record.display_title),
        author=_esc(record.display_author),
        meta=" &middot; ".join(meta),
        genre=_esc(record.genre),
        spine=_esc(record.spine_text) or "<em>(none)</em>",
        rawtitle=_esc(record.raw_title),
        rawauthor=(" / " + _esc(record.raw_author)) if record.raw_author else "",
        leg=_esc(record.legibility),
        badge=badge,
    )


CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin:0; padding:32px; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
       background:#fbfaf8; color:#1b1a17; }
h1 { font-size:24px; margin:0 0 4px; letter-spacing:-0.01em; }
.sub { color:#6b6862; margin:0 0 24px; }
.stats { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:28px; }
.stat { background:#fff; border:1px solid #e6e2da; border-radius:10px;
        padding:12px 18px; min-width:132px; }
.stat b { display:block; font-size:24px; font-weight:600; letter-spacing:-0.02em; }
.stat span { font-size:12px; color:#6b6862; text-transform:uppercase;
             letter-spacing:0.06em; }
table { width:100%; border-collapse:collapse; background:#fff;
        border:1px solid #e6e2da; border-radius:10px; overflow:hidden; }
th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:0.07em;
     color:#6b6862; padding:10px 14px; border-bottom:1px solid #e6e2da;
     background:#f6f4f0; }
td { padding:12px 14px; border-bottom:1px solid #f0ece5; vertical-align:top; }
tr.review { background:#fffdf5; }
tr:last-child td { border-bottom:none; }
.cover { width:56px; }
.cover img { width:44px; border-radius:3px; display:block;
             box-shadow:0 1px 4px rgba(0,0,0,.18); }
.nocover { width:44px; height:64px; border:1px dashed #d5cec2; border-radius:3px;
           font-size:9px; color:#a49c8d; display:flex; align-items:center;
           justify-content:center; text-align:center; }
.title { font-weight:600; }
.author { color:#57534a; font-size:14px; }
.meta { color:#8d8778; font-size:12px; margin-top:3px; }
.genre { font-size:13px; color:#57534a; max-width:170px; }
.raw { font-size:12px; max-width:290px; }
.spine { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:11px;
         background:#f4f1ea; padding:4px 6px; border-radius:4px; word-break:break-word; }
.asread { color:#8d8778; margin-top:4px; }
.leg { display:inline-block; margin-top:4px; font-size:10px; text-transform:uppercase;
       letter-spacing:0.05em; }
.leg-clear { color:#2f7a3f; } .leg-partial { color:#a5751a; } .leg-guess { color:#b3452c; }
.badge { display:inline-block; padding:3px 8px; border-radius:20px; font-size:11px;
         font-weight:600; white-space:nowrap; }
.badge.good { background:#e3f3e6; color:#256b34; }
.badge.warn { background:#fdf0d8; color:#8a6412; }
.badge.bad  { background:#fbe4de; color:#9c3823; }
.note { margin-top:22px; color:#6b6862; font-size:13px; max-width:70ch; }
@media (prefers-color-scheme: dark) {
  body { background:#14130f; color:#eae6dd; }
  .stat, table { background:#1d1b16; border-color:#33302a; }
  th { background:#23211b; border-color:#33302a; color:#a49c8d; }
  td { border-color:#28251f; }
  tr.review { background:#231f14; }
  .spine { background:#26231c; }
  .author { color:#b8b2a5; } .genre { color:#b8b2a5; }
}
"""


def to_html(records, stats):
    review_count = sum(1 for r in records if r.needs_review)
    confident = len(records) - review_count

    cost = stats.get("cost_usd", 0.0)
    stat_cards = [
        ("books found", len(records)),
        ("auto-confirmed", confident),
        ("need review", review_count),
        ("tiles scanned", stats.get("tiles_processed", 0)),
        ("tiles suspect", stats.get("tiles_truncated", 0)),
        ("cost", "free" if not cost else "${:.3f}".format(cost)),
        ("seconds", "{:.0f}".format(stats.get("elapsed", 0.0))),
    ]
    cards = "".join(
        '<div class="stat"><b>{}</b><span>{}</span></div>'.format(_esc(value), _esc(label))
        for label, value in stat_cards
    )

    order = sorted(records, key=lambda r: (r.needs_review is False,
                                           (r.display_author or "").lower(),
                                           r.display_title.lower()))
    rows = "".join(_row(r) for r in order)

    hit_rate = (confident / len(records) * 100) if records else 0.0

    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shelf scan results</title><style>{css}</style></head>
<body>
<h1>Shelf scan results</h1>
<p class="sub">{n} books extracted by <strong>{provider}</strong> &middot;
{rate:.0f}% matched a catalogue record without needing review. Rows highlighted
in yellow are what a user would be asked to confirm or correct.</p>
<div class="stats">{cards}</div>
<table>
<thead><tr><th></th><th>Catalogue record</th><th>Genre</th>
<th>What the model actually saw</th><th>Match</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="note">The <em>What the model actually saw</em> column is the honesty
check: it shows the literal spine transcription before any interpretation, so a
plausible-but-wrong title is easy to spot. A book that reads cleanly but matches
nothing is usually a genuinely obscure edition rather than a mistake.</p>
</body></html>""".format(css=CSS, n=len(records), rate=hit_rate, cards=cards,
           rows=rows, provider=_esc(stats.get("provider", "unknown model")))
