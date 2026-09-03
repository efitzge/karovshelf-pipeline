"""The manual-ingestion path: Elliot runs a batch through Claude/ChatGPT by hand.

This exists because the automated worker (Ollama, or later a paid API) is not
always the right tool - some shelves (folder-heavy, very dense, poor lighting)
are cheaper to hand to a frontier model's own chat interface than to debug.
Rather than treat that as a special case bolted onto the pipeline, it is built
as its own module with exactly two responsibilities:

  1. `build_manual_packet()` - produce what Elliot needs to paste into
     Claude.ai or ChatGPT: the instructions and which photos to attach.
  2. `ingest_manual_reply()` - take whatever comes back and feed it through
     the SAME `process_reply()` the automated providers use, so dedupe,
     catalogue matching, and the cache quality bar apply identically
     regardless of which "provider" actually read the spines.

Nothing downstream of ingestion needs to know a human was involved.
"""

import copy
import json
from dataclasses import dataclass

from .providers import BOOK_SCHEMA, SYSTEM_PROMPT, process_reply

# Unlike the tiled per-provider prompt, a manual submission is the whole batch
# in one go - the photos are not pre-cut into tiles (a human pasting into a
# chat window wants to attach 3-10 original photos once, not thirty tile
# crops), and the same physical book may appear in more than one photo if they
# overlap. The model is asked to de-duplicate itself; the pipeline's own
# dedupe.py still runs afterward as a safety net, not the primary mechanism.
#
# `photo_index` (0-based, matching the attach order below) is the one thing
# this prompt needs beyond the shared BOOK_SCHEMA: without it there is no way
# to link a book back to the single photo it came from, so the review UI can
# only ever show "here are all N photos from that batch" instead of the one
# relevant image. See MANUAL_BOOK_SCHEMA below.
MANUAL_USER_PROMPT = (
    "The attached photos all show bookshelves belonging to ONE person. There "
    "may be several photos, and some may overlap or show the same shelf "
    "section from a different angle - if you can tell a book already appears "
    "in an earlier photo, list it once (using the photo it's most clearly "
    "visible in), not once per photo.\n\n"
    "Work through the photos in the order given, left to right within each "
    "one. For every book, set `photo_index` to which attached photo it came "
    "from, counting from 0 in the order the photos are listed below. Return "
    "every book you find as a single combined list, in the exact JSON shape "
    "described by the schema - no prose outside the JSON."
)

# BOOK_SCHEMA plus the one manual-only field. Copied rather than mutated in
# place so the automated per-tile providers (where photo_index would be
# redundant - a tile is inherently one known photo, tracked via Tile.label
# instead) are never affected by this.
MANUAL_BOOK_SCHEMA = copy.deepcopy(BOOK_SCHEMA)
MANUAL_BOOK_SCHEMA["properties"]["books"]["items"]["properties"]["photo_index"] = {
    "type": "integer",
    "description": (
        "0-based index of which attached photo (in the order they were "
        "attached) this book was found in."
    ),
}
MANUAL_BOOK_SCHEMA["properties"]["books"]["items"]["required"].append("photo_index")


@dataclass
class ManualPacket:
    """Everything needed to run one batch through a chat UI by hand."""

    batch_id: str
    photo_paths: list
    instructions: str

    def render(self):
        """A single copy-paste-ready block: instructions, schema, photo list.

        Paste this text into Claude.ai or ChatGPT, attach the listed photos to
        the same message, and send it. Copy the reply back in full - prose
        wrapper around the JSON is fine, `ingest_manual_reply` tolerates it.
        """
        photo_list = "\n".join("  - {}".format(p) for p in self.photo_paths)
        return (
            "=== SYSTEM CONTEXT (paste this first, or as the system prompt "
            "if your chat tool supports one) ===\n"
            + SYSTEM_PROMPT
            + "\n\n=== INSTRUCTIONS ===\n"
            + self.instructions
            + "\n\n=== ATTACH THESE PHOTOS TO THE SAME MESSAGE ===\n"
            + photo_list
            + "\n\n=== REQUIRED JSON SHAPE (return exactly this schema) ===\n"
            + json.dumps(MANUAL_BOOK_SCHEMA, indent=2)
            + "\n"
        )


def build_manual_packet(batch_id, photo_paths):
    """Prepare one scan_batch for manual processing.

    `photo_paths` should be the batch's photos already downloaded locally (or
    signed URLs, if Elliot's chat tool of choice can fetch them directly) -
    this function only assembles the text side, it does not fetch anything.
    """
    if not photo_paths:
        raise ValueError("a manual packet needs at least one photo")
    return ManualPacket(batch_id=batch_id, photo_paths=list(photo_paths),
                        instructions=MANUAL_USER_PROMPT)


def ingest_manual_reply(raw_text):
    """Turn whatever Elliot pastes back into the same (books, truncated) shape
    every automated provider produces.

    Reuses `process_reply` unchanged - the repetition-loop guard is nearly
    irrelevant here (a human is not going to get stuck in a 200x loop), but
    costs nothing to leave active, and the empty-title / fragment handling is
    just as relevant to a human's typos as to a model's.
    """
    return process_reply(raw_text)


def books_to_extracted(books, batch_label):
    """Adapt manual-path book dicts into the pipeline's ExtractedBook shape,
    so the rest of extract.py/enrich.py (dedupe, cache, catalogue match)
    runs completely unaware this batch was not read by a model.

    Reuses the same `source`/`seen_in` fields the automated tiler uses for
    tile labels ("photo00.jpeg [r1c1]"), just in the simpler "photo00" shape
    - `worker.resolve_source_photo()` parses either shape identically to map
    a reading back to its actual Storage path. Falls back to `batch_label`
    when a reply predates `photo_index` (old-format pasted replies), which
    just means "View shelf photo" shows the whole batch instead of one photo.
    """
    from .extract import ExtractedBook  # local import avoids a cycle at load time

    extracted = []
    for item in books:
        photo_index = item.get("photo_index")
        source = "photo{:02d}".format(photo_index) if photo_index is not None else batch_label
        extracted.append(ExtractedBook(
            spine_text=item["spine_text"],
            title=item["title"],
            author=item["author"],
            legibility=item["legibility"],
            orientation=item["orientation"],
            source=source,
            seen_in=[source],
        ))
    return extracted
