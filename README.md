# KarovShelf shelf-scan pipeline

Photos of a bookshelf in, a catalogue out. This Python pipeline reads book spines from ordinary phone photos with a vision model, matches each book against Open Library and Google Books, and produces a reviewable catalogue that feeds the [KarovShelf](https://github.com/efitzge/karovshelf) book-sharing app. I built it because nobody will type in 300 books to share them with neighbours, but most people will take three photos of a shelf. It also runs the same job on a local model, so I can compare a paid hosted model against a free one on my own GPU using identical inputs and prompts.

<!-- Screenshot placeholder: save docs/screenshot-report.png, then replace this comment with:
![Screenshot: scan report showing each detected book next to its raw spine text](docs/screenshot-report.png) -->

**Live app it feeds:** [karovshelf.vercel.app](https://karovshelf.vercel.app) (invite-only beta)

## Stack

- Python 3, Pillow, requests
- Claude API (vision, structured output) for the hosted backend
- Ollama with Qwen2.5-VL for the local backend
- Open Library and Google Books APIs for catalogue matching
- Supabase (Postgres + Storage) to pick up uploaded batches and write results back
- Built with Claude Code

## How it works

```
photos ──► tile ──► vision model ──► dedupe ──► catalogue match ──► report
```

1. **Tile.** A full-bookcase photo is about 4000px wide, but vision input works at about 1568px on the long edge. The pipeline cuts each photo into an overlapping grid so spine text keeps its resolution. An 18% overlap means a book on a seam appears whole in at least one tile.
2. **Read.** Each tile goes to the model with a fixed output schema. The model returns the literal spine text and its best reading of title and author, so misreads are easy to spot.
3. **Dedupe.** Books seen in two overlapping tiles are merged on a fuzzy title match, keeping the most legible reading.
4. **Match.** Each reading is looked up in Open Library, then Google Books, to get a real title, author, ISBN, and cover. A reading that matches nothing goes to a review queue.
5. **Report.** The pipeline writes the catalogue as JSON and an HTML report that shows every book next to its raw transcription. Nothing becomes searchable in the app until the owner confirms it.

A typical phone photo costs about $0.45 to $0.70 to read with the hosted model, or roughly $3 to $5 for a 300-book collection.

## Run it

```bash
python -m pip install -r requirements.txt
cp .env.example .env    # add your own keys

# Local model (no API cost, needs ~8GB VRAM)
ollama pull qwen2.5vl:7b
python -m pipeline.run samples/ --provider ollama --out out/local

# Hosted model
python -m pipeline.run samples/ --provider claude --out out/claude
```

`python selftest.py` checks the non-AI parts without calling any model.

More detail on options, costs and internals: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
