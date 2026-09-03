"""Benchmark vision models against a human-checked shelf.

    python bench.py qwen2.5vl:7b minicpm-v:8b
    python bench.py qwen2.5vl:7b --ctx 8192 16384

Each model reads the same photo through the same pipeline, and is scored on the
three things that actually matter for cataloguing:

  recall   - how many real books it found        (misses annoy the user)
  invented - books reported that do not exist    (worst error; corrupts data)
  authors  - how often the attribution is right  (silent hallucination)

Extraction runs with catalogue lookup disabled, so the numbers reflect the model
rather than Open Library's uptime.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from compare import score

GROUND_TRUTH_IMAGE = "samples/WhatsApp Image 2026-08-23 at 5.12.18 PM.jpeg"


def run_one(model, ctx, image, keep_going=True):
    """Run the pipeline once with a given model and context size."""
    tag = "{}_ctx{}".format(model.replace(":", "-").replace("/", "-"), ctx)
    out = os.path.join("out", "bench_" + tag)

    env = dict(os.environ)
    env["OLLAMA_NUM_CTX"] = str(ctx)
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, "-m", "pipeline.run", image,
           "--provider", "ollama", "--model", model,
           "--out", out, "--language", "any", "--no-enrich",
           # Never let a checkpoint from an earlier benchmark leak into this
           # one - a resumed tile would score the OLD model's reading.
           "--no-resume"]

    started = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    elapsed = time.time() - started

    if proc.returncode != 0 or not os.path.exists(out + ".json"):
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
        message = " | ".join(tail) if tail else "no output"
        if not keep_going:
            raise RuntimeError(message)
        return {"model": model, "ctx": ctx, "failed": message}

    result = score(out + ".json")
    result["model"] = model
    result["ctx"] = ctx
    result["wall"] = elapsed
    # How many tiles reported a hard failure, as a robustness signal.
    with open(out + ".json", encoding="utf-8") as handle:
        result["tiles_failed"] = json.load(handle)["stats"].get("tiles_failed", 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Ollama model tags to compare.")
    parser.add_argument("--ctx", nargs="+", type=int, default=[8192],
                        help="Context sizes to try for each model.")
    parser.add_argument("--image", default=GROUND_TRUTH_IMAGE,
                        help="Photo to benchmark against (must match realshelf.py).")
    args = parser.parse_args()

    results = []
    total = len(args.models) * len(args.ctx)
    n = 0
    for model in args.models:
        for ctx in args.ctx:
            n += 1
            print("[{}/{}] {} @ ctx={} ...".format(n, total, model, ctx),
                  flush=True)
            result = run_one(model, ctx, args.image)
            if "failed" in result:
                print("      FAILED: {}".format(result["failed"]), flush=True)
            else:
                print("      found {}/{}  invented {}  in {:.0f}s".format(
                    result["found"], result["expected"], result["invented"],
                    result["wall"]), flush=True)
            results.append(result)

    ok = [r for r in results if "failed" not in r]
    if not ok:
        print("\nEvery run failed.")
        return 1

    ok.sort(key=lambda r: (-r["found"], r["invented"], r["wall"]))

    print()
    print("=" * 92)
    print("{:<22} {:>6} {:>7} {:>9} {:>9} {:>8} {:>8} {:>7}".format(
        "MODEL", "CTX", "FOUND", "RECALL", "INVENTED", "AUTHORS", "FAILED", "SECS"))
    print("-" * 92)
    for r in ok:
        print("{:<22} {:>6} {:>3}/{:<3} {:>8.0f}% {:>9} {:>7}/{:<2} {:>7} {:>7.0f}".format(
            r["model"][:21], r["ctx"], r["found"], r["expected"],
            r["recall"] * 100, r["invented"],
            r["right_author"], r["found"], r["tiles_failed"], r["wall"]))
    print("=" * 92)

    best = ok[0]
    print()
    print("Best recall: {} @ ctx={} - found {}/{} ({:.0f}%), invented {}.".format(
        best["model"], best["ctx"], best["found"], best["expected"],
        best["recall"] * 100, best["invented"]))
    if best["invented"]:
        print("  invented: {}".format(", ".join(best["invented_titles"][:5])))
    if best["miss_titles"]:
        print("  missed  : {}".format(", ".join(best["miss_titles"][:6])))

    with open("out/bench_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print()
    print("Full results: out/bench_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
