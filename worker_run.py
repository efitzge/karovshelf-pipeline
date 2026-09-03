"""Run the extraction worker: poll Supabase for pending scan_batches.

    python worker_run.py --once           # process one pending batch, then exit
    python worker_run.py                  # poll forever, every --interval seconds
    python worker_run.py --once --provider claude   # one-off hosted run

Needs SUPABASE_URL / SUPABASE_SERVICE_KEY in .env. Defaults to the free local
Ollama backend; --provider claude uses the hosted Anthropic API instead (needs
ANTHROPIC_API_KEY in .env, and costs money per token).
"""

import argparse
import os
import sys
import time

from pipeline.worker import run_once


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true",
                        help="Process at most one pending batch, then exit.")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds to wait between polls when not --once "
                             "(default: %(default)s).")
    parser.add_argument("--language", default="any",
                        help="Catalogue language preference (default: any, "
                             "since shelves may include non-English books).")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "claude"],
                        help="Vision backend (default: %(default)s). 'claude' is "
                             "hosted and billed per token - needs ANTHROPIC_API_KEY.")
    args = parser.parse_args()

    if args.provider == "claude" and not (os.environ.get("ANTHROPIC_API_KEY")
                                          or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("ANTHROPIC_API_KEY is not set, and --provider claude needs it.")
        return 2

    if args.once:
        did_work = run_once(language=args.language, provider_name=args.provider)
        if not did_work:
            print("No pending batches.")
        return 0

    print("Polling every {:.0f}s. Ctrl+C to stop.".format(args.interval))
    try:
        while True:
            did_work = run_once(language=args.language, provider_name=args.provider)
            if not did_work:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
