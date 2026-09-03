"""Create a small batch of beta tester accounts via the Admin API.

For the staggered onboarding rollout (2-3 testers per wave, not all ~10-15 at
once) - see PROJECT_SUMMARY.md Sec10. Each tester gets a real account, a
random 6-digit PIN, and their real name set as `profiles.display_name`
immediately (via `handle_new_user()`'s user_metadata support) instead of the
email local-part they'd otherwise have to fix themselves on the Profile page.

    python create_testers.py --tester "Sarah Cohen <sarah@example.com>"
    python create_testers.py --tester "David Levy <david@example.com> @Anafa" \\
                             --tester "Rina Katz <rina@example.com>"

Prints one copy-paste-ready block per tester (email + PIN) to pair with
BETA_WELCOME.md when you message them - nothing is written to a file, since a
plaintext PIN list is not something that belongs sitting around on disk or in
git. Needs SUPABASE_URL / SUPABASE_SERVICE_KEY in .env, same as worker_run.py.
"""

import argparse
import re
import secrets
import sys

from pipeline.worker import SupabaseWorkerClient

_TESTER_RE = re.compile(r"^\s*(?P<name>.+?)\s*<(?P<email>[^>]+)>\s*(?:@(?P<neighbourhood>.+?))?\s*$")


def parse_tester(raw):
    """'Name <email>' or 'Name <email> @Neighbourhood' -> (name, email, neighbourhood|None)."""
    match = _TESTER_RE.match(raw)
    if not match:
        raise ValueError(
            "expected 'Name <email>' or 'Name <email> @Neighbourhood', got {!r}".format(raw))
    return match.group("name"), match.group("email"), match.group("neighbourhood")


def generate_pin():
    """A 6-digit numeric PIN - easy to type on a phone, meets Supabase Auth's
    6-character password minimum exactly, matches the 'PIN' framing already
    used on the login page and in BETA_WELCOME.md."""
    return "".join(secrets.choice("0123456789") for _ in range(6))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tester", action="append", required=True, dest="testers",
                        help="'Name <email>' or 'Name <email> @Neighbourhood'. "
                             "Repeat for multiple testers in one wave.")
    args = parser.parse_args()

    try:
        parsed = [parse_tester(raw) for raw in args.testers]
    except ValueError as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2

    client = SupabaseWorkerClient()
    created, failed = [], []

    for name, email, neighbourhood in parsed:
        pin = generate_pin()
        try:
            client.create_tester_account(email, pin, name, neighbourhood)
        except Exception as exc:
            failed.append((name, email, exc))
            continue
        created.append((name, email, pin))

    if created:
        print("Created {} account(s):\n".format(len(created)))
        for name, email, pin in created:
            print("  {} -- {}".format(name, email))
            print("    email: {}".format(email))
            print("    PIN:   {}".format(pin))
            print()
        print("Send each person their own email + PIN (pair with BETA_WELCOME.md) "
             "over WhatsApp or however you'd reach them - do not paste this whole "
             "block into a shared channel.", file=sys.stderr)

    if failed:
        print("\nFailed to create {} account(s):".format(len(failed)), file=sys.stderr)
        for name, email, exc in failed:
            print("  {} <{}>: {}".format(name, email, exc), file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
