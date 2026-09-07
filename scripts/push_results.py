#!/usr/bin/env python3
"""Commit and push run artefacts from a throwaway machine, without leaking the token.

Written for the Kaggle notebook, which runs on a container that disappears when
the session ends. Anything not pushed is lost.

    GITHUB_TOKEN=... python scripts/push_results.py \
        --paths results/scaling/kaggle --branch kaggle-results

Handling of the credential is the whole point of this script existing rather
than being three lines of shell in a notebook cell:

* The token is read from the environment, never from an argument. A command line
  is visible in process listings and lands in shell history.
* It is passed to git through a credential helper on stdin, so it is never
  written into ``.git/config`` and never appears in a remote URL. A token in a
  remote URL shows up in ``git remote -v``, in error messages, and in the
  notebook output cell that a public Kaggle notebook publishes.
* Every subprocess's output is scrubbed before it is printed, so a token echoed
  back inside a git error message does not reach the log either.

Only the paths given are committed. Checkpoints are large and already ignored;
this pushes results, not weights.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[1]
# Committed as the repository's owner, per the project's convention.
GIT_NAME = "abyyworld"
GIT_EMAIL = "annolieberto@gmail.com"

# Anything shaped like a GitHub token, whatever the surrounding text.
TOKEN_PATTERN = re.compile(r"gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{20,}")


def redact(text: str, secret: str | None = None) -> str:
    """Remove any credential from text before it is printed.

    Both the specific secret and anything token-shaped, because git error
    messages sometimes echo a rewritten URL rather than the literal input.
    """
    if not text:
        return text
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return TOKEN_PATTERN.sub("[REDACTED]", text)


def run(args: list[str], secret: str | None = None, repo: Path | None = None,
        **kwargs) -> subprocess.CompletedProcess:
    """Run a git command, scrubbing its output whether it succeeds or fails."""
    done = subprocess.run(args, cwd=repo or DEFAULT_REPO, capture_output=True,
                          text=True, **kwargs)
    if done.returncode != 0:
        raise SystemExit(
            f"command failed: {' '.join(args[:2])}\n"
            f"{redact(done.stderr or done.stdout, secret)}")
    return done


def credential_helper() -> str:
    """A helper that answers from the environment.

    Keeps the token out of ``.git/config``: git invokes this, the shell function
    reads ``GITHUB_TOKEN`` from the environment it inherits, and nothing is
    persisted to disk.
    """
    return ('!f() { echo "username=x-access-token"; echo "password=$GITHUB_TOKEN"; }; f')


def push(paths: list[str], branch: str, token: str, repo: Path | None = None,
         message: str | None = None, remote: str | None = None) -> list[str]:
    """Commit ``paths`` on ``branch`` and push. Returns the files committed.

    Split out of ``main`` so it can be exercised against a throwaway repository
    rather than only against this one. The mechanism a notebook depends on to get
    results off a machine that is about to disappear should not be tested only by
    being used for real.
    """
    repo = Path(repo) if repo else DEFAULT_REPO

    missing = [p for p in paths if not (repo / p).exists()]
    if missing:
        raise SystemExit(f"nothing to push, these paths do not exist: {missing}")

    run(["git", "config", "user.name", GIT_NAME], repo=repo)
    run(["git", "config", "user.email", GIT_EMAIL], repo=repo)
    run(["git", "config", "credential.helper", credential_helper()], repo=repo)
    if remote:
        run(["git", "remote", "set-url", "origin",
             f"https://github.com/{remote}.git"], repo=repo)

    run(["git", "checkout", "-B", branch], token, repo=repo)
    run(["git", "add", "--", *paths], token, repo=repo)

    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo,
                            capture_output=True, text=True).stdout.split()
    if not staged:
        return []

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run(["git", "commit", "-m", message or f"Add run artefacts from Kaggle, {stamp}"],
        token, repo=repo)
    run(["git", "push", "-u", "origin", branch], token, repo=repo,
        env={**os.environ, "GITHUB_TOKEN": token})
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", nargs="+", default=["results"],
                        help="paths to commit, relative to the repository root")
    parser.add_argument("--branch", default="kaggle-results",
                        help="branch to push to. Deliberately not the default branch: "
                             "a notebook should propose results, not overwrite history")
    parser.add_argument("--message", default=None)
    parser.add_argument("--remote", default=None,
                        help="owner/repo. Defaults to the existing origin")
    parser.add_argument("--repo", default=None,
                        help="repository to operate on. Defaults to this checkout")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "GITHUB_TOKEN is not set.\n"
            "On Kaggle: Add-ons, Secrets, add GITHUB_TOKEN, then attach it to the "
            "notebook. Do not paste the token into a cell: a public notebook "
            "publishes its own source.")

    staged = push(args.paths, args.branch, token, repo=args.repo,
                  message=args.message, remote=args.remote)
    if not staged:
        print("nothing changed, so nothing pushed")
        return 0

    print(f"pushed {len(staged)} files to '{args.branch}':")
    for name in staged[:20]:
        print(f"  {name}")
    if len(staged) > 20:
        print(f"  ... and {len(staged) - 20} more")
    print("\nOpen a pull request from that branch when you are happy with it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
