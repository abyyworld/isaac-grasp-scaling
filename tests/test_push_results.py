"""Credential handling for the Kaggle push.

The failure this guards against is specific and permanent: a token printed into
a public Kaggle notebook's output cell, or written into .git/config, is a token
published. Revoking it afterwards does not un-publish it.

The tokens below are fabricated. Nothing in this repository should ever contain
a real one, which is what the last test checks.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "push_results", ROOT / "scripts" / "push_results.py")
push_results = importlib.util.module_from_spec(spec)
sys.modules["push_results"] = push_results
spec.loader.exec_module(push_results)

# Fabricated, in the two shapes GitHub issues.
FAKE_CLASSIC = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FAKE_FINE_GRAINED = "github_pat_" + "11ABCDEFG0abcdefghijklm_" + "N0pQrStUvWxYz1234567890abcdefghijklmno"


@pytest.mark.parametrize("token", [FAKE_CLASSIC, FAKE_FINE_GRAINED])
def test_the_exact_secret_is_removed(token):
    text = f"fatal: could not read from https://x-access-token:{token}@github.com/o/r.git"
    scrubbed = push_results.redact(text, token)
    assert token not in scrubbed
    assert "[REDACTED]" in scrubbed


@pytest.mark.parametrize("token", [FAKE_CLASSIC, FAKE_FINE_GRAINED])
def test_a_token_is_removed_even_when_it_was_not_the_one_passed(token):
    """git sometimes echoes a rewritten URL, so shape matching is the backstop."""
    scrubbed = push_results.redact(f"remote: rejected, token {token} lacks scope")
    assert token not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_redaction_leaves_ordinary_text_alone():
    text = "Everything up-to-date\nbranch 'kaggle-results' set up to track origin."
    assert push_results.redact(text) == text
    assert push_results.redact("") == ""
    assert push_results.redact(None) is None


def test_the_credential_helper_does_not_embed_the_secret():
    """The helper must read from the environment, not carry the value.

    git config values are written to .git/config in plain text, so a helper that
    interpolated the token would persist it to disk.
    """
    helper = push_results.credential_helper()
    assert "$GITHUB_TOKEN" in helper
    assert "ghp_" not in helper and "github_pat_" not in helper


def test_it_refuses_to_run_without_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["push_results.py"])
    with pytest.raises(SystemExit) as excinfo:
        push_results.main()
    message = str(excinfo.value)
    assert "GITHUB_TOKEN is not set" in message
    assert "Do not paste the token into a cell" in message


def test_no_real_credential_is_committed_anywhere():
    """A repository-wide sweep for anything token-shaped in tracked files."""
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                             text=True, check=True).stdout.split()
    offenders = []
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path.suffix in {".png", ".gif", ".pt", ".npy"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for match in push_results.TOKEN_PATTERN.finditer(text):
            # The fabricated tokens in this file are assembled from fragments so
            # they never appear literally in the source, but guard anyway.
            if name != "tests/test_push_results.py":
                offenders.append(f"{name}: {match.group()[:12]}...")
    assert not offenders, "credential-shaped strings in tracked files:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# The push itself, against a throwaway repository.
# --------------------------------------------------------------------------- #


@pytest.fixture
def throwaway(tmp_path):
    """A bare 'remote' and a clone of it, standing in for GitHub."""
    remote = tmp_path / "remote.git"
    clone = tmp_path / "clone"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True)
    for name, value in (("user.name", "test"), ("user.email", "test@example.com")):
        subprocess.run(["git", "-C", str(clone), "config", name, value], check=True)
    (clone / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(clone), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-qm", "seed"], check=True)
    subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "main"], check=True)
    return remote, clone


def _branches(remote):
    out = subprocess.run(["git", "-C", str(remote), "branch", "--format=%(refname:short)"],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def test_results_reach_the_remote_on_their_own_branch(throwaway):
    remote, clone = throwaway
    results = clone / "results" / "scaling" / "kaggle"
    results.mkdir(parents=True)
    (results / "scaling.csv").write_text("train_samples,unseen_rate\n6144,0.484\n")
    (results / "scaling.json").write_text('{"points": []}')

    staged = push_results.push(["results/scaling/kaggle"], "kaggle-results",
                               FAKE_FINE_GRAINED, repo=clone, message="results")

    assert sorted(staged) == ["results/scaling/kaggle/scaling.csv",
                              "results/scaling/kaggle/scaling.json"]
    assert "kaggle-results" in _branches(remote)
    assert "main" in _branches(remote), "the default branch must be left alone"


def test_the_default_branch_is_not_touched(throwaway):
    """A notebook proposes results; it does not rewrite history."""
    remote, clone = throwaway
    before = subprocess.run(["git", "-C", str(remote), "rev-parse", "main"],
                            capture_output=True, text=True, check=True).stdout
    (clone / "results").mkdir()
    (clone / "results" / "out.json").write_text("{}")
    push_results.push(["results"], "kaggle-results", FAKE_CLASSIC, repo=clone)
    after = subprocess.run(["git", "-C", str(remote), "rev-parse", "main"],
                           capture_output=True, text=True, check=True).stdout
    assert before == after


def test_the_token_is_not_written_into_git_config(throwaway):
    """A token in .git/config survives the session and shows in git remote -v."""
    _remote, clone = throwaway
    (clone / "results").mkdir()
    (clone / "results" / "out.json").write_text("{}")
    push_results.push(["results"], "kaggle-results", FAKE_FINE_GRAINED, repo=clone)

    config = (clone / ".git" / "config").read_text()
    assert FAKE_FINE_GRAINED not in config
    assert "github_pat_" not in config
    assert "$GITHUB_TOKEN" in config, "the helper reads the environment instead"


def test_nothing_to_push_is_not_an_error(throwaway):
    _remote, clone = throwaway
    (clone / "results").mkdir()
    (clone / "results" / "out.json").write_text("{}")
    push_results.push(["results"], "kaggle-results", FAKE_CLASSIC, repo=clone)
    again = push_results.push(["results"], "kaggle-results", FAKE_CLASSIC, repo=clone)
    assert again == []


def test_a_missing_results_path_fails_before_touching_git(throwaway):
    _remote, clone = throwaway
    with pytest.raises(SystemExit, match="do not exist"):
        push_results.push(["results/never-made"], "kaggle-results", FAKE_CLASSIC,
                          repo=clone)


def test_the_commit_is_attributed_to_the_repository_owner(throwaway):
    _remote, clone = throwaway
    (clone / "results").mkdir()
    (clone / "results" / "out.json").write_text("{}")
    push_results.push(["results"], "kaggle-results", FAKE_CLASSIC, repo=clone)
    author = subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%an <%ae>"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert author == f"{push_results.GIT_NAME} <{push_results.GIT_EMAIL}>"
