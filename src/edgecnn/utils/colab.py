"""Colab fallback plumbing.

Owner: Member 1.

Notebooks run locally. When a laptop is too slow for a run, a member opens the
same notebook in Colab: cell 1 clones and installs the repository there, and
the notebook's last cell calls :func:`push_results` to send the run's results
back to GitHub before the Colab session is wiped. The full procedure is in
``notebooks/README.md``.

Token handling - the rule that matters most in this file
--------------------------------------------------------
The repository is public and notebooks are committed WITH their outputs, so
anything a cell prints is published. The GitHub token therefore:

* is read from Colab Secrets, never typed into a cell;
* reaches git only through an environment variable read by an inline
  credential helper - never inside a URL or a command-line argument;
* is scrubbed from any git output before that output is printed.

``tests/test_colab.py`` checks all three with git mocked out.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from edgecnn.contracts.paths import REPO_ROOT

#: Environment variable the credential helper reads the token from.
TOKEN_ENV = "GH_TOKEN"

#: Inline helper: git runs it through a shell, which expands $GH_TOKEN from the
#: environment - so the token never appears in git's argv or in a remote URL.
CREDENTIAL_HELPER = "!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f"


def in_colab() -> bool:
    """True inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def colab_secret(name: str) -> str | None:
    """Read a Colab Secret (key icon in the left sidebar); None if unavailable."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        return userdata.get(name)
    except Exception:  # not in Colab, secret missing, or access not granted
        return None


def push_results(
    paths: Iterable[str | Path],
    message: str,
    *,
    branch: str | None = None,
    token: str | None = None,
    name: str | None = None,
    email: str | None = None,
    repo_root: Path = REPO_ROOT,
    runner: Callable[..., Any] = subprocess.run,
) -> bool:
    """Commit ``paths`` in the Colab clone and push them to GitHub.

    Call it from a notebook's last cell, for official runs only::

        if IN_COLAB and cfg.is_official:
            push_results([paths.metrics_dir(cfg.run_id), paths.figures_dir(cfg.run_id)],
                         "Add model_a official results (Colab)")

    Then save the executed notebook itself with File > Save a copy in GitHub.

    Args:
        paths: Files or directories to commit, absolute or repo-relative.
        message: Commit message - short, as for any commit.
        branch: Defaults to the clone's current branch.
        token, name, email: Default to the Colab Secrets ``GH_TOKEN``,
            ``GIT_NAME`` and ``GIT_EMAIL``. The commit is attributed to you.
        runner: Injected for tests; defaults to :func:`subprocess.run`.

    Returns:
        True if a commit was pushed, False if there was nothing new to commit.

    Raises:
        RuntimeError: if the token or identity is missing, or a git step fails.
            The message never contains the token.
    """
    token = token or colab_secret("GH_TOKEN")
    if not token:
        raise RuntimeError(
            "No GitHub token. Add GH_TOKEN to Colab Secrets (key icon, left sidebar) "
            "and allow this notebook to access it - see notebooks/README.md."
        )
    name = name or colab_secret("GIT_NAME")
    email = email or colab_secret("GIT_EMAIL")
    if not name or not email:
        raise RuntimeError(
            "Add GIT_NAME and GIT_EMAIL to Colab Secrets so the commit is attributed to you."
        )

    network_env = {**os.environ, TOKEN_ENV: token}
    relative = [_repo_relative(path, repo_root) for path in paths]

    def git(*args: str, network: bool = False, show: bool = False, check: bool = True) -> Any:
        command = ["git", "-C", str(repo_root)]
        if network:
            # Reset any configured helper, then use ours. The token itself is
            # only in network_env, and only for commands that talk to GitHub.
            command += ["-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}"]
        command += list(args)
        proc = runner(
            command,
            env=network_env if network else None,
            capture_output=True,
            text=True,
        )
        output = _scrub(f"{proc.stdout or ''}{proc.stderr or ''}", token).strip()
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed (exit {proc.returncode}):\n{output}")
        if show and output:
            print(output)
        return proc

    branch = branch or git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    git("config", "user.name", name)
    git("config", "user.email", email)
    git("add", "-A", "--", *relative)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Nothing new to commit.")
        return False

    git("commit", "-m", message, show=True)
    git("pull", "--rebase", "origin", branch, network=True, show=True)
    git("push", "origin", f"HEAD:{branch}", network=True, show=True)
    print(
        f"Results pushed to '{branch}'.\n"
        "Now save the notebook itself: File > Save a copy in GitHub, "
        "same branch and same path, so its outputs land too."
    )
    return True


def _repo_relative(path: str | Path, repo_root: Path) -> str:
    """Repo-relative POSIX path, so nothing machine-specific reaches git or output."""
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.resolve().relative_to(repo_root.resolve())
    return candidate.as_posix()


def _scrub(text: str, secret: str) -> str:
    """Remove a secret from text before it can be printed into a committed notebook."""
    return text.replace(secret, "***") if secret else text
