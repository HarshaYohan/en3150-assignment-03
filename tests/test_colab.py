"""push_results: the right git commands, and the token never leaks.

git is replaced by a recording fake, so these run anywhere - no Colab, no
network, no repository changes. The real Colab path is verified once by
Member 1 in Phase 0 (see COLAB.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from edgecnn.contracts import paths
from edgecnn.utils.colab import TOKEN_ENV, in_colab, push_results

TOKEN = "github_pat_TESTTOKEN0123456789abcdefghij"


@dataclass
class FakeProcess:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeGit:
    """Records every call; `diff --cached --quiet` reports staged changes by default."""

    def __init__(self, staged: bool = True, echo_token: bool = False) -> None:
        self.calls: list[tuple[list[str], dict | None]] = []
        self.staged = staged
        self.echo_token = echo_token

    def __call__(self, command, env=None, capture_output=True, text=True):
        self.calls.append((list(command), env))
        if "rev-parse" in command:
            return FakeProcess(stdout="main\n")
        if command[-2:] == ["--cached", "--quiet"]:
            return FakeProcess(returncode=1 if self.staged else 0)
        if self.echo_token and "push" in command:
            return FakeProcess(stderr=f"remote: using credentials {TOKEN}\n")
        return FakeProcess()

    def subcommands(self) -> list[str]:
        out = []
        for command, _ in self.calls:
            rest = [part for part in command[3:] if not part.startswith("credential.helper")]
            rest = [part for part in rest if part != "-c"]
            out.append(rest[0])
        return out


def _push(fake: FakeGit, **kwargs) -> bool:
    return push_results(
        [paths.metrics_dir("model_a__adam__seed42"), "results/figures/model_a__adam__seed42"],
        "Add model_a results (Colab)",
        token=TOKEN,
        name="Member Three",
        email="m3@example.com",
        runner=fake,
        **kwargs,
    )


def test_issues_the_expected_git_sequence() -> None:
    fake = FakeGit()
    assert _push(fake) is True
    assert fake.subcommands() == [
        "rev-parse", "config", "config", "add", "diff", "commit", "pull", "push",
    ]


def test_paths_are_passed_repo_relative() -> None:
    fake = FakeGit()
    _push(fake)
    add = next(command for command, _ in fake.calls if "add" in command)
    assert "results/metrics/model_a__adam__seed42" in add
    assert not any(":\\" in part or part.startswith("/") for part in add[add.index("--") + 1:])


def test_token_never_appears_in_any_command() -> None:
    fake = FakeGit()
    _push(fake)
    for command, _ in fake.calls:
        assert TOKEN not in " ".join(command)


def test_token_is_only_given_to_commands_that_reach_github() -> None:
    fake = FakeGit()
    _push(fake)
    for command, env in fake.calls:
        talks_to_github = "pull" in command or "push" in command
        if talks_to_github:
            assert env is not None and env[TOKEN_ENV] == TOKEN
        else:
            assert env is None or TOKEN_ENV not in env


def test_token_is_scrubbed_from_printed_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Notebook outputs are committed to a public repo."""
    _push(FakeGit(echo_token=True))
    printed = capsys.readouterr().out
    assert TOKEN not in printed
    assert "***" in printed


def test_nothing_staged_means_no_commit_and_no_network() -> None:
    fake = FakeGit(staged=False)
    assert _push(fake) is False
    assert "commit" not in fake.subcommands()
    assert "push" not in fake.subcommands()


def test_missing_token_fails_with_a_helpful_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("edgecnn.utils.colab.colab_secret", lambda name: None)
    with pytest.raises(RuntimeError, match="GH_TOKEN"):
        push_results(["results"], "msg", runner=FakeGit())


def test_pull_autostashes_other_edits() -> None:
    """A stray edit in the Colab copy must not block pushing the results."""
    fake = FakeGit()
    _push(fake)
    pull = next(command for command, _ in fake.calls if "pull" in command)
    assert "--rebase" in pull and "--autostash" in pull


def test_in_colab_is_false_on_this_machine() -> None:
    """Checks the installed package, so it is False locally and True on any Colab runtime."""
    assert in_colab() is False
