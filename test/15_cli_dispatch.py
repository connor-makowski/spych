"""
Test 15: CLI dispatch smoke tests.

Runs `spych.cli.main()` for every subcommand with real argument parsing and
real object construction, but with the actual blocking work (wake-word
listening, live transcription, voice recording) replaced with no-ops. This
exercises `spych/cli.py`'s own dispatch code -- kwarg building, `args.*`
attribute access, local imports -- which the other test files never touch,
since they call the underlying responder/factory classes directly instead of
going through the CLI. That gap is exactly how bugs like a missing `import
os` or an `args.verbose` read with no matching `--verbose` flag survived
undetected: they only raised once a specific dispatch branch actually ran.

Run via: pytest test/15_cli_dispatch.py or nox
"""

import sys

import pytest

from spych.cli import main


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["spych"] + argv)
    main()


@pytest.fixture(autouse=True)
def _no_orchestrator_start(monkeypatch):
    """Prevent SpychOrchestrator.start() from blocking on the wake-word loop."""
    from spych.orchestrator import SpychOrchestrator

    monkeypatch.setattr(SpychOrchestrator, "start", lambda self: None)


def test_cli_ollama_dispatch(monkeypatch):
    _run(
        monkeypatch,
        [
            "ollama",
            "--model",
            "llama3.2:latest",
            "--use-speaker",
            "false",
            "--verbose",
        ],
    )


def test_cli_claude_code_cli_dispatch(monkeypatch):
    _run(
        monkeypatch,
        ["claude_code_cli", "--use-speaker", "false", "--verbose"],
    )


def test_cli_claude_code_sdk_dispatch(monkeypatch):
    _run(
        monkeypatch,
        ["claude_code_sdk", "--use-speaker", "false", "--verbose"],
    )


def test_cli_codex_cli_dispatch(monkeypatch):
    _run(monkeypatch, ["codex_cli", "--use-speaker", "false", "--verbose"])


def test_cli_antigravity_cli_dispatch(monkeypatch):
    _run(
        monkeypatch,
        ["antigravity_cli", "--use-speaker", "false", "--verbose"],
    )


def test_cli_opencode_cli_dispatch(monkeypatch):
    _run(monkeypatch, ["opencode_cli", "--use-speaker", "false", "--verbose"])


def test_cli_multi_dispatch(monkeypatch):
    _run(
        monkeypatch,
        [
            "multi",
            "--agents",
            "ollama",
            "--ollama-model",
            "llama3.2:latest",
            "--use-speaker",
            "false",
            "--verbose",
        ],
    )


def test_cli_live_dispatch(monkeypatch):
    from spych.live import SpychLive

    monkeypatch.setattr(SpychLive, "start", lambda self: None)
    _run(monkeypatch, ["live", "--whisper-model", "tiny.en"])


def test_cli_live_translation_dispatch(monkeypatch):
    from spych.live_translation import SpychLiveTranslation

    monkeypatch.setattr(SpychLiveTranslation, "start", lambda self: None)
    _run(
        monkeypatch,
        ["live-translation", "--languages", "en", "es", "--no-speaker"],
    )


def test_cli_profile_my_voice_dispatch(monkeypatch):
    import spych.voice_manager

    called = {}

    def fake_profile_my_voice(
        name, device_index=-1, alternate_output_file=None
    ):
        called["name"] = name

    monkeypatch.setattr(
        spych.voice_manager, "profile_my_voice", fake_profile_my_voice
    )
    _run(monkeypatch, ["profile_my_voice", "--name", "cli_test_profile"])
    assert called["name"] == "cli_test_profile"


def test_cli_users_dispatch(monkeypatch, tmp_path):
    import spych.utils

    def fake_cache_dir(folder="voices"):
        path = tmp_path / folder
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr(spych.utils, "get_cache_dir", fake_cache_dir)

    # Walk every menu branch: create, edit, set default, set theme, delete,
    # clear default, exit. This is a direct regression test for a bug where
    # `os` and `get_cache_dir` were used in the "users" dispatch branch
    # without being imported -- a NameError that only raised once a specific
    # menu option (e.g. delete) was actually exercised, not on menu display.
    responses = iter(
        [
            "1",
            "cli_test_user",
            "Test User",
            "30",
            "other",
            "test extra",  # create
            "2",
            "cli_test_user",
            "",
            "",
            "",
            "",  # edit (keep current values)
            "4",
            "cli_test_user",  # set default
            "5",
            "dark",  # set theme
            "3",
            "cli_test_user",  # delete
            "4",
            "none",  # clear default
            "6",  # exit
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    _run(monkeypatch, ["users"])
