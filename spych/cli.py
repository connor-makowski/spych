"""
spych CLI entry point.

Usage:
    spych <agent> [options]

Examples:
    spych ollama --model llama3.2:latest
    spych claude_code_cli
    spych claude_code_sdk --setting-sources user project local
    spych codex_cli --listen-duration 8
    spych gemini_cli
    spych opencode_cli --model anthropic/claude-sonnet-4-5
"""

import argparse
import sys


def _parse_bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value!r}")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by all agents."""
    parser.add_argument(
        "--wake-words",
        nargs="+",
        metavar="WORD",
        help="One or more wake words that trigger the agent",
    )
    parser.add_argument(
        "--terminate-words",
        nargs="+",
        metavar="WORD",
        default=["terminate"],
        help="Words that stop the listener (default: terminate)",
    )
    parser.add_argument(
        "--listen-duration",
        type=float,
        metavar="SECONDS",
        help="Seconds to listen after wake word (default: 5)",
    )


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by all coding agents (non-Ollama)."""
    parser.add_argument(
        "--continue-conversation",
        type=_parse_bool,
        metavar="BOOL",
        default=True,
        help="Resume the most recent session (default: true)",
    )
    parser.add_argument(
        "--show-tool-events",
        type=_parse_bool,
        metavar="BOOL",
        default=True,
        help="Print live tool start/end events (default: true)",
    )


def _build_shared_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {}
    if args.wake_words:
        kwargs["wake_words"] = args.wake_words
    if args.terminate_words:
        kwargs["terminate_words"] = args.terminate_words
    if args.listen_duration is not None:
        kwargs["listen_duration"] = args.listen_duration
    return kwargs


def _build_agent_kwargs(args: argparse.Namespace) -> dict:
    kwargs = _build_shared_kwargs(args)
    kwargs["continue_conversation"] = args.continue_conversation
    kwargs["show_tool_events"] = args.show_tool_events
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        prog="spych",
        description="Launch a voice agent from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="agent", metavar="agent")
    subparsers.required = True

    # ------------------------------------------------------------------ #
    # ollama                                                               #
    # ------------------------------------------------------------------ #
    p_ollama = subparsers.add_parser("ollama", help="Talk to a local Ollama model")
    _add_shared_args(p_ollama)
    p_ollama.add_argument(
        "--model",
        default="llama3.2:latest",
        metavar="MODEL",
        help="Ollama model name (default: llama3.2:latest)",
    )
    p_ollama.add_argument(
        "--history-length",
        type=int,
        default=10,
        metavar="N",
        help="Past interactions to include in context (default: 10)",
    )
    p_ollama.add_argument(
        "--host",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama instance URL (default: http://localhost:11434)",
    )

    # ------------------------------------------------------------------ #
    # claude_code_cli                                                      #
    # ------------------------------------------------------------------ #
    p_claude_cli = subparsers.add_parser(
        "claude_code_cli", help="Voice-control Claude Code via the CLI"
    )
    _add_shared_args(p_claude_cli)
    _add_agent_args(p_claude_cli)

    # ------------------------------------------------------------------ #
    # claude_code_sdk                                                      #
    # ------------------------------------------------------------------ #
    p_claude_sdk = subparsers.add_parser(
        "claude_code_sdk", help="Voice-control Claude Code via the Agent SDK"
    )
    _add_shared_args(p_claude_sdk)
    _add_agent_args(p_claude_sdk)
    p_claude_sdk.add_argument(
        "--setting-sources",
        nargs="+",
        metavar="SOURCE",
        default=["user", "project", "local"],
        help="Claude Code settings sources to load (default: user project local)",
    )

    # ------------------------------------------------------------------ #
    # codex_cli                                                            #
    # ------------------------------------------------------------------ #
    p_codex = subparsers.add_parser(
        "codex_cli", help="Voice-control the OpenAI Codex agent"
    )
    _add_shared_args(p_codex)
    _add_agent_args(p_codex)

    # ------------------------------------------------------------------ #
    # gemini_cli                                                           #
    # ------------------------------------------------------------------ #
    p_gemini = subparsers.add_parser(
        "gemini_cli", help="Voice-control the Google Gemini agent"
    )
    _add_shared_args(p_gemini)
    _add_agent_args(p_gemini)

    # ------------------------------------------------------------------ #
    # opencode_cli                                                         #
    # ------------------------------------------------------------------ #
    p_opencode = subparsers.add_parser(
        "opencode_cli", help="Voice-control the OpenCode agent"
    )
    _add_shared_args(p_opencode)
    _add_agent_args(p_opencode)
    p_opencode.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model in provider/model format, e.g. anthropic/claude-sonnet-4-5",
    )

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #
    args = parser.parse_args()

    if args.agent == "ollama":
        from spych.agents import ollama

        kwargs = _build_shared_kwargs(args)
        kwargs["model"] = args.model
        kwargs["history_length"] = args.history_length
        kwargs["host"] = args.host
        ollama(**kwargs)

    elif args.agent == "claude_code_cli":
        from spych.agents import claude_code_cli

        claude_code_cli(**_build_agent_kwargs(args))

    elif args.agent == "claude_code_sdk":
        from spych.agents import claude_code_sdk

        kwargs = _build_agent_kwargs(args)
        kwargs["setting_sources"] = args.setting_sources
        claude_code_sdk(**kwargs)

    elif args.agent == "codex_cli":
        from spych.agents import codex_cli

        codex_cli(**_build_agent_kwargs(args))

    elif args.agent == "gemini_cli":
        from spych.agents import gemini_cli

        gemini_cli(**_build_agent_kwargs(args))

    elif args.agent == "opencode_cli":
        from spych.agents import opencode_cli

        kwargs = _build_agent_kwargs(args)
        if args.model is not None:
            kwargs["model"] = args.model
        opencode_cli(**kwargs)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()