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

    # Live transcription
    spych live
    spych live --output-path my_transcript --output-format srt
    spych live --stop-key q --terminate-words "stop recording"
    spych live --no-timestamps --whisper-model small.en

    # Multi-agent: run several agents under different wake words at once
    spych multi --agents claude_code_sdk ollama --ollama-model llama3.2:latest
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
    p_ollama = subparsers.add_parser(
        "ollama", help="Talk to a local Ollama model"
    )
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
    # live — continuous transcription to file                             #
    # ------------------------------------------------------------------ #
    p_live = subparsers.add_parser(
        "live",
        help="Continuously transcribe speech to .txt and/or .srt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Start a live transcription session. Records continuously using VAD\n"
            "and writes output to disk in real time.\n\n"
            "Stop by pressing the stop key (default: q + Enter), saying a\n"
            "terminate word, or pressing Ctrl+C."
        ),
    )
    p_live.add_argument(
        "--output-path",
        default="transcript",
        metavar="PATH",
        help="Base output file path without extension (default: transcript)",
    )
    p_live.add_argument(
        "--output-format",
        default="srt",
        choices=["txt", "srt", "both"],
        metavar="FORMAT",
        help="Output format: txt, srt, or both (default: both)",
    )
    p_live.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Omit timestamps from terminal and .txt output",
    )
    p_live.add_argument(
        "--stop-key",
        default="q",
        metavar="KEY",
        help="Key to type (then Enter) to stop the session (default: q)",
    )
    p_live.add_argument(
        "--terminate-words",
        nargs="+",
        metavar="WORD",
        help="Spoken words that stop the session (e.g. 'stop recording')",
    )
    p_live.add_argument(
        "--device-index",
        type=int,
        default=-1,
        metavar="N",
        help="Microphone device index; -1 uses system default (default: -1)",
    )
    p_live.add_argument(
        "--whisper-model",
        default="base.en",
        metavar="MODEL",
        help="faster-whisper model name (default: base.en)",
    )
    p_live.add_argument(
        "--whisper-device",
        default="cpu",
        choices=["cpu", "cuda"],
        metavar="DEVICE",
        help="Device for whisper inference: cpu or cuda (default: cpu)",
    )
    p_live.add_argument(
        "--whisper-compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        metavar="TYPE",
        help="Compute type for whisper: int8, float16, float32 (default: int8)",
    )
    p_live.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Whisper no_speech_prob cutoff — segments above this are dropped (default: 0.3)",
    )
    p_live.add_argument(
        "--speech-threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="VAD speech onset probability (default: 0.5)",
    )
    p_live.add_argument(
        "--silence-threshold",
        type=float,
        default=0.35,
        metavar="FLOAT",
        help="VAD silence probability during speech (default: 0.35)",
    )
    p_live.add_argument(
        "--silence-frames",
        type=int,
        default=20,
        metavar="N",
        help="Consecutive silent frames required to end a segment (~32ms each, default: 20)",
    )
    p_live.add_argument(
        "--speech-pad-frames",
        type=int,
        default=5,
        metavar="N",
        help="Pre-roll frames and onset confirmation count (default: 5)",
    )
    p_live.add_argument(
        "--max-speech-duration",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Hard cap on a single segment in seconds (default: 30.0)",
    )
    p_live.add_argument(
        "--context-words",
        type=int,
        default=32,
        metavar="N",
        help="Trailing words passed as whisper initial_prompt for context (default: 32)",
    )

    # ------------------------------------------------------------------ #
    p_multi = subparsers.add_parser(
        "multi",
        help="Run multiple agents simultaneously under different wake words",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run several agents at once. Each agent uses its own default wake "
            "words unless overridden.\n\n"
            "Example:\n"
            "  spych multi --agents claude_code_cli gemini_cli\n"
            "  spych multi --agents claude_code_cli ollama --ollama-model llama3.2:latest\n"
            "  spych multi --agents claude_code_sdk codex_cli --listen-duration 8"
        ),
    )
    p_multi.add_argument(
        "--agents",
        nargs="+",
        required=True,
        metavar="AGENT",
        choices=[
            "claude_code_cli",
            "claude_code_sdk",
            "codex_cli",
            "gemini_cli",
            "opencode_cli",
            "ollama",
        ],
        help=(
            "Agents to run. Choices: claude_code_cli, claude_code_sdk, "
            "codex_cli, gemini_cli, opencode_cli, ollama"
        ),
    )
    p_multi.add_argument(
        "--terminate-words",
        nargs="+",
        metavar="WORD",
        default=["terminate"],
        help="Words that stop all agents (default: terminate)",
    )
    p_multi.add_argument(
        "--listen-duration",
        type=float,
        default=5,
        metavar="SECONDS",
        help="Seconds to listen after a wake word (default: 5)",
    )
    p_multi.add_argument(
        "--continue-conversation",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="Resume most recent session for each coding agent (default: true)",
    )
    p_multi.add_argument(
        "--show-tool-events",
        type=_parse_bool,
        default=True,
        metavar="BOOL",
        help="Print live tool start/end events (default: true)",
    )
    # ollama-specific flags (only used when 'ollama' is in --agents)
    p_multi.add_argument(
        "--ollama-model",
        default="llama3.2:latest",
        metavar="MODEL",
        help="Ollama model (default: llama3.2:latest). Only used when ollama is in --agents.",
    )
    p_multi.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama instance URL (default: http://localhost:11434). Only used when ollama is in --agents.",
    )
    p_multi.add_argument(
        "--ollama-history-length",
        type=int,
        default=10,
        metavar="N",
        help="Ollama context history length (default: 10). Only used when ollama is in --agents.",
    )
    # opencode-specific flag
    p_multi.add_argument(
        "--opencode-model",
        default=None,
        metavar="MODEL",
        help="OpenCode model in provider/model format. Only used when opencode_cli is in --agents.",
    )
    # claude_code_sdk-specific flag
    p_multi.add_argument(
        "--setting-sources",
        nargs="+",
        metavar="SOURCE",
        default=["user", "project", "local"],
        help="Claude Code SDK setting sources (default: user project local). Only used when claude_code_sdk is in --agents.",
    )

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Single-agent dispatch                                                #
    # ------------------------------------------------------------------ #
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

    elif args.agent == "live":
        from spych.live import SpychLive

        SpychLive(
            output_format=args.output_format,
            output_path=args.output_path,
            show_timestamps=not args.no_timestamps,
            stop_key=args.stop_key,
            terminate_words=args.terminate_words,
            device_index=args.device_index,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            whisper_compute_type=args.whisper_compute_type,
            no_speech_threshold=args.no_speech_threshold,
            speech_threshold=args.speech_threshold,
            silence_threshold=args.silence_threshold,
            silence_frames_threshold=args.silence_frames,
            speech_pad_frames=args.speech_pad_frames,
            max_speech_duration_s=args.max_speech_duration,
            context_words=args.context_words,
        ).start()

    # ------------------------------------------------------------------ #
    # Multi-agent dispatch                                                 #
    # ------------------------------------------------------------------ #
    elif args.agent == "multi":
        from spych.core import Spych
        from spych.orchestrator import SpychOrchestrator

        # A single Spych transcription object shared by all responders.
        spych_object = Spych(whisper_model="base.en")

        entries = []

        for agent_name in args.agents:
            if agent_name == "claude_code_cli":
                from spych.agents.claude import LocalClaudeCodeCLIResponder

                entries.append({
                    "responder": LocalClaudeCodeCLIResponder(
                        spych_object=spych_object,
                        continue_conversation=args.continue_conversation,
                        listen_duration=args.listen_duration,
                        show_tool_events=args.show_tool_events,
                    ),
                    "wake_words": ["claude", "clod", "cloud", "clawed"],
                    "terminate_words": args.terminate_words,
                })

            elif agent_name == "claude_code_sdk":
                from spych.agents.claude import LocalClaudeCodeSDKResponder

                entries.append({
                    "responder": LocalClaudeCodeSDKResponder(
                        spych_object=spych_object,
                        continue_conversation=args.continue_conversation,
                        listen_duration=args.listen_duration,
                        setting_sources=args.setting_sources,
                        show_tool_events=args.show_tool_events,
                    ),
                    "wake_words": ["claude", "clod", "cloud", "clawed"],
                    "terminate_words": args.terminate_words,
                })

            elif agent_name == "codex_cli":
                from spych.agents.codex import LocalCodexCLIResponder

                entries.append({
                    "responder": LocalCodexCLIResponder(
                        spych_object=spych_object,
                        continue_conversation=args.continue_conversation,
                        listen_duration=args.listen_duration,
                        show_tool_events=args.show_tool_events,
                    ),
                    "wake_words": ["codex"],
                    "terminate_words": args.terminate_words,
                })

            elif agent_name == "gemini_cli":
                from spych.agents.gemini import LocalGeminiCLIResponder

                entries.append({
                    "responder": LocalGeminiCLIResponder(
                        spych_object=spych_object,
                        continue_conversation=args.continue_conversation,
                        listen_duration=args.listen_duration,
                        show_tool_events=args.show_tool_events,
                    ),
                    "wake_words": ["gemini"],
                    "terminate_words": args.terminate_words,
                })

            elif agent_name == "opencode_cli":
                from spych.agents.opencode import LocalOpenCodeCLIResponder

                entries.append({
                    "responder": LocalOpenCodeCLIResponder(
                        spych_object=spych_object,
                        continue_conversation=args.continue_conversation,
                        listen_duration=args.listen_duration,
                        show_tool_events=args.show_tool_events,
                        model=args.opencode_model,
                    ),
                    "wake_words": ["opencode", "open code"],
                    "terminate_words": args.terminate_words,
                })

            elif agent_name == "ollama":
                from spych.agents.ollama import OllamaResponder

                entries.append({
                    "responder": OllamaResponder(
                        spych_object=spych_object,
                        model=args.ollama_model,
                        history_length=args.ollama_history_length,
                        host=args.ollama_host,
                        listen_duration=args.listen_duration,
                    ),
                    "wake_words": ["llama", "ollama", "lama"],
                    "terminate_words": args.terminate_words,
                })

        SpychOrchestrator(entries=entries).start()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()