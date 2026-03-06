"""
Agent implementations for Spych.

This package contains various AI agent implementations that can be used with Spych.
Each agent provides different capabilities and interfaces for interacting with
various language models and AI services.

Agents are organized by the underlying AI service they interface with:
- Claude agents (claude.py)
- Ollama agents (ollama.py)
- Gemini agents (gemini.py)
- Codex agents (codex.py)
- OpenCode agents (opencode.py)

Each file may have multiple agent variants for different usage patterns.
"""
from spych.agents.claude import (
    claude_code_cli,
    LocalClaudeCodeCLIResponder,
    claude_code_sdk,
    LocalClaudeCodeSDKResponder,
)
from spych.agents.ollama import ollama, OllamaResponder
from spych.agents.gemini import gemini_cli, LocalGeminiCLIResponder
from spych.agents.codex import codex_cli, LocalCodexCLIResponder
from spych.agents.opencode import opencode_cli, LocalOpenCodeCLIResponder
