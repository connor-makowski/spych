from spych.agents import claude_code_cli

print("Starting Claude Code CLI agent with CPU support.")
print("Listening for wake word 'claude'...")
claude_code_cli(whisper_device="cpu")