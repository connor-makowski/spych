from spych.agents import gemini_cli

print("Starting Gemini CLI agent")
print("Listening for wake word 'gemini'...")
gemini_cli(wake_words=["gemini"])
