from spych.agents import claude_code_cli, ollama

print("Starting Ollama agent")
print("Listening for wake word 'llama'...")
ollama(model="llama3.2:latest")