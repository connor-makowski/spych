---
name: dashboard
description: Develop, customize, or debug the AgentDashboard rich TUI interface. Use when working on terminal UI layout, thought streaming, tool tracking, or interactive logging.
---

# Agent Dashboard TUI

`AgentDashboard` (in `spych/dashboard.py`) provides an interactive, rich terminal user interface (TUI) for real-time monitoring and control of voice AI agent sessions.

---

## 1. Key TUI Features

- **Alternate Screen Buffer**: Uses terminal alternate screen buffer to isolate dashboard UI from main shell output.
- **Thought Streaming & Tool Tracking**: Displays real-time agent reasoning, intermediate tools, and execution steps.
- **Conversation History & Log Viewer**: Toggleable view modes, including scrollable "All Logs" mode with wrapped text rendering.
- **Dedicated Input Thread**: Non-blocking keyboard input loop for scrolling, switching modes, and terminating sessions.

---

## 2. Component Integration (`dashboard.py`)

When modifying `AgentDashboard`:
- Maintain internal text wrapping cache to prevent high CPU rendering overhead during streaming.
- Ensure thread synchronization between background agent execution events and TUI draw refreshes.
- Gracefully handle terminal resizing (`SIGWINCH`) and screen clearing upon exit.

---

## 3. Testing TUI Changes

Test dashboard rendering with `test/12_dashboard.py`:
```bash
python test/12_dashboard.py
```
Or launch via CLI:
```bash
spych chat --dashboard
```

Run [prettify](../prettify/SKILL.md) after updating UI logic.
