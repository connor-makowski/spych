"""
spinners.py — Spinner frame definitions for CliSpinner.

Usage:
    from spych.spinners import Spinner

    # Access the enum-style constants directly
    frames = Spinner.BRAILLE
"""


class Spinner:
    """
    Named frame sequences.  Each is a plain list[str] you can pass directly
    to CliSpinner.start() or store on CliSpinner.FRAMES.
    """

    # Braille / classic dots — original default
    BRAILLE: list[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    # Circle arc sweep (Claude-style)
    ARC: list[str] = ["◐", "◓", "◑", "◒"]

    # Circle arc corners
    CIRCLE_ARCS: list[str] = ["◜", "◝", "◞", "◟"]

    # Filled-circle pulse
    CIRCLE_FILL: list[str] = ["○", "◎", "●", "◎"]

    # Rotating line / clock hand
    LINE: list[str] = ["─", "╲", "│", "╱"]

    # Classic ASCII pipe
    PIPE: list[str] = ["|", "/", "─", "\\"]

    # Growing/shrinking dot
    DOT_PULSE: list[str] = ["·", "•", "●", "•"]

    # Bouncing-ball braille
    BOUNCE: list[str] = ["⠁", "⠂", "⠄", "⡀", "⠄", "⠂"]

    # Block / bar fill
    BLOCK: list[str] = ["░", "▒", "▓", "█", "▓", "▒", "░", " "]

    # Arrow chase (8 directions)
    ARROW: list[str] = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]

    # Vertical equalizer bar
    EQUALIZER: list[str] = [
        "▁",
        "▂",
        "▃",
        "▄",
        "▅",
        "▆",
        "▇",
        "█",
        "▇",
        "▆",
        "▅",
        "▄",
        "▃",
        "▂",
    ]

    # Zen / full-braille rotation
    ZEN: list[str] = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

    # Moon phases  (emoji — use a slower interval, ~150 ms)
    MOON: list[str] = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]

    # Clock face  (emoji — use a slower interval, ~100 ms)
    CLOCK: list[str] = [
        "🕛",
        "🕐",
        "🕑",
        "🕒",
        "🕓",
        "🕔",
        "🕕",
        "🕖",
        "🕗",
        "🕘",
        "🕙",
        "🕚",
    ]
