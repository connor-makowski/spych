import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests"]


@nox.session(python=[
    "3.11", 
    "3.12", 
    "3.13", 
    "3.14"
])
def tests(session: nox.Session) -> None:
    """Run standard pytest suite across Python versions (Kokoro on 3.11/3.12, Chatterbox on 3.13+)."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)