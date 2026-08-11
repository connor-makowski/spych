import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent.parent

VERSION = "4.3.0"
OLD_DOC_VERSIONS = ["3.6.0", "2.0.2", "1.0.0"]

env = {
    **os.environ,
    "version_options": " ".join([VERSION] + OLD_DOC_VERSIONS),
}


def generate_docs(version: str) -> None:
    out_dir = str(root / "docs" / version)
    template_dir = str(root / "doc_template")

    if version != "./" and version != VERSION:
        tarball = str(root / "dist" / f"spych-{version}.tar.gz")
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--with",
                tarball,
                "--with",
                "pdoc",
                "pdoc",
                "-o",
                out_dir,
                "-t",
                template_dir,
                "spych",
                "!spych.agents.sdk_workers",
            ],
            check=True,
            env=env,
            cwd=str(root),
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pdoc",
                "-o",
                out_dir,
                "-t",
                template_dir,
                "spych",
                "!spych.agents.sdk_workers",
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    generate_docs("./")
    generate_docs(VERSION)
    for v in OLD_DOC_VERSIONS:
        generate_docs(v)
