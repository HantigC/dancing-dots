#!/usr/bin/env python3
"""Upload the project code and assets to a Kaggle dataset."""

import argparse
import subprocess
from importlib.metadata import version
from pathlib import Path

from mts.helpers.kaggle.version import get_project_version, get_project_name
import kagglehub

# Directories and files that are not needed at Kaggle runtime
IGNORE_PATTERNS = [
    ".vscode/",
    ".claude/",
    "CLAUDE.md",
    # data — mounted as a competition dataset on Kaggle
    "data/",
    # output directories
    "iterations/",
    "runs/",
    "tmp/",
    # local tooling

    "**/.venv/",
    ".venv/",
    "kaggle/",
    "ipynbs/",
    # build artefacts
    "*.egg-info/",
    "__pycache__/",
    ".ipynb_checkpoints/",
    # model weights — upload separately or mount as a Kaggle dataset
    "model-weights/",
    # misc
    ".git/",
    ".DS_Store",
    "*.zip",
    "*.ipynb",
    "uv.lock",
]


def _default_dataset_name() -> str:
    pkg_version = version("dancing-dots").replace(".", "-")
    short_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    return f"dancing-dots-{pkg_version}-{short_commit}"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload project code and assets to a Kaggle dataset."
    )
    parser.add_argument("username", help="Kaggle username")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help=(
            "Kaggle dataset slug "
            "(default: dancing-dots-<version>-<short-commit>)"
        ),
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Local project root to upload (default: current directory)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Version notes for this upload",
    )
    args = parser.parse_args()

    dataset = args.dataset or get_project_name()
    notes = args.notes or get_project_version()
    handle = f"{args.username}/{dataset}"
    local_dir = Path(args.root).resolve()
    print(f"Uploading {local_dir} → {handle!r}")
    kagglehub.dataset_upload(
        handle,
        str(local_dir),
        version_notes=notes,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print("Done.")


if __name__ == "__main__":
    main()
