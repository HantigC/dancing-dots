#!/usr/bin/env python3
"""Upload model weights to a Kaggle model."""

import argparse
import subprocess
from pathlib import Path

import kagglehub

DEFAULT_VARIATION = "default"
DEFAULT_FRAMEWORK = "pytorch"


def _default_version_notes() -> str:
    short_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    return f"commit {short_commit}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload model weights to a Kaggle model."
    )
    parser.add_argument("username", help="Kaggle username")
    parser.add_argument("model", help="Kaggle model slug")
    parser.add_argument(
        "variation",
        nargs="?",
        default=DEFAULT_VARIATION,
        help=f"Model variation slug (default: {DEFAULT_VARIATION!r})",
    )
    parser.add_argument(
        "--framework",
        default=DEFAULT_FRAMEWORK,
        help=f"Model framework (default: {DEFAULT_FRAMEWORK!r})",
    )
    parser.add_argument(
        "--root",
        default="checkpoints",
        help="Local directory containing model weights (default: checkpoints/)",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Version notes (default: 'commit <short-commit>')",
    )
    args = parser.parse_args()

    handle = f"{args.username}/{args.model}/{args.framework}/{args.variation}"
    notes = args.notes or _default_version_notes()
    local_dir = Path(args.root).resolve()

    print(f"Uploading {local_dir} → {handle!r}")
    kagglehub.model_upload(
        handle=handle,
        local_model_dir=str(local_dir),
        version_notes=notes,
    )
    print("Done.")


if __name__ == "__main__":
    main()
