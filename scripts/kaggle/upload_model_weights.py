#!/usr/bin/env python3
"""Upload model weights to a Kaggle model."""

import argparse
from pathlib import Path

import kagglehub
from .version import get_project_version

DEFAULT_VARIATION = "default"
DEFAULT_FRAMEWORK = "pytorch"


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
    notes = args.notes or get_project_version()
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
