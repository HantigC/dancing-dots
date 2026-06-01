#!/usr/bin/env python3
"""Upload model weights to a Kaggle model."""

import argparse
from pathlib import Path

import kagglehub
from mts.helpers.kaggle.version import get_project_version

DEFAULT_VARIATION = "default"
DEFAULT_FRAMEWORK = "pytorch"


def upload_one(handle: str, local_dir: Path, notes: str) -> None:
    print(f"Uploading {local_dir} → {handle!r}")
    kagglehub.model_upload(
        handle=handle,
        local_model_dir=str(local_dir),
        version_notes=notes,
    )
    print(f"Done: {handle}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload model weights to a Kaggle dataset."
    )
    parser.add_argument("username", help="Username")
    parser.add_argument("from_dir", help="Directory containing model subdirectories")
    parser.add_argument(
        "models",
        nargs="*",
        metavar="model",
        help="Model names to upload (default: all subdirectories)",
    )
    parser.add_argument(
        "--framework",
        default=DEFAULT_FRAMEWORK,
        help=f"Model framework (default: {DEFAULT_FRAMEWORK!r})",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Version notes (default: 'commit <short-commit>')",
    )
    args = parser.parse_args()

    notes = args.notes or get_project_version()
    base_dir = Path(args.from_dir).resolve()

    subdirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        return

    allowed = set(args.models) if args.models else None
    for subdir in subdirs:
        if allowed is not None and subdir.name not in allowed:
            continue
        handle = f"{args.username}/{subdir.name}/{args.framework}/{DEFAULT_VARIATION}"
        upload_one(handle, subdir, notes)


if __name__ == "__main__":
    main()
