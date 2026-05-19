#!/usr/bin/env python3
"""Download a directory from a Kaggle dataset or competition."""

import argparse
import os
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApiExtended


def download_dataset_dir(
    dataset: str,
    remote_path: str,
    dest: Path,
) -> None:
    """Download a specific path from a Kaggle dataset (owner/name)."""
    api = KaggleApiExtended()
    api.authenticate()

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {dataset!r} path {remote_path!r} -> {dest}")

    api.dataset_download_files(
        dataset,
        path=str(dest),
        unzip=False,
    )

    # kaggle downloads a single zip — unzip only the requested sub-path
    zip_name = dataset.split("/")[-1] + ".zip"
    zip_path = dest / zip_name
    if zip_path.exists():
        _extract_subdir(zip_path, remote_path, dest)
        zip_path.unlink()


def download_competition_dir(
    competition: str,
    remote_path: str,
    dest: Path,
) -> None:
    """Download a specific path from a Kaggle competition."""
    api = KaggleApiExtended()
    api.authenticate()

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading competition {competition!r} path {remote_path!r} -> {dest}")

    api.competition_download_files(
        competition,
        path=str(dest),
    )

    zip_path = dest / f"{competition}.zip"
    if zip_path.exists():
        _extract_subdir(zip_path, remote_path, dest)
        zip_path.unlink()


def _extract_subdir(zip_path: Path, subdir: str, dest: Path) -> None:
    subdir = subdir.strip("/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if not subdir or m.startswith(subdir + "/") or m == subdir
        ]
        if not members:
            print(f"Warning: no entries matching {subdir!r} found in archive, extracting all.")
            members = zf.namelist()
        for member in members:
            zf.extract(member, dest)
    print(f"Extracted {len(members)} entries to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a directory from Kaggle")
    subparsers = parser.add_subparsers(dest="source", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Download from a dataset")
    dataset_parser.add_argument("dataset", help="Dataset slug: owner/name")
    dataset_parser.add_argument("remote_path", help="Sub-path inside the dataset to extract ('' for all)")
    dataset_parser.add_argument("dest", help="Local destination directory")

    comp_parser = subparsers.add_parser("competition", help="Download from a competition")
    comp_parser.add_argument("competition", help="Competition name/slug")
    comp_parser.add_argument("remote_path", help="Sub-path inside the competition to extract ('' for all)")
    comp_parser.add_argument("dest", help="Local destination directory")

    args = parser.parse_args()
    dest = Path(args.dest)

    if args.source == "dataset":
        download_dataset_dir(args.dataset, args.remote_path, dest)
    else:
        download_competition_dir(args.competition, args.remote_path, dest)


if __name__ == "__main__":
    main()
