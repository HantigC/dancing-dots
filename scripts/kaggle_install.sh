#!/usr/bin/env bash
# Run this in a Kaggle notebook cell before importing the project:
#   !bash /kaggle/input/<dataset-slug>/scripts/kaggle_install.sh

set -e

REPO="${1:-/kaggle/input/dancing-dots}"

# Local wheels not on PyPI
pip install --quiet \
    "$REPO/deps/whls/croco-0.1.2-py3-none-any.whl" \
    "$REPO/deps/whls/dust3r-0.1.0-py3-none-any.whl" \
    "$REPO/deps/whls/mast3r-0.1.0-py3-none-any.whl"

# asmk submodule (needs setuptools at build time)
pip install --quiet "$REPO/deps/submodules/asmk"

# hloc from GitHub
pip install --quiet git+https://github.com/cvg/Hierarchical-Localization.git

# The project itself (no-deps: all real deps already in Kaggle's base image or installed above)
pip install --quiet --no-deps "$REPO"
