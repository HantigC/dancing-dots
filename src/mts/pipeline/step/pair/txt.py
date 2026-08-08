import logging
from pathlib import Path

from mts.core.types import PathLike, PairType
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository

LOGGER = logging.getLogger(__name__)


class LoadPairsFromTxtStep(BasePipelineStep):
    """Load image pairs from a plain-text file and add them to the repository.

    File format: one pair per line, two filenames separated by whitespace.
    Filenames are matched against the repository by basename.
    """

    def __init__(self, filepath: PathLike) -> None:
        super().__init__()
        self._filepath = Path(filepath)

    @use_image_repository
    def run(self, image_repository: BaseImageRepository) -> None:
        pairs = self._load(image_repository)
        LOGGER.info("Loaded %d pairs from '%s'", len(pairs), self._filepath)
        image_repository.add_pairs(pairs)

    def _load(self, image_repository: BaseImageRepository) -> list[PairType[int]]:
        basename_to_id: dict[str, int] = {}
        for image_id in image_repository.image_ids():
            filepath = image_repository.get_filepath(image_id)
            if filepath is not None:
                basename_to_id[Path(filepath).name] = image_id

        pairs: list[PairType[int]] = []
        missing: set[str] = set()

        with open(self._filepath) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    LOGGER.warning(
                        "%s:%d — expected 2 filenames, got %d; skipping",
                        self._filepath,
                        lineno,
                        len(parts),
                    )
                    continue
                name_a, name_b = parts
                id_a = basename_to_id.get(name_a)
                id_b = basename_to_id.get(name_b)
                if id_a is None:
                    missing.add(name_a)
                if id_b is None:
                    missing.add(name_b)
                if id_a is None or id_b is None:
                    continue
                pairs.append((min(id_a, id_b), max(id_a, id_b)))

        if missing:
            LOGGER.warning(
                "%d filename(s) from '%s' not found in repository: %s",
                len(missing),
                self._filepath,
                sorted(missing),
            )

        return pairs


class SavePairsToTxtStep(BasePipelineStep):
    """Write the repository's current pairs to a per-dataset txt file.

    The output file is written to `dirpath/<dataset_name>.txt` where
    `dataset_name` is read from repository metadata.  Each line contains
    two space-separated image basenames.
    """

    def __init__(self, dirpath: PathLike) -> None:
        super().__init__()
        self._dirpath = Path(dirpath)

    @use_image_repository
    def run(self, image_repository: BaseImageRepository) -> None:
        dataset_name = image_repository.get_repository_metadata("dataset_name") or "pairs"
        self._dirpath.mkdir(parents=True, exist_ok=True)
        dest = self._dirpath / f"{dataset_name}.txt"

        pairs = image_repository.get_pairs()
        with open(dest, "w") as fh:
            for id_a, id_b in pairs:
                name_a = Path(image_repository.get_filepath(id_a)).name
                name_b = Path(image_repository.get_filepath(id_b)).name
                fh.write(f"{name_a} {name_b}\n")

        LOGGER.info("Saved %d pairs to '%s'", len(pairs), dest)
