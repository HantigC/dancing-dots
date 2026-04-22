from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import torch
from tqdm.auto import tqdm

from mts.utils.image import imread_rgb

T = TypeVar("T")
K = TypeVar("K")

class BaseEmbedder(ABC, Generic[T, K]):
    @abstractmethod
    def embed_image(self, image: T) -> K:
        pass


def extract_embeddings(
    embedder: BaseEmbedder,
    image_filepaths: list[Path | str],
) -> np.ndarray:
    embeddings = []
    with torch.no_grad():
        for image_filepath in tqdm(image_filepaths):
            img = imread_rgb(image_filepath)
            embedding = embedder.embed_image(img)
            embeddings.append(embedding.cpu())

    return embeddings


def extract_embeddings_from_images(
    embedder: BaseEmbedder,
    images: list[np.ndarray],
) -> list[np.ndarray]:
    results = []
    with torch.no_grad():
        for image in tqdm(images):
            embedding = embedder.embed_image(image)
            results.append(embedding.cpu())
    return results
