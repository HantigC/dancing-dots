#!/usr/bin/env python3
"""
Reconstruct train scenes using ground truth poses from train_labels.csv.

Pipeline per scene:
  1. Extract SIFT keypoints + exhaustive matching -> COLMAP DB
  2. Geometric verification (verify_matches)
  3. Build pycolmap.Reconstruction from GT R, t
  4. Run pycolmap.triangulate_points for 3D structure
  5. Save sparse reconstruction to iterations/gt_reconstruct/<dataset>/<scene>/

Usage:
  uv run python scripts/reconstruct_gt.py
  uv run python scripts/reconstruct_gt.py --dataset imc2023_haiper --scene fountain
  uv run python scripts/reconstruct_gt.py --dataset imc2023_haiper
"""

import argparse
import itertools
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from mts.helpers.colmap.database import COLMAPDatabase
from mts.helpers.colmap.h5_to_db import CameraModel, create_camera
from mts.pipeline.step.pair.sift import extract_sift_features, match_descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

DATA_ROOT = Path("data/image-matching-challenge-2025")
TRAIN_LABELS = DATA_ROOT / "train_labels.csv"
TRAIN_IMAGES = DATA_ROOT / "train"
OUTPUT_ROOT = Path("iterations/gt_reconstruct")
MIN_MATCHES = 15


def rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return w, x, y, z


def load_images(image_paths: list[Path]) -> list[np.ndarray]:
    from PIL import Image
    images = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB"))
        images.append(img)
    return images


def build_colmap_db(
    images_dir: Path,
    image_names: list[str],
    db_path: Path,
) -> dict[str, int]:
    """Extract SIFT, match exhaustively, write COLMAP DB. Returns name->db_image_id."""
    image_paths = [images_dir / n for n in image_names]
    images = load_images(image_paths)

    LOGGER.info("Extracting SIFT for %d images", len(images))
    kpts_desc = extract_sift_features(images, num_features=2048, tqdm_kwargs={"desc": "SIFT"})

    db = COLMAPDatabase(db_path)
    db.create_tables()

    name_to_db_id: dict[str, int] = {}
    for name, img_path, (kpts, descs) in zip(image_names, image_paths, kpts_desc):
        cam_id = create_camera(db, str(img_path), CameraModel.SIMPLE_RADIAL)
        db_img_id = db.add_image(name, cam_id)
        name_to_db_id[name] = db_img_id
        if kpts is not None and len(kpts) > 0:
            db.add_keypoints(db_img_id, kpts)

    LOGGER.info("Matching %d pairs", len(image_names) * (len(image_names) - 1) // 2)
    pairs_matched = []
    for i, j in tqdm(
        list(itertools.combinations(range(len(image_names)), 2)), desc="Matching"
    ):
        _, desc_i = kpts_desc[i]
        _, desc_j = kpts_desc[j]
        if desc_i is None or desc_j is None:
            continue
        _, matches = match_descriptors(desc_i, desc_j, ratio_threshold=0.75)
        if len(matches) >= MIN_MATCHES:
            db.add_matches(name_to_db_id[image_names[i]], name_to_db_id[image_names[j]], matches)
            pairs_matched.append((image_names[i], image_names[j]))

    db.commit()
    db.close()
    LOGGER.info("Matched %d pairs with >= %d matches", len(pairs_matched), MIN_MATCHES)
    return name_to_db_id, pairs_matched


_COLMAP_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def _read_db_cameras_images(db_path: Path):
    """Read cameras and images from a COLMAP DB via sqlite3."""
    from mts.helpers.colmap.database import COLMAPDatabase

    db = COLMAPDatabase.connect(db_path)
    cameras = {}
    for cam_id, model_int, w, h, params_blob in db.execute(
        "SELECT camera_id, model, width, height, params FROM cameras"
    ).fetchall():
        params = np.frombuffer(params_blob, dtype=np.float64)
        cameras[cam_id] = {
            "model": _COLMAP_MODEL_NAMES.get(model_int, str(model_int)),
            "width": w,
            "height": h,
            "params": params,
        }
    images = {}
    for img_id, name, cam_id in db.execute(
        "SELECT image_id, name, camera_id FROM images"
    ).fetchall():
        images[name] = {"image_id": img_id, "camera_id": cam_id}
    db.close()
    return cameras, images


def build_gt_reconstruction(
    scene_df: pd.DataFrame,
    db_path: Path,
) -> pycolmap.Reconstruction:
    """Build a pycolmap.Reconstruction with known GT poses via text files."""
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in scene_df.iterrows():
        R = np.array(row["rotation_matrix"].split(";"), dtype=float).reshape(3, 3)
        t = np.array(row["translation_vector"].split(";"), dtype=float)
        poses[row["image"]] = (R, t)

    db_cameras, db_images = _read_db_cameras_images(db_path)

    cam_lines: list[str] = [
        "# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"
    ]
    img_lines: list[str] = [
        "# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]

    for img_name, (R, t) in poses.items():
        if img_name not in db_images:
            LOGGER.warning("Image %s not in DB, skipping", img_name)
            continue
        db_img = db_images[img_name]
        cam_id = db_img["camera_id"]
        cam = db_cameras[cam_id]

        cam_line = (
            f"{cam_id} {cam['model']} {cam['width']} {cam['height']} "
            + " ".join(str(p) for p in cam["params"])
        )
        if cam_line not in cam_lines:
            cam_lines.append(cam_line)

        qw, qx, qy, qz = rot_to_quat_wxyz(R)
        tx, ty, tz = t
        img_lines.append(
            f"{db_img['image_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"
            f" {cam_id} {img_name}"
        )
        img_lines.append("")  # empty points2D line

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "cameras.txt"), "w") as f:
            f.write("\n".join(cam_lines) + "\n")
        with open(os.path.join(tmpdir, "images.txt"), "w") as f:
            f.write("\n".join(img_lines) + "\n")
        with open(os.path.join(tmpdir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")

        recon = pycolmap.Reconstruction()
        recon.read_text(tmpdir)

    return recon


def reconstruct_scene(
    dataset: str,
    scene: str,
    scene_df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "colmap.db"
    pairs_path = output_dir / "pairs.txt"
    triangulated_dir = output_dir / "triangulated"

    image_names = scene_df["image"].tolist()
    LOGGER.info("[%s/%s] %d images", dataset, scene, len(image_names))

    if db_path.exists():
        db_path.unlink()

    _, pairs_matched = build_colmap_db(images_dir, image_names, db_path)

    pairs_path.write_text(
        "\n".join(f"{a} {b}" for a, b in pairs_matched) + "\n"
    )
    LOGGER.info("Running geometric verification on %d pairs", len(pairs_matched))
    pycolmap.verify_matches(str(db_path), str(pairs_path))

    gt_recon = build_gt_reconstruction(scene_df, db_path)
    LOGGER.info(
        "GT reconstruction: %d cameras, %d registered images",
        gt_recon.num_cameras(),
        gt_recon.num_reg_images(),
    )

    if triangulated_dir.exists():
        shutil.rmtree(triangulated_dir)
    triangulated_dir.mkdir()

    triangulated = pycolmap.triangulate_points(
        gt_recon,
        str(db_path),
        str(images_dir),
        str(triangulated_dir),
    )

    n_pts = triangulated.num_points3D()
    n_reg = triangulated.num_reg_images()
    LOGGER.info("[%s/%s] Done: %d registered images, %d 3D points", dataset, scene, n_reg, n_pts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Process only this dataset")
    parser.add_argument("--scene", help="Process only this scene (requires --dataset)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Root output directory (default: {OUTPUT_ROOT})",
    )
    args = parser.parse_args()

    labels = pd.read_csv(TRAIN_LABELS)
    groups = labels.groupby(["dataset", "scene"])

    for (dataset, scene), scene_df in groups:
        if args.dataset and dataset != args.dataset:
            continue
        if args.scene and scene != args.scene:
            continue

        images_dir = TRAIN_IMAGES / dataset
        output_dir = args.output_root / dataset / scene

        try:
            reconstruct_scene(dataset, scene, scene_df, images_dir, output_dir)
        except Exception:
            LOGGER.exception("Failed for %s/%s", dataset, scene)


if __name__ == "__main__":
    main()
