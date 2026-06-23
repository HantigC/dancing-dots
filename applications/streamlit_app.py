from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mts.pipeline.repository.h5 import H5ImageRepository

BASE_DIR = REPO_ROOT
ITERATIONS_DIR = BASE_DIR / "iterations"
DATA_DIR = BASE_DIR / "data" / "image-matching-challenge-2025"
TRAIN_DIR = DATA_DIR / "train"

st.set_page_config(page_title="Dancing Dots", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_summary(iteration: str) -> dict | None:
    path = ITERATIONS_DIR / iteration / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_submission(iteration: str) -> pd.DataFrame | None:
    path = ITERATIONS_DIR / iteration / "submission.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_train_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train_labels.csv")


def find_h5_files(iteration: str) -> list[Path]:
    root = ITERATIONS_DIR / iteration
    h5_files = sorted(root.glob("h5_repositories/*.h5")) + sorted(root.glob("*.h5"))
    return h5_files


def iter_has_summary(iteration: str) -> bool:
    return (ITERATIONS_DIR / iteration / "summary.json").exists()


def list_iterations() -> list[str]:
    dirs = sorted(
        (d.name for d in ITERATIONS_DIR.iterdir() if d.is_dir() and d.name not in ("dummy", "gt_reconstruct")),
        reverse=True,
    )
    return dirs


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio("Page", ["Results Inspector", "Dataset Browser"])

# ---------------------------------------------------------------------------
# Results Inspector
# ---------------------------------------------------------------------------

if page == "Results Inspector":
    st.title("Results Inspector")

    iterations = list_iterations()
    scored = [i for i in iterations if iter_has_summary(i)]
    all_iters = scored + [i for i in iterations if i not in scored]

    col_iter, col_h5 = st.sidebar.columns([1, 1])
    selected_iter = st.sidebar.selectbox("Iteration", all_iters)

    summary = load_summary(selected_iter)

    # --- Score overview tab / H5 tab ---
    tab_scores, tab_h5 = st.tabs(["Scores", "Repository Browser"])

    # ---- Scores tab ----
    with tab_scores:
        if summary is None:
            st.info("No summary.json for this iteration.")
        else:
            final = summary.get("final_score", 0.0)
            st.metric("Final score", f"{final:.4f}")

            scene_maa = summary.get("scene_mAA_dict", {})
            clusterness = summary.get("scene_clusterness_dict", {})

            scenes = sorted(scene_maa.keys())
            df_scenes = pd.DataFrame(
                {
                    "scene": scenes,
                    "mAA": [scene_maa.get(s, 0.0) for s in scenes],
                    "clusterness": [clusterness.get(s, 0.0) for s in scenes],
                }
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(scenes))
            w = 0.4
            ax.bar(x - w / 2, df_scenes["mAA"], w, label="mAA")
            ax.bar(x + w / 2, df_scenes["clusterness"], w, label="clusterness")
            ax.set_xticks(x)
            ax.set_xticklabels(scenes, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Score (%)")
            ax.legend()
            ax.set_title(f"Iteration {selected_iter} — per-scene scores")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(df_scenes.set_index("scene").style.format("{:.2f}"), use_container_width=True)

        # Score history across all scored iterations
        if len(scored) > 1:
            st.subheader("Score history")
            rows = []
            for it in sorted(scored):
                s = load_summary(it)
                if s:
                    rows.append({"iteration": it, "final_score": s.get("final_score", 0.0)})
            hist = pd.DataFrame(rows).set_index("iteration")
            st.line_chart(hist["final_score"])

    # ---- H5 Browser tab ----
    with tab_h5:
        h5_files = find_h5_files(selected_iter)
        if not h5_files:
            st.info("No H5 repositories found for this iteration.")
        else:
            dataset_name = st.selectbox(
                "Dataset", [f.stem for f in h5_files], key="h5_dataset"
            )
            h5_path = next(f for f in h5_files if f.stem == dataset_name)
            repo = H5ImageRepository(h5_path)

            n_images = repo.images_num()
            n_pairs = repo.pair_num()
            st.write(f"**{n_images}** images · **{n_pairs}** pairs")

            sub_tab_images, sub_tab_pairs = st.tabs(["Images", "Pairs & Matches"])

            # ---- Images sub-tab ----
            with sub_tab_images:
                image_ids = sorted(repo.image_ids())
                if not image_ids:
                    st.info("No images in this repository.")
                else:
                    with repo._reading() as h5:
                        feat_grp = h5.get("features", {})
                        ids_with_kp = {
                            int(iid)
                            for iid in feat_grp.keys()
                            if "keypoints" in feat_grp[iid] and len(feat_grp[iid]["keypoints"]) > 0
                        }

                    kp_filter = st.radio(
                        "Filter",
                        ["All", "With keypoints", "Without keypoints"],
                        horizontal=True,
                        key="kp_filter",
                    )
                    if kp_filter == "With keypoints":
                        image_ids = [i for i in image_ids if i in ids_with_kp]
                    elif kp_filter == "Without keypoints":
                        image_ids = [i for i in image_ids if i not in ids_with_kp]

                    if not image_ids:
                        st.info("No images match the current filter.")
                    else:
                        id_to_name = {
                            iid: Path(repo.get_filepath(iid)).name for iid in image_ids
                        }
                        sel_id = st.selectbox(
                            "Image",
                            image_ids,
                            format_func=lambda i: f"[{i}] {id_to_name[i]}",
                            key="sel_image",
                        )

                        filepath = repo.get_filepath(sel_id)
                        img_path = BASE_DIR / filepath if not Path(filepath).is_absolute() else Path(filepath)

                        kp_names = []
                        with repo._reading() as h5:
                            feat_grp = h5.get("features", {}).get(str(sel_id))
                            if feat_grp and "keypoints" in feat_grp:
                                kp_names = list(feat_grp["keypoints"].keys())

                        show_kp = st.checkbox("Overlay keypoints", value=bool(kp_names), disabled=not kp_names)
                        kp_name = None
                        if kp_names:
                            kp_name = st.selectbox("Keypoint type", kp_names, key="kp_name")

                        if img_path.exists():
                            img = np.array(Image.open(img_path))
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.imshow(img)
                            ax.axis("off")
                            if show_kp and kp_name:
                                kp = repo.get_keypoints(sel_id, name=kp_name)
                                if kp is not None and len(kp):
                                    ax.scatter(kp[:, 0], kp[:, 1], s=4, c="lime", linewidths=0)
                                    ax.set_title(f"{id_to_name[sel_id]}  ({len(kp)} keypoints)")
                                else:
                                    ax.set_title(id_to_name[sel_id])
                            else:
                                ax.set_title(id_to_name[sel_id])
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.warning(f"Image file not found: `{img_path}`")

                        meta = repo.get_metadata(sel_id)
                        if meta:
                            st.json(meta)

                        pose = repo.get_pose(sel_id)
                        if pose is not None:
                            st.write("**Pose**")
                            st.code(f"R =\n{np.array(pose.rotation)}\nt = {np.array(pose.translation)}")

            # ---- Pairs & Matches sub-tab ----
            with sub_tab_pairs:
                pairs = repo.get_pairs()
                if not pairs:
                    st.info("No pairs stored in this repository.")
                else:
                    pair_labels = [
                        f"[{a},{b}]  {Path(repo.get_filepath(a)).name}  ↔  {Path(repo.get_filepath(b)).name}"
                        for a, b in pairs
                    ]
                    sel_pair_idx = st.selectbox("Pair", range(len(pairs)), format_func=lambda i: pair_labels[i], key="sel_pair")
                    id_a, id_b = pairs[sel_pair_idx]

                    match_names = []
                    with repo._reading() as h5:
                        matches_grp = h5.get("matches", {})
                        for mn in matches_grp.keys():
                            key = H5ImageRepository._pair_key(id_a, id_b)
                            if key in matches_grp[mn]:
                                match_names.append(mn)

                    sel_match_name = st.selectbox("Match type", match_names or ["(none)"], key="match_name")
                    matches = repo.get_matches(id_a, id_b, name=sel_match_name) if match_names else None

                    fp_a = repo.get_filepath(id_a)
                    fp_b = repo.get_filepath(id_b)
                    path_a = BASE_DIR / fp_a if not Path(fp_a).is_absolute() else Path(fp_a)
                    path_b = BASE_DIR / fp_b if not Path(fp_b).is_absolute() else Path(fp_b)

                    if path_a.exists() and path_b.exists():
                        img_a = np.array(Image.open(path_a))
                        img_b = np.array(Image.open(path_b))

                        kp_a = repo.get_keypoints(id_a, name=sel_match_name) if match_names else None
                        kp_b = repo.get_keypoints(id_b, name=sel_match_name) if match_names else None

                        h_a, w_a = img_a.shape[:2]
                        h_b, w_b = img_b.shape[:2]
                        h = max(h_a, h_b)
                        canvas = np.zeros((h, w_a + w_b, 3), dtype=np.uint8)
                        canvas[:h_a, :w_a] = img_a
                        canvas[:h_b, w_a:] = img_b

                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.imshow(canvas)
                        ax.axis("off")

                        if matches is not None and kp_a is not None and kp_b is not None and len(matches):
                            max_draw = 200
                            step = max(1, len(matches) // max_draw)
                            rng = np.random.default_rng(42)
                            colors = rng.uniform(0, 1, (len(matches[::step]), 3))
                            for idx, (ma, mb) in enumerate(matches[::step]):
                                xa, ya = kp_a[ma, :2]
                                xb, yb = kp_b[mb, :2]
                                ax.plot([xa, xb + w_a], [ya, yb], lw=0.5, c=colors[idx], alpha=0.7)
                            ax.set_title(f"{len(matches)} matches  (showing {len(matches[::step])})")

                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    match_meta = repo.get_match_metadata(id_a, id_b)
                    if match_meta:
                        st.json(match_meta)

# ---------------------------------------------------------------------------
# Dataset Browser
# ---------------------------------------------------------------------------

elif page == "Dataset Browser":
    st.title("Dataset Browser")

    if not TRAIN_DIR.exists():
        st.error(f"Train data not found at `{TRAIN_DIR}`")
        st.stop()

    datasets = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
    sel_dataset = st.sidebar.selectbox("Dataset", datasets)
    dataset_dir = TRAIN_DIR / sel_dataset

    labels = load_train_labels()
    ds_labels = labels[labels["dataset"] == sel_dataset]

    images = sorted(dataset_dir.glob("*.png")) + sorted(dataset_dir.glob("*.jpg"))
    scenes = sorted(ds_labels["scene"].unique()) if not ds_labels.empty else []

    st.subheader(f"{sel_dataset}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Images", len(images))
    col2.metric("Scenes", len(scenes))
    col3.metric("GT poses", len(ds_labels))

    tab_grid, tab_scene, tab_gt = st.tabs(["Image Grid", "Scenes", "Ground Truth"])

    with tab_grid:
        filter_scene = st.selectbox("Filter by scene", ["(all)"] + scenes, key="grid_scene")
        if filter_scene != "(all)":
            scene_images = set(ds_labels[ds_labels["scene"] == filter_scene]["image"].tolist())
            shown_images = [p for p in images if p.name in scene_images]
        else:
            shown_images = images

        max_show = st.slider("Max images", 4, min(100, len(shown_images)), min(24, len(shown_images)), step=4, key="grid_max")
        shown_images = shown_images[:max_show]

        cols_per_row = 4
        for row_start in range(0, len(shown_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, img_path in zip(cols, shown_images[row_start: row_start + cols_per_row]):
                with col:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)

    with tab_scene:
        if not scenes:
            st.info("No scene labels available.")
        else:
            scene_counts = ds_labels.groupby("scene")["image"].count().reset_index()
            scene_counts.columns = ["scene", "image_count"]

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(scene_counts["scene"], scene_counts["image_count"])
            ax.set_xlabel("Images")
            ax.set_title(f"{sel_dataset} — images per scene")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(scene_counts.set_index("scene"), use_container_width=True)

    with tab_gt:
        if ds_labels.empty:
            st.info("No ground truth labels for this dataset.")
        else:
            sel_scene = st.selectbox("Scene", scenes, key="gt_scene")
            scene_df = ds_labels[ds_labels["scene"] == sel_scene][["image", "rotation_matrix", "translation_vector"]]
            st.dataframe(scene_df.set_index("image"), use_container_width=True)
