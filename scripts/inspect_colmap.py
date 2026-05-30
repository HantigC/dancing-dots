import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mts.helpers.colmap.database import COLMAPDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--image-id", type=int, default=None)
    args = parser.parse_args()

    db = COLMAPDatabase.connect(args.db_path)

    images = db.fetch_images(eager=True)
    for img in images:
        print(img["image_id"], img["name"])

    if args.image_id is not None:
        kps = db.select_kp(args.image_id)
        if kps is None:
            print(f"No keypoints found for image_id={args.image_id}")
        else:
            print(f"\nimage_id={args.image_id}  shape={kps.shape}")
            print(kps[:5])

    db.close()


if __name__ == "__main__":
    main()
