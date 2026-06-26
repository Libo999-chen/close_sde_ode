"""
Download 1000 validation images from ImageNet 256x256 (evanarlian/imagenet_1k_resized_256).
Public dataset, no HuggingFace authentication required.
Saves images to data/imagenet256/.
"""

import io
import argparse
from pathlib import Path
from datasets import load_dataset
from datasets.features import Image as HFImage
import PIL.Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/imagenet256")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Number of images to download")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (streaming, no image decode)...")
    ds = load_dataset(
        "evanarlian/imagenet_1k_resized_256",
        split=args.split,
        streaming=True,
    )
    # Disable automatic PIL decoding to work around Pillow<10 incompatibility
    ds = ds.cast_column("image", HFImage(decode=False))

    print(f"Downloading {args.num_images} images to {out_dir} ...")
    count = 0
    for sample in tqdm(ds, total=args.num_images):
        if count >= args.num_images:
            break
        raw = sample["image"]
        # raw is a dict with 'bytes' and/or 'path'
        if raw.get("bytes"):
            img = PIL.Image.open(io.BytesIO(raw["bytes"]))
        else:
            img = PIL.Image.open(raw["path"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Center-crop to 256x256 (standard ImageNet preprocessing for DPS)
        w, h = img.size
        left = (w - 256) // 2
        top = (h - 256) // 2
        img = img.crop((left, top, left + 256, top + 256))
        save_path = out_dir / f"{str(count).zfill(5)}.png"
        img.save(save_path)
        count += 1

    print(f"Done. Saved {count} images to {out_dir}")


if __name__ == "__main__":
    main()
