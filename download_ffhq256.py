"""
Download 1000 validation images from FFHQ 256x256 (bitmind/ffhq-256).
Public dataset, no HuggingFace authentication required.
Saves images to data/samples/ by default.
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
    parser.add_argument("--output_dir", type=str, default="./data/samples")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Number of images to download")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Skip this many images before saving (for selecting validation split)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (streaming)...")
    ds = load_dataset(
        "bitmind/ffhq-256",
        split="train",
        streaming=True,
    )
    ds = ds.cast_column("image", HFImage(decode=False))

    if args.start_index > 0:
        print(f"Skipping first {args.start_index} images...")
        ds = ds.skip(args.start_index)

    print(f"Downloading {args.num_images} images to {out_dir} ...")
    count = 0
    for sample in tqdm(ds, total=args.num_images):
        if count >= args.num_images:
            break
        raw = sample["image"]
        if raw.get("bytes"):
            img = PIL.Image.open(io.BytesIO(raw["bytes"]))
        else:
            img = PIL.Image.open(raw["path"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        save_path = out_dir / f"{str(count + args.start_index).zfill(5)}.png"
        img.save(save_path)
        count += 1

    print(f"Done. Saved {count} images to {out_dir}")


if __name__ == "__main__":
    main()
