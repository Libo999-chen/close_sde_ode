"""
Compute LPIPS, FID, JFID and PSNR for one or more results directories.

Each results dir must contain:
  recon/   00000.png ...
  label/   00000.png ...

Usage (single):
  python3 eval_metrics.py --results_dir ./results/inpainting

Usage (compare two methods):
  python3 eval_metrics.py \
      --results_dir  ./results/inpainting \
      --results_dir  ./results/joint_plugin/inpainting \
      --name  "DPS" \
      --name  "Joint-Plugin"

Metrics follow the protocol of arXiv 2407.01521:
  - images normalised to [0, 1]
  - LPIPS and FID computed via the piq library
  - JFID (Joint FID): FID between {concat(recon,label)} and {concat(label,label)}
  - PSNR: average peak signal-to-noise ratio (dB), data_range=1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import piq
from piq.feature_extractors import InceptionV3
from tqdm import tqdm


def load_img_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load PNG as float32 tensor in [0, 1], shape (1, 3, H, W)."""
    img = plt.imread(str(path))[:, :, :3]   # [H, W, 3], float32 in [0, 1]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_lpips(recon_dir: Path, label_dir: Path, device: torch.device,
                  loss_fn, max_images: int = None) -> float:
    fnames = sorted(recon_dir.glob("*.png"))
    if not fnames:
        raise FileNotFoundError(f"No PNG files in {recon_dir}")
    if max_images is not None:
        fnames = fnames[:max_images]
    scores = []
    for f in tqdm(fnames, desc=f"LPIPS {recon_dir.parent.name}"):
        recon = load_img_tensor(f, device)
        label = load_img_tensor(label_dir / f.name, device)
        with torch.no_grad():
            scores.append(loss_fn(recon, label).item())
    return float(np.mean(scores))


def extract_inception_features(img_dir: Path, model: torch.nn.Module,
                                device: torch.device,
                                max_images: int = None) -> torch.Tensor:
    fnames = sorted(img_dir.glob("*.png"))
    if max_images is not None:
        fnames = fnames[:max_images]
    feats = []
    batch_size = 8
    for i in tqdm(range(0, len(fnames), batch_size),
                  desc=f"FID feats {img_dir.parent.name}/{img_dir.name}"):
        batch_paths = fnames[i:i + batch_size]
        imgs = torch.cat([load_img_tensor(p, device) for p in batch_paths], dim=0)
        with torch.no_grad():
            feat = model(imgs)[0]           # (B, 2048, 1, 1)
        feats.append(feat.squeeze(-1).squeeze(-1))
    return torch.cat(feats, dim=0)          # (N, 2048)


def compute_ssim(recon_dir: Path, label_dir: Path, device: torch.device,
                 max_images: int = None) -> float:
    fnames = sorted(recon_dir.glob("*.png"))
    if not fnames:
        raise FileNotFoundError(f"No PNG files in {recon_dir}")
    if max_images is not None:
        fnames = fnames[:max_images]
    scores = []
    for f in tqdm(fnames, desc=f"SSIM {recon_dir.parent.name}"):
        recon = load_img_tensor(f, device)
        label = load_img_tensor(label_dir / f.name, device)
        with torch.no_grad():
            scores.append(piq.ssim(recon, label, data_range=1.0).item())
    return float(np.mean(scores))


def compute_psnr(recon_dir: Path, label_dir: Path, device: torch.device,
                 max_images: int = None) -> float:
    fnames = sorted(recon_dir.glob("*.png"))
    if not fnames:
        raise FileNotFoundError(f"No PNG files in {recon_dir}")
    if max_images is not None:
        fnames = fnames[:max_images]
    scores = []
    for f in tqdm(fnames, desc=f"PSNR {recon_dir.parent.name}"):
        recon = load_img_tensor(f, device)
        label = load_img_tensor(label_dir / f.name, device)
        with torch.no_grad():
            scores.append(piq.psnr(recon, label, data_range=1.0).item())
    return float(np.mean(scores))


def compute_fid(recon_dir: Path, label_dir: Path, model: torch.nn.Module,
                device: torch.device, max_images: int = None) -> float:
    recon_feats = extract_inception_features(recon_dir, model, device, max_images)
    label_feats = extract_inception_features(label_dir, model, device, max_images)
    fid_metric = piq.FID()
    return fid_metric(recon_feats, label_feats).item()


def extract_joint_inception_features(recon_dir: Path, label_dir: Path,
                                      model: torch.nn.Module, device: torch.device,
                                      max_images: int = None,
                                      label_label: bool = False) -> torch.Tensor:
    """Features for JFID: each sample is concat(recon, label) or concat(label, label)."""
    fnames = sorted(recon_dir.glob("*.png"))
    if max_images is not None:
        fnames = fnames[:max_images]
    feats = []
    batch_size = 8
    desc = f"JFID {'label-label' if label_label else 'recon-label'} {recon_dir.parent.name}"
    for i in tqdm(range(0, len(fnames), batch_size), desc=desc):
        batch = fnames[i:i + batch_size]
        imgs = []
        for p in batch:
            label = load_img_tensor(label_dir / p.name, device)
            left  = label if label_label else load_img_tensor(p, device)
            imgs.append(torch.cat([left, label], dim=-1))  # concat width-wise → (1,3,H,2W)
        imgs = torch.cat(imgs, dim=0)
        with torch.no_grad():
            feat = model(imgs)[0]
        feats.append(feat.squeeze(-1).squeeze(-1))
    return torch.cat(feats, dim=0)


def compute_jfid(recon_dir: Path, label_dir: Path, model: torch.nn.Module,
                 device: torch.device, max_images: int = None) -> float:
    recon_label = extract_joint_inception_features(
        recon_dir, label_dir, model, device, max_images, label_label=False)
    label_label = extract_joint_inception_features(
        recon_dir, label_dir, model, device, max_images, label_label=True)
    return piq.FID()(recon_label, label_label).item()


def evaluate(results_dir: Path, name: str, device: torch.device,
             lpips_fn, inception: torch.nn.Module,
             max_images: int = None) -> dict:
    recon_dir = results_dir / "recon"
    label_dir = results_dir / "label"
    for d in (recon_dir, label_dir):
        if not d.exists():
            raise FileNotFoundError(f"Not found: {d}")
    n_total = len(list(recon_dir.glob("*.png")))
    n = min(n_total, max_images) if max_images else n_total
    print(f"\n[{name}]  {n} images  —  {results_dir}")
    lpips_val = compute_lpips(recon_dir, label_dir, device, lpips_fn, max_images)
    psnr_val  = compute_psnr(recon_dir, label_dir, device, max_images)
    ssim_val  = compute_ssim(recon_dir, label_dir, device, max_images)
    fid_val   = compute_fid(recon_dir, label_dir, inception, device, max_images)
    jfid_val  = compute_jfid(recon_dir, label_dir, inception, device, max_images)
    return {"lpips": lpips_val, "psnr": psnr_val, "ssim": ssim_val, "fid": fid_val, "jfid": jfid_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, action="append", required=True,
                        dest="results_dirs", metavar="DIR",
                        help="Results directory (repeat for multiple methods)")
    parser.add_argument("--name", type=str, action="append",
                        dest="names", metavar="NAME",
                        help="Method name (repeat to match --results_dir)")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Only evaluate the first N images (sorted by name)")
    args = parser.parse_args()

    # Fill in default names if not provided
    names = args.names or []
    while len(names) < len(args.results_dirs):
        names.append(f"Method {len(names) + 1}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lpips_fn  = piq.LPIPS(reduction='none').to(device).eval()
    inception = InceptionV3(normalize_input=True).to(device).eval()

    results = []
    for d, n in zip(args.results_dirs, names):
        results.append((n, evaluate(Path(d), n, device, lpips_fn, inception, args.max_images)))

    print("\n" + "=" * 76)
    print(f"{'Method':<20}  {'LPIPS':>8}  {'PSNR':>7}  {'SSIM':>6}  {'FID':>10}  {'JFID':>10}")
    print("-" * 76)
    for name, r in results:
        print(f"{name:<20}  {r['lpips']:>8.4f}  {r['psnr']:>7.3f}  {r['ssim']:>6.4f}  {r['fid']:>10.4f}  {r['jfid']:>10.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
