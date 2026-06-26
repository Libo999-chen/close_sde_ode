import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = "/tmp/lc2762/diffusion-posterior-sampling-death/results"

DIRS = {
    "gt":      os.path.join(BASE, "joint_plugin/inpainting/label"),
    "input":   os.path.join(BASE, "joint_plugin/inpainting/input"),
    "dps":     os.path.join(BASE, "inpainting/recon"),
    "diffpir": os.path.join(BASE, "diffpir/inpainting/recon"),
    "ours":    os.path.join(BASE, "joint_plugin/inpainting/recon"),
}

# Find common files across all dirs
file_sets = [set(os.listdir(d)) for d in DIRS.values()]
common = sorted(set.intersection(*file_sets))
print(f"Common files: {len(common)}")

def psnr(img1, img2):
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Score each file by how much "ours" beats both DPS and DiffPIR
scores = []
for fname in common:
    gt_img      = Image.open(os.path.join(DIRS["gt"],      fname)).convert("RGB")
    our_img     = Image.open(os.path.join(DIRS["ours"],    fname)).convert("RGB")
    dps_img     = Image.open(os.path.join(DIRS["dps"],     fname)).convert("RGB")
    diffpir_img = Image.open(os.path.join(DIRS["diffpir"], fname)).convert("RGB")

    p_ours    = psnr(gt_img, our_img)
    p_dps     = psnr(gt_img, dps_img)
    p_diffpir = psnr(gt_img, diffpir_img)

    # advantage = total margin over both baselines
    advantage = (p_ours - p_dps) + (p_ours - p_diffpir)
    scores.append((advantage, p_ours, p_dps, p_diffpir, fname))

scores.sort(reverse=True)
top3 = scores[:3]
print("Top 3 (ours - DPS) + (ours - DiffPIR) total margin:")
for adv, p_o, p_d, p_dp, f in top3:
    print(f"  {f}: ours={p_o:.2f}  dps={p_d:.2f}  diffpir={p_dp:.2f}  total_margin={adv:.2f} dB")

# Build figure: 3 rows x 5 cols
methods = ["gt", "input", "dps", "diffpir", "ours"]
labels  = ["GT", "Input", "DPS", "DiffPIR", "Ours"]
nrows, ncols = 3, 5

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
fig.patch.set_facecolor('white')

for row, (adv, p_o, p_d, p_dp, fname) in enumerate(top3):
    for col, key in enumerate(methods):
        ax = axes[row][col]
        img_path = os.path.join(DIRS[key], fname)
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        # Column header on first row
        if row == 0:
            ax.set_title(labels[col], fontsize=14, fontweight='bold', pad=6)
        # PSNR annotations under each recon
        if col == 2:
            ax.set_xlabel(f"PSNR={p_d:.2f} dB", fontsize=9, labelpad=3)
        elif col == 3:
            ax.set_xlabel(f"PSNR={p_dp:.2f} dB", fontsize=9, labelpad=3)
        elif col == 4:
            ax.set_xlabel(f"PSNR={p_o:.2f} dB\n(Σmargin={adv:.2f} dB)", fontsize=9, labelpad=3, color='darkgreen')

plt.tight_layout(pad=0.5)
out_path = "/tmp/lc2762/diffusion-posterior-sampling-death/comparison_top3.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to {out_path}")
