# Diffusion Posterior Sampling for Image Inpainting — DPS / DiffPIR / Joint Plug-in

This repository extends [Diffusion Posterior Sampling (DPS, ICLR 2023)](https://github.com/DPS2022/diffusion-posterior-sampling)
to compare several diffusion-based solvers on **noisy image inpainting** over the
FFHQ-256 and ImageNet-256 datasets.

Three samplers are implemented and benchmarked:

| Method | Script | Idea |
|--------|--------|------|
| **DPS** (baseline) | `sample_condition.py` | Posterior sampling with the manifold-constrained gradient. |
| **DiffPIR** | `sample_DiffPIR.py` | Plug-and-play restoration with a proximal data-consistency step ([arXiv 2305.08995](https://arxiv.org/abs/2305.08995)). |
| **Joint Plug-in** (ours) | `sample_joint_plugin.py` | A joint score with a plug-in conditioning correction `c_y(t) = mask ⊙ (Y₀ − Xₜ) / σ²(t)`, optionally combined with Langevin corrector steps. |

Reconstructions are scored with **LPIPS, FID, JFID and PSNR** via `eval_metrics.py`.

<br />

## Prerequisites
- python 3.8+
- pytorch 1.11.0 (CUDA 11.3)
- See `requirements.txt`. Evaluation additionally needs `piq` and `lpips`; the
  download scripts need `datasets` (HuggingFace).

It is fine to use a different CUDA / pytorch combination as long as they match.

<br />

## Getting started

### 1) Clone the repository

```
git clone https://github.com/Libo999-chen/close_sde_ode
cd close_sde_ode
```

### 2) Clone the external motion-blur dependency

```
git clone https://github.com/LeviBorodenko/motionblur motionblur
```

### 3) Install dependencies

```
conda create -n dps python=3.8
conda activate dps
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

### 4) Download pretrained checkpoints

From the original DPS [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing),
download `ffhq_10m.pt` (and the ImageNet checkpoint) into `./models/`:

```
mkdir -p models
mv {DOWNLOAD_DIR}/ffhq_10m.pt    ./models/
mv {DOWNLOAD_DIR}/imagenet256.pt ./models/
```

> `models/`, datasets and `results/` are git-ignored — they are **not** stored in this repo.

### 5) Download evaluation data

```
# FFHQ-256 validation images  -> data/samples/
python download_ffhq256.py --num_images 1000

# ImageNet-256 validation images -> data/imagenet256/
python download_imagenet256.py --num_images 1000
```

Both use public HuggingFace datasets and require no authentication.

<br />

## Running the samplers

The task is configured by `configs/inpainting_config.yaml` (random mask, 30–70%
missing pixels, Gaussian noise σ = 0.05).

**DPS baseline**

```
python sample_condition.py \
    --model_config     configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config      configs/inpainting_config.yaml
```

**DiffPIR** (supports multi-GPU via `torchrun`)

```
torchrun --nproc_per_node=8 sample_DiffPIR.py \
    --model_config     configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config      configs/inpainting_config.yaml \
    --save_dir         results/diffpir \
    --num_steps 100 --lambda_ 7.0 --max_images 100
```

**Joint Plug-in (ours)**

```
python sample_joint_plugin.py \
    --model_config     configs/joint_model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config      configs/inpainting_config.yaml \
    --plugin_scale 1.0 \
    --n_corrector_steps 1 \
    --save_dir results/joint_plugin
```

Key options: `--plugin_scale` (strength of the plug-in correction),
`--n_corrector_steps` / `--corrector_snr` (Langevin corrector).

<br />

## Evaluation

Each results directory must contain `recon/` and `label/` subfolders.

```
# single method
python eval_metrics.py --results_dir ./results/inpainting

# compare two methods
python eval_metrics.py \
    --results_dir ./results/inpainting              --name "DPS" \
    --results_dir ./results/joint_plugin/inpainting --name "Joint-Plugin"
```

Metrics (LPIPS, FID, JFID, PSNR) follow the protocol of
[arXiv 2407.01521](https://arxiv.org/abs/2407.01521).

Generate a qualitative side-by-side figure (GT / input / DPS / DiffPIR / ours):

```
python plot_comparison.py
```

<br />

## Repository layout

```
sample_condition.py      DPS baseline sampler
sample_DiffPIR.py        DiffPIR plug-and-play sampler
sample_joint_plugin.py   Joint plug-in sampler (ours)
eval_metrics.py          LPIPS / FID / JFID / PSNR evaluation
plot_comparison.py       Qualitative comparison grid
download_ffhq256.py      FFHQ-256 downloader
download_imagenet256.py  ImageNet-256 downloader
configs/                 model / diffusion / task configs
guided_diffusion/        diffusion model, measurements, conditioning
```

<br />

## Acknowledgements & Citation

This work builds on the official DPS implementation. If you use it, please cite
the original paper:

```
@inproceedings{chung2023diffusion,
  title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
  author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```
