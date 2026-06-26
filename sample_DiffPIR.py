"""
DiffPIR sampler — "Denoising Diffusion Models for Plug-and-Play Image Restoration"
arXiv 2305.08995

Each timestep:
  1. x0_pred  = (xt + (1 - alpha_bar_t) * score) / sqrt(alpha_bar_t)
               (= pred_xstart from p_mean_variance)
  2. x0_hat   = proximal data-consistency correction
               argmin_x  ||y - H(x)||^2 + rho_t * ||x - x0_pred||^2
  3. eps_hat  = (xt - sqrt(alpha_bar_t) * x0_hat) / sqrt(1 - alpha_bar_t)
  4. x_{t-1}  = sqrt(alpha_bar_{t-1}) * x0_hat
              + sqrt(1 - alpha_bar_{t-1}) * (sqrt(1-zeta)*eps_hat + sqrt(zeta)*eps)

Launch on 8 GPUs:
  torchrun --nproc_per_node=8 sample_DiffPIR.py \
      --model_config     configs/model_config.yaml \
      --diffusion_config configs/diffusion_config.yaml \
      --task_config      configs/inpainting_config.yaml \
      --save_dir         results/diffpir \
      --num_steps 100 --lambda_ 7.0 --max_images 100
"""

import os
import argparse
import yaml
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler, extract_and_expand
from data.dataloader import get_dataset
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


def proximal_step(x0_pred, y, operator, rho_t, operator_name, mask=None):
    """
    Closed-form proximal step for inpainting; gradient step for other operators.

    For inpainting (H = diagonal binary mask M):
        x_hat = (M * y + rho_t * x0_pred) / (M + rho_t)

    For general linear operators (one gradient step):
        x_hat = x0_pred - (1/rho_t) * H^T (H(x0_pred) - y)
    """
    if operator_name == 'inpainting':
        x0_hat = (mask * y + rho_t * x0_pred) / (mask + rho_t)
    else:
        residual = operator.forward(x0_pred) - y
        x0_hat = x0_pred - (1.0 / rho_t) * operator.transpose(residual)
    return x0_hat


def diffpir_sample(sampler, model, x_start, y, operator, operator_name,
                   lambda_=1.0, sigma_n=0.05, zeta=1.0, device='cuda', mask=None, rank=0):
    img = x_start
    T = sampler.num_timesteps

    pbar = tqdm(list(range(T))[::-1], desc="DiffPIR", disable=(rank != 0))
    for idx in pbar:
        t = torch.tensor([idx] * img.shape[0], device=device)

        # 1. Predict x0 from xt
        with torch.no_grad():
            out = sampler.p_mean_variance(model, img, t)
        x0_pred = out['pred_xstart']

        alpha_bar_t      = extract_and_expand(sampler.alphas_cumprod,     t, img)
        alpha_bar_t_prev = extract_and_expand(sampler.alphas_cumprod_prev, t, img)
        sqrt_ab_t        = alpha_bar_t.sqrt()
        sqrt_1mab_t      = (1.0 - alpha_bar_t).sqrt()

        # 2. Proximal data-consistency correction
        # rho_t = lambda * sigma_n^2 / sigma_bar_t^2,  sigma_bar_t^2 = (1-alpha_bar) / alpha_bar
        rho_t  = lambda_ * (sigma_n ** 2) * alpha_bar_t / (1.0 - alpha_bar_t)
        x0_hat = proximal_step(x0_pred, y, operator, rho_t, operator_name, mask=mask)

        # 3. Recompute effective epsilon
        eps_hat = (img - sqrt_ab_t * x0_hat) / sqrt_1mab_t

        # 4. DDIM-like update
        noise     = torch.randn_like(img)
        direction = (1.0 - zeta).sqrt() * eps_hat + zeta.sqrt() * noise
        img = alpha_bar_t_prev.sqrt() * x0_hat + (1.0 - alpha_bar_t_prev).sqrt() * direction

    # Paste known pixels back — rho→∞ at t→0 makes the proximal step ignore y
    # in the final timesteps, so we restore consistency explicitly.
    if operator_name == 'inpainting' and mask is not None:
        mask3 = mask.expand_as(img).bool()
        img[mask3] = y[mask3]

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config',     type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config',      type=str, required=True)
    parser.add_argument('--save_dir',         type=str, default='./results')
    parser.add_argument('--num_steps',        type=int,   default=100)
    parser.add_argument('--lambda_',          type=float, default=1.0)
    parser.add_argument('--zeta',             type=float, default=1.0)
    parser.add_argument('--max_images',       type=int,   default=None)
    args = parser.parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist    = world_size > 1

    if is_dist:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    logger = get_logger()

    if local_rank == 0:
        logger.info(f"World size: {world_size}  |  steps={args.num_steps}"
                    f"  lambda={args.lambda_}  zeta={args.zeta}")

    start_time = time.time()

    # ── Configs ───────────────────────────────────────────────────────────────
    model_config     = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config      = load_yaml(args.task_config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = create_model(**model_config).to(device).eval()

    # ── Operator & noiser ─────────────────────────────────────────────────────
    measure_config = task_config['measurement']
    operator_name  = measure_config['operator']['name']
    operator       = get_operator(device=device, **measure_config['operator'])
    noiser         = get_noise(**measure_config['noise'])
    sigma_n        = float(measure_config['noise'].get('sigma', 0.05))

    # ── Sampler ───────────────────────────────────────────────────────────────
    diffusion_config['timestep_respacing'] = str(args.num_steps)
    sampler = create_sampler(**diffusion_config)
    zeta    = torch.tensor(args.zeta, device=device)

    # ── Output dirs (rank 0 only, then barrier) ───────────────────────────────
    out_path = os.path.join(args.save_dir, operator_name)
    if local_rank == 0:
        os.makedirs(out_path, exist_ok=True)
        for sub in ['input', 'recon', 'label']:
            os.makedirs(os.path.join(out_path, sub), exist_ok=True)
    if is_dist:
        dist.barrier()

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    full_dataset = IndexedDataset(get_dataset(**data_config, transforms=transform))

    # Limit to first max_images before splitting across GPUs
    if args.max_images is not None:
        indices = list(range(min(args.max_images, len(full_dataset))))
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    if is_dist:
        sampler_dist = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=False
        )
        loader = DataLoader(dataset, batch_size=1, sampler=sampler_dist, num_workers=0)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if operator_name == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])

    # ── Inference ─────────────────────────────────────────────────────────────
    for item, global_idx in loader:
        global_idx = global_idx.item()
        logger.info(f"[Rank {local_rank}] Image {global_idx}")
        fname   = str(global_idx).zfill(5) + '.png'
        ref_img = item.to(device)

        if operator_name == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(1)
            y    = operator.forward(ref_img, mask=mask)
            y_n  = mask * noiser(y)
            sample = diffpir_sample(
                sampler, model,
                x_start=torch.randn_like(ref_img),
                y=y_n, operator=operator, operator_name=operator_name,
                lambda_=args.lambda_, sigma_n=sigma_n, zeta=zeta, device=device,
                mask=mask, rank=local_rank,
            )
        else:
            y   = operator.forward(ref_img)
            y_n = noiser(y)
            sample = diffpir_sample(
                sampler, model,
                x_start=torch.randn_like(ref_img),
                y=y_n, operator=operator, operator_name=operator_name,
                lambda_=args.lambda_, sigma_n=sigma_n, zeta=zeta, device=device,
                rank=local_rank,
            )

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

    if local_rank == 0:
        elapsed = time.time() - start_time
        logger.info(f"Done. Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if is_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
