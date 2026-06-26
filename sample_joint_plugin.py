"""
Joint diffusion posterior sampling with plug-in conditioning term (inpainting).

The model s_θ is a standard 3-channel FFHQ diffusion model.  For inpainting,
Y_0 = mask ⊙ X_0 is the observed (masked) region.  The plug-in correction
(Eq. 18) is applied only over the observed pixels:

  c_y(t; Y_0, X_t) = mask ⊙ (Y_0 - X_t) / σ²(t),   σ²(t) = 1 - ᾱ_t

In epsilon-space this becomes:

  ε_corr = ε_θ(X_t, t) - σ_t · c_y

Usage:
  python sample_joint_plugin.py \
      --model_config configs/joint_model_config.yaml \
      --diffusion_config configs/diffusion_config.yaml \
      --task_config configs/inpainting_config.yaml \
      --plugin_scale 1.0 \
      --save_dir results/joint_plugin
"""

import os
import time
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from guided_diffusion.unet import UNetModel
from guided_diffusion.gaussian_diffusion import (
    create_sampler,
    get_named_beta_schedule,
    extract_and_expand,
)
from guided_diffusion.measurements import get_noise, get_operator
from data.dataloader import get_dataset
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_joint_model(cfg: dict, device: torch.device) -> nn.Module:
    """Load ffhq_10m.pt (3-channel) as the joint score model."""
    image_size = cfg["image_size"]
    channel_mult_str = cfg.get("channel_mult", "")
    if channel_mult_str == "":
        _table = {512: (0.5, 1, 1, 2, 2, 4, 4),
                  256: (1, 1, 2, 2, 4, 4),
                  128: (1, 1, 2, 3, 4),
                  64:  (1, 2, 3, 4)}
        channel_mult = _table[image_size]
    else:
        channel_mult = tuple(int(x) for x in channel_mult_str.split(","))

    attention_ds = []
    ar = cfg.get("attention_resolutions", 16)
    for res in (ar if isinstance(ar, str) else str(ar)).split(","):
        attention_ds.append(image_size // int(res))

    in_channels  = cfg.get("in_channels", 3)
    learn_sigma  = cfg.get("learn_sigma", True)
    out_channels = in_channels if not learn_sigma else 2 * in_channels

    model = UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=cfg["num_channels"],
        out_channels=out_channels,
        num_res_blocks=cfg["num_res_blocks"],
        attention_resolutions=tuple(attention_ds),
        dropout=cfg.get("dropout", 0.0),
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=cfg.get("use_checkpoint", False),
        use_fp16=cfg.get("use_fp16", False),
        num_heads=cfg.get("num_heads", 4),
        num_head_channels=cfg.get("num_head_channels", 64),
        num_heads_upsample=cfg.get("num_heads_upsample", -1),
        use_scale_shift_norm=cfg.get("use_scale_shift_norm", True),
        resblock_updown=cfg.get("resblock_updown", True),
        use_new_attention_order=cfg.get("use_new_attention_order", False),
    )

    ckpt = torch.load(cfg["model_path"], map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Plug-in wrapper
# ---------------------------------------------------------------------------

class JointPluginModel(nn.Module):
    """
    Wraps the 3-channel joint score model and injects the plug-in conditioning
    term over the observed (masked) pixels at every forward pass.

    For inpainting, Y_0 = mask ⊙ X_0.  The correction is:
      c_y = mask ⊙ (Y_0 - X_t) / σ²(t)
      ε_corr = ε_θ(X_t, t) − σ_t · c_y · plugin_scale

    `t` here is the *original* 1000-step index (SpacedDiffusion remaps it
    before calling this forward), so we index into `orig_abar` directly.
    """

    def __init__(
        self,
        model: nn.Module,
        y_0: torch.Tensor,
        mask: torch.Tensor,
        original_alphas_cumprod: np.ndarray,
        plugin_scale: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("y_0", y_0)
        self.register_buffer("mask", mask)
        self.register_buffer(
            "orig_abar", torch.from_numpy(original_alphas_cumprod).float()
        )
        self.plugin_scale = plugin_scale

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        raw = self.model(x_t, t, **kwargs)

        # Split eps from learned-variance part if present (6-ch → 3+3)
        has_var = raw.shape[1] == 2 * x_t.shape[1]
        if has_var:
            eps, var = torch.split(raw, x_t.shape[1], dim=1)
        else:
            eps = raw

        # σ²(t) = 1 - ᾱ_t  (t is the original 1000-step index)
        abar           = self.orig_abar[t].view(-1, 1, 1, 1).expand_as(x_t)
        one_minus_abar = 1.0 - abar
        sigma_t        = one_minus_abar.sqrt()

        # Plug-in correction over observed pixels only
        c_y      = self.mask * (self.y_0 - x_t) / one_minus_abar
        eps_corr = eps - sigma_t * c_y * self.plugin_scale

        if has_var:
            return torch.cat([eps_corr, var], dim=1)
        return eps_corr


# ---------------------------------------------------------------------------
# Langevin corrector
# ---------------------------------------------------------------------------

def langevin_step(
    diffusion,
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    snr: float,
) -> torch.Tensor:
    """One annealed Langevin step at noise level t (SNR-based step size).

    score ≈ -ε_θ / σ_t,  step_size = (snr * ||noise|| / ||score||)² * 2
    Plug-in correction is already baked into model (JointPluginModel).
    """
    with torch.no_grad():
        out = diffusion.p_mean_variance(model, x, t)

    # ε from pred_x0: ε = (x_t − √ᾱ_t · x̂_0) / √(1 − ᾱ_t)
    abar_t  = extract_and_expand(diffusion.alphas_cumprod, t, x)
    sigma_t = (1.0 - abar_t).sqrt().clamp(min=1e-8)
    eps     = (x - abar_t.sqrt() * out["pred_xstart"]) / sigma_t
    score   = -eps / sigma_t

    # SNR-based step size (Song et al. 2021, Algorithm 5)
    grad_norm  = score.view(x.shape[0], -1).norm(dim=-1).mean()
    noise_norm = float(x[0].numel()) ** 0.5
    step_size  = (snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2.0

    noise = torch.randn_like(x)
    return (x + step_size * score + (2.0 * step_size) ** 0.5 * noise).detach()


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------

def joint_sample_loop(
    diffusion,
    model: nn.Module,
    x_start: torch.Tensor,
    record: bool,
    save_root: str,
    device: torch.device,
    n_corrector_steps: int = 0,
    corrector_snr: float = 0.16,
) -> torch.Tensor:
    """Predictor-Corrector reverse diffusion.

    Each iteration costs (1 + n_corrector_steps) NFE.  n_predictor_steps is
    set by the caller so that total NFE matches the pure-predictor baseline.
    Order: K Langevin corrector steps at x_t, then one predictor step to x_{t-1}.
    """
    x = x_start

    pbar = tqdm(list(range(diffusion.num_timesteps))[::-1])
    for idx in pbar:
        t = torch.full((x.shape[0],), idx, device=device, dtype=torch.long)

        # Corrector: refine x_t before the predictor step
        for _ in range(n_corrector_steps):
            x = langevin_step(diffusion, model, x, t, corrector_snr)

        # Predictor: x_t → x_{t-1}
        out = diffusion.p_sample(model=model, x=x, t=t)
        x   = out["sample"].detach()

        if record and idx % 10 == 0:
            plt.imsave(
                os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png"),
                clear_color(x),
            )

    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config",    type=str, default="configs/joint_model_config.yaml")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_config.yaml")
    parser.add_argument("--task_config",     type=str, required=True)
    parser.add_argument("--gpu",             type=int, default=0)
    parser.add_argument("--save_dir",        type=str, default="./results/joint_plugin")
    parser.add_argument(
        "--plugin_scale", type=float, default=1.0,
        help="Scale factor η multiplied onto c_y (1.0 = full plug-in correction)",
    )
    parser.add_argument(
        "--n_corrector_steps", type=int, default=1,
        help="Langevin corrector steps per predictor step. "
             "Predictor steps = total_nfe // (1 + n_corrector_steps). "
             "Set to 0 for pure-predictor (matches sample_condition NFE exactly).",
    )
    parser.add_argument(
        "--corrector_snr", type=float, default=0.16,
        help="SNR coefficient for Langevin step-size (Song et al. 2021 default: 0.16).",
    )
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist = world_size > 1

    if is_dist:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    logger = get_logger()
    logger.info(f"[Rank {local_rank}/{world_size}] Device: {device}")
    start_time = time.time()

    model_config    = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config     = load_yaml(args.task_config)

    model = build_joint_model(model_config, device)

    # Split NFE budget between predictor and corrector.
    # total_nfe = timestep_respacing (e.g. 1000), matching sample_condition.
    # n_predictor_steps * (1 + n_corrector_steps) == total_nfe
    total_nfe = int(diffusion_config["timestep_respacing"])
    n_corrector = args.n_corrector_steps
    n_pred = total_nfe // (1 + n_corrector)
    diffusion_config = dict(diffusion_config)          # don't mutate the original
    diffusion_config["timestep_respacing"] = n_pred
    logger.info(
        f"NFE budget: {total_nfe}  |  predictor steps: {n_pred}  |  "
        f"corrector steps/iter: {n_corrector}  (total NFE: {n_pred * (1 + n_corrector)})"
    )

    # Diffusion sampler
    diffusion = create_sampler(**diffusion_config)

    # Original (full-schedule) ᾱ_t needed inside the plug-in wrapper.
    # SpacedDiffusion remaps spaced indices to original indices before calling
    # the model, so we must index the *original* 1000-step schedule here.
    orig_betas             = get_named_beta_schedule(
        diffusion_config["noise_schedule"], diffusion_config["steps"]
    )
    original_alphas_cumprod = np.cumprod(1.0 - orig_betas)

    # Measurement operator (used to create Y_0 from the reference image)
    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser   = get_noise(**measure_config["noise"])
    logger.info(
        f"Operator: {measure_config['operator']['name']} / "
        f"Noise: {measure_config['noise']['name']}"
    )

    # Output dirs — only rank 0 creates, others wait
    out_path = os.path.join(args.save_dir, measure_config["operator"]["name"])
    if local_rank == 0:
        for d in ("input", "recon", "label"):
            os.makedirs(os.path.join(out_path, d), exist_ok=True)
    if is_dist:
        dist.barrier()

    # Dataset
    data_config = task_config["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = IndexedDataset(get_dataset(**data_config, transforms=transform))

    if is_dist:
        dist_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=1, sampler=dist_sampler, num_workers=0)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    mask_gen = mask_generator(**measure_config["mask_opt"])

    for ref_img, global_idx in loader:
        global_idx = global_idx.item()
        logger.info(f"[Rank {local_rank}] Inference for image {global_idx}")
        fname   = str(global_idx).zfill(5) + ".png"
        ref_img = ref_img.to(device)

        # Build Y_0 = mask ⊙ X_0 (observed pixels)
        mask = mask_gen(ref_img)
        mask = mask[:, 0:1, :, :]           # [B, 1, H, W], binary
        y_0  = noiser(operator.forward(ref_img, mask=mask))

        # Wrap the model: injects plug-in correction over observed pixels
        wrapped = JointPluginModel(
            model=model,
            y_0=y_0,
            mask=mask,
            original_alphas_cumprod=original_alphas_cumprod,
            plugin_scale=args.plugin_scale,
        ).to(device)

        x_start = torch.randn(ref_img.shape, device=device)

        sample = joint_sample_loop(
            diffusion=diffusion,
            model=wrapped,
            x_start=x_start,
            record=False,
            save_root=out_path,
            device=device,
            n_corrector_steps=n_corrector,
            corrector_snr=args.corrector_snr,
        )

        plt.imsave(os.path.join(out_path, "input", fname), clear_color(y_0))
        plt.imsave(os.path.join(out_path, "label", fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, "recon", fname), clear_color(sample))

    if local_rank == 0:
        elapsed = time.time() - start_time
        logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
