from functools import partial
import os
import argparse
import yaml

import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_dist = world_size > 1

    if is_dist:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # logger
    logger = get_logger()
    logger.info(f"[Rank {local_rank}/{world_size}] Device: {device}")
    start_time = time.time()

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"[Rank {local_rank}] Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"[Rank {local_rank}] Conditioning method: {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory — only rank 0 creates dirs, others wait
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    if local_rank == 0:
        os.makedirs(out_path, exist_ok=True)
        for img_dir in ['input', 'recon', 'label']:
            os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    if is_dist:
        dist.barrier()

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = IndexedDataset(get_dataset(**data_config, transforms=transform))

    if is_dist:
        dist_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=1, sampler=dist_sampler, num_workers=0)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])

    # Do Inference
    for ref_img, global_idx in loader:
        global_idx = global_idx.item()
        logger.info(f"[Rank {local_rank}] Inference for image {global_idx}")
        fname = str(global_idx).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        if measure_config['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            _measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            _sample_fn = partial(sample_fn, measurement_cond_fn=_measurement_cond_fn)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = _sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)

            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

    if local_rank == 0:
        elapsed = time.time() - start_time
        logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if is_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
