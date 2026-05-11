"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import json

import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
import pandas as pd

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from mhcac.utils import save_to_csv, compute_metrics_for_tasks, aggregate_results, query_attention_visualization, expert_atttention_visualization, visualize_images_with_labels
import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.dist_utils import get_rank, is_main_process, init_distributed_mode
from model.lavis.common.logger import setup_logger

from local_config import WANDB_ENTITY, WANDB_PROJECT, VIS_ROOT
from model.lavis.common.registry import registry
from model.lavis.common.utils import now

# imports modules for registration
from model.lavis.common.optims import (
   LinearWarmupCosineLRScheduler,
   LinearWarmupStepLRScheduler,
)
from model.lavis.datasets.builders import *
from model.lavis.models import *
from model.lavis.processors import *
from model.lavis.runners import *
from model.lavis.tasks import *
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset


# python -m torch.distributed.run --standalone --nproc_per_node=2 -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2gpu.yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls


def main():
    registry.mapping['paths']['cache_root'] = '.'
    cfg = Config(parse_args())

    job_id = now()

    # Initialize distributed training (reads RANK, WORLD_SIZE, LOCAL_RANK set by torchrun)
    init_distributed_mode(cfg)

    # Bridge cfg.gpu into OmegaConf so runner_base can read it via cfg.run_cfg.gpu
    if hasattr(cfg, 'gpu'):
        OmegaConf.update(cfg.config, "run.gpu", cfg.gpu)
    if hasattr(cfg, 'distributed'):
        OmegaConf.update(cfg.config, "run.distributed", cfg.distributed)

    setup_seeds(cfg)
    setup_logger()

    # Only rank 0 logs to wandb; other ranks use disabled mode to silence any stray calls
    if is_main_process():
        try:
            wandb_run = wandb.init(
                project=cfg.run_cfg.get("project_name", WANDB_PROJECT),
                entity=WANDB_ENTITY if WANDB_ENTITY else None,
                name=cfg.run_cfg.run_name
            )
        except wandb.errors.UsageError:
            print("wandb: No API key found — logging disabled")
            wandb_run = wandb.init(mode="disabled")
    else:
        wandb_run = wandb.init(mode="disabled")

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    # Only MIMIC-CXR-JPG dataset
    datasets = {}
    datasets['mimic_cxr'] = {}

    datasets['mimic_cxr']['train'] = MIMIC_CXR_Dataset(
        vis_processor=None, text_processor=None,
        vis_root=VIS_ROOT,
        split="train", cfg=cfg, truncate=None
    )

    if not cfg.run_cfg.evaluate:
        datasets['mimic_cxr']['val'] = MIMIC_CXR_Dataset(
            vis_processor=None, text_processor=None,
            vis_root=VIS_ROOT,
            split="val", cfg=cfg, truncate=None
        )

    model = task.build_model(cfg)

    if not cfg.run_cfg.evaluate:
        runner = RunnerBase(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
        )
        runner.train(wandb_run)

    else:
        # Precompute Q-Former output embeddings for all images
        model.cuda()
        model.eval()

        split = 'train'
        dataset = 'mimic_cxr'
        batch_size = 64
        dataloader = DataLoader(datasets[dataset][split], batch_size=batch_size, shuffle=False, num_workers=cfg.run_cfg.num_workers)
        embeddings = {}
        cls_logits_dict = {}

        dataloader_len = len(dataloader)
        for i, batch in enumerate(tqdm(dataloader)):
            qformer_embs, _, cls_logits, attention_weights = model.forward_image(batch['image'].cuda(), None)

            for j, id in enumerate(batch['image_id']):
                if dataset == 'mimic_cxr':
                    dicom = datasets['mimic_cxr'][split].id_to_dicom[id.item()]
                embeddings[dicom] = qformer_embs[j].cpu().detach().numpy()
                cls_logits_dict[dicom] = cls_logits[j].cpu().detach().numpy()

        os.makedirs("pretraining/cls", exist_ok=True)
        os.makedirs("pretraining/embs", exist_ok=True)

        with open(f"pretraining/cls/{cfg.run_cfg.run_name}_cls_logits_{dataset}_{split}.pkl", "wb") as f:
            pickle.dump(cls_logits_dict, f)

        with open(f"pretraining/embs/{cfg.run_cfg.run_name}_embeddings_{dataset}_{split}.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        with open(f"pretraining/cls/{cfg.run_cfg.run_name}_meta_{dataset}_{split}.json", "w") as f:
            json.dump({"dataset": dataset, "split": split, "batch_size": batch_size, "model": cfg.run_cfg.run_name}, f)


if __name__ == "__main__":
    main()
