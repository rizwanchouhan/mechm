import argparse  # Module for parsing command-line arguments
import os  # Module provides a portable way of using operating system dependent functionality
import random  # Module to generate random numbers

import numpy as np  # Numerical computing library
import torch  # PyTorch deep learning framework
import torch.backends.cudnn as cudnn  # Interface to NVIDIA CuDNN
import wandb  # Library for experiment tracking and visualization

import gpt.tasks as tasks  # Module containing task definitions
from gpt.common.config import Config  # Configuration class
from gpt.common.dist_utils import get_rank, init_distributed_mode  # Utilities for distributed training
from gpt.common.logger import setup_logger  # Logger setup function
from gpt.common.optims import (  # Optimization schedulers
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from gpt.common.registry import registry  # Registry for dynamically loading classes
from gpt.common.utils import now  # Utility function for getting current timestamp
from gpt.datasets.builders import *  # Builders for datasets
from gpt.models import *  # Model definitions
from gpt.processors import *  # Data processors
from gpt.runners import *  # Training runners
from gpt.tasks import *  # Task definitions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
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
    """Setup random seeds for reproducibility."""
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

def main():
    """Main function to run training."""
    job_id = now()
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)  # Initialize distributed training settings
    setup_seeds(cfg)  # Setup random seeds
    setup_logger()  # Setup logger
    cfg.pretty_print()  # Print configuration details

    task = functions.setup_task(cfg)  # Setup task based on configuration
    datasets = task.build_datasets(cfg)  # Build datasets based on configuration
    model = task.build_model(cfg)  # Build model based on configuration

    if cfg.run_cfg.wandb_log:
        wandb.login()  # Login to Weights & Biases
        wandb.init(project="MECHM", name=cfg.run_cfg.job_name)  # Initialize experiment tracking
        wandb.watch(model)  # Watch model for tracking gradients and parameters

    runner = get_runner_class(cfg)(  # Instantiate runner for training
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()  # Start training

if __name__ == "__main__":
    main()
