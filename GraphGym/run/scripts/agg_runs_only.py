import logging
import os
import sys

import torch
from torch_geometric import seed_everything

# Add upper directory to sys path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# Get the second upper directory
second_upper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add the second upper directory to the system path
sys.path.append(second_upper_dir)

from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    ## set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
