print("Starting Ray Job!")

import logging
import os
import sys

import torch
from torch_geometric import seed_everything

# Add upper directory to sys path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from graphgym.cmd_args import parse_args
from graphgym.config import get_fname
from yacs.config import CfgNode

import ray
from ray import tune
from ray.air import RunConfig, ScalingConfig, FailureConfig, session
from ray.tune.tuner import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
#from ray.tune.search.bayesopt import BayesOptSearch
#from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch

if __name__ == '__main__':
	# Initialize Ray
	ray.init(address="auto", _redis_password=os.environ["redis_password"], runtime_env={"working_dir": parent_dir}) #os.environ["ip_head"] f"ray://{os.environ['ip_head']}"
	print("Nodes in the Ray cluster:")
	print(ray.nodes())

	# Load cmd line args
	args = parse_args()

	# Define Search Space
	param_space = {
			"mires": {"RES": tune.choice(["A", "L"]), "k": tune.choice([3, 4, 5, 6]), "delta": tune.choice([1, 2, 3]),
					  "nunits": tune.choice([50, 100, 150, 200])},
			"gnn": {"layers_post_mp": tune.randint(1, 8), "dim_inner": tune.qrandint(40, 400, 10)},
			"optim": {"base_lr": tune.loguniform(1e-4, 1e-2), "weight_decay": tune.loguniform(1e-6, 1e-2)}}


	# Define trainable
	def trainable(config):

		from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir, get_fname
		from graphgym.loader import create_dataset, create_loader
		from graphgym.logger import create_logger, setup_printing
		from graphgym.model_builder import create_model
		from graphgym.optimizer import create_optimizer, create_scheduler
		from graphgym.register import train_dict
		from graphgym.train import train
		from graphgym.utils.agg_runs_ray import agg_runs # RAY EDIT
		from graphgym.utils.comp_budget import params_count
		from graphgym.utils.device import auto_select_device

		# Load general config file
		load_cfg(cfg, args)

		# Combine it with RayTune
		ray_Cfg = CfgNode(config)
		#print(f"Ray cfg: {ray_Cfg}")
		#print(f"Gym cfg before merge: {cfg}")
		# Merge configurations
		cfg.merge_from_other_cfg(ray_Cfg)
		#print(f"Gym cfg after merge: \n{cfg}")

		#  Set configurations
		set_out_dir(cfg.out_dir, session.get_trial_id())
		# Set Pytorch environment
		# torch.set_num_threads(cfg.num_threads)
		dump_cfg(cfg)
		# Repeat for different random seeds
		for i in range(args.repeat):
			set_run_dir(cfg.out_dir)
			setup_printing()
			# Set configurations for each run
			cfg.seed = cfg.seed + 1
			seed_everything(cfg.seed)
			auto_select_device()
			# Set machine learning pipeline
			datasets = create_dataset()
			loaders = create_loader(datasets)
			loggers = create_logger()
			model = create_model()
			optimizer = create_optimizer(model.parameters())
			scheduler = create_scheduler(optimizer)
			# Print model info
			logging.info(model)
			logging.info(cfg)
			cfg.params = params_count(model)
			logging.info('Num parameters: %s', cfg.params)
			# Start training
			if cfg.train.mode == 'standard':
				train(loggers, loaders, model, optimizer, scheduler)
			else:
				train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

		# Aggregate results from different seeds
		agg_runs(cfg.out_dir, cfg.metric_best)


	# trainable = tune.with_parameters(trainable, loggers=loggers, loaders=loaders, model=model, optimizer=optimizer, scheduler=)

	# sched = ASHAScheduler(max_t=500, grace_period=50, reduction_factor=2)

	resources_per_trial = {"gpu": 1}  # set this for GPUs
	tuner = tune.Tuner(tune.with_resources(trainable, resources=resources_per_trial),
					   tune_config=TuneConfig(metric="val_accuracy", mode="max", #scheduler=sched,
												   search_alg=OptunaSearch(), num_samples=300, reuse_actors=False),
					   run_config=RunConfig(name=get_fname(args.cfg_file), storage_path='/home/frazzetp/MI-GNN/GraphGym/run/results/SVD',
												log_to_file=True, failure_config=FailureConfig()),
					   param_space=param_space)
	results = tuner.fit()

	best_result = results.get_best_result()  # Get best result object
	best_config = best_result.config  # Get best trial's hyperparameters
	best_logdir = best_result.log_dir  # Get best trial's logdir
	best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
	best_metrics = best_result.metrics  # Get best trial's last results

	print("Best config is:", best_config)
	print("Best metrics are:", best_metrics)
	print("Best checkpoint is:", best_checkpoint)