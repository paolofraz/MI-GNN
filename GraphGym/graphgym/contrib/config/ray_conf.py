from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_ray(cfg):
	r'''
	This function sets the default config value for customized options
	:return: customized configuration use by the experiment.
	'''

	# ----------------------------------------------------------------------- #
	# Customized options
	# ----------------------------------------------------------------------- #

	# example argument group
	cfg.ray = CN()

	# then argument can be specified within the group
	cfg.ray.RES = ['A']
	cfg.ray.k = [3]
	cfg.ray.nunits = [100]
	cfg.ray.delta = [1]

register_config('ray', set_cfg_ray)