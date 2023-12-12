from yacs.config import CfgNode as CN

from graphgym.register import register_config


def set_cfg_mires(cfg):
    r'''
    This function sets the default config value for MI Reservoir Experiment
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # MI Reservoir Experiment
    # ----------------------------------------------------------------------- #

    # MI Reservoir
    cfg.mires = cfg.example_group = CN()

    # Data Dir
    cfg.mires.dir = "/storage/lpasa/Dataset/MI_Reservoir_Dataset"

    # Layers
    #cfg.mires.layers_post_mp = 1

    # MI RUN == cfg.seed
    # cfg.mires.run

    # MI Matrix, A or L
    cfg.mires.RES = 'A'

    # MI k 3 to 9
    cfg.mires.k = 3

    # MI delta 1 to 3
    cfg.mires.delta = 1

    # MI nunits 15, 30, 60, 100
    cfg.mires.nunits = 15

register_config('mires', set_cfg_mires)