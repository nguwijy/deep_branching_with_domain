"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""
import logging

import tensorflow as tf

from .equation import AllenCahn
from .solver import BSDESolver


class Dict2Class(object):
    """ Turns dict into a class """
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


eqn_config = Dict2Class({
    "_comment": "Allen-Cahn equation",
    "eqn_name": "AllenCahn",
    "total_time": 0.3,
    "dim": 1,
    "num_time_interval": 20,
})
net_config = Dict2Class({
    "y_init_range": [-1, 0],
    "num_hiddens": [110, 110],
    "lr_values": [5e-4, 5e-4],
    "lr_boundaries": [2000],
    "num_iterations": 4000,
    "batch_size": 64,
    "valid_size": 256,
    "logging_frequency": 100,
    "dtype": "float64",
    "verbose": True,
})
config = Dict2Class({
    "eqn_config": eqn_config,
    "net_config": net_config,
})


def main(total_time=0.3):
    eqn_config.total_time = total_time
    bsde = AllenCahn(eqn_config)
    tf.keras.backend.set_floatx(net_config.dtype)

    formatter = "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=formatter,
    )
    logging.info('Begin to solve %s ' % eqn_config.eqn_name)
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
    return training_history


if __name__ == '__main__':
    _ = main()
