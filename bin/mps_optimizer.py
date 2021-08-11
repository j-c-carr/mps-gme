import typing
import logging
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from utils import negative_overlap, normalize_state, approximate_mps
from quimb.tensor.optimize import TNOptimizer
from typing import Optional, List


def init_logger(name: str,
                f: Optional[str] = None,
                level: Optional[int] = logging.INFO) -> logging.Logger:
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")

    if f is not None:
        file_handler = logging.FileHandler(f)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger
logger = init_logger(__name__)


class MPSOptimizer():

    """Class for optimization algorithm of matrix product state. """

    def __init__(self, 
                 N: int, 
                 target: qtn.tensor_1d.MatrixProductState, 
                 bond_dim: int = 1, 
                 optimizer: Optional[str] = "L-BFGS-B",
                 mps0: Optional[qtn.tensor_1d.MatrixProductState] = None):

        self.target = target
        logger.debug(f"Target:\n {target}")

        if mps0 is None:
            mps0 = qtn.MPS_rand_state(N, bond_dim, phys_dim=target.phys_dim())
            # qtn.tensor_core.TensorNetwork.from_TN(qtn.MatrixProductState, target, bond_dim=bond_dim, dtype=np.float64)
            logger.debug(f"mps0:\n {mps0}")

        assert type(mps0) == qtn.tensor_1d.MatrixProductState, \
                f"Expected matrix product state but mps0 is of type" \
                f" {type(mps0)}"

        assert mps0.dtype == target.dtype, \
                f"target is of type {target.dtype}" \
                f"but should be of type {mps0.dtype}."


        self.optimizer = TNOptimizer(mps0,
                                     loss_fn=negative_overlap,
                                     norm_fn=normalize_state,
                                     loss_constants={'target': self.target},
                                     autodiff_backend='tensorflow',
                                     optimizer=optimizer)

    def optimize(self, num_steps: int):
        """Performs optimization for :num_steps:"""
        self.mps_optimal = self.optimizer.optimize(num_steps)

