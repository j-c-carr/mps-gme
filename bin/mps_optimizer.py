import typing
import logging
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from utils import negative_overlap, normalize_state
from quimb.tensor.optimize import TNOptimizer
from typing import Optional, List


class MPSOptimizer():

    """Class for optimization algorithm of matrix product state. """

    def __init__(self, 
                 N: int, 
                 target: qtn.tensor_1d.MatrixProductState, 
                 bond_dim: int = 1, 
                 mps0: Optional[qtn.tensor_1d.MatrixProductState] = None):

        self.target = target

        if mps0 is None:
            mps0 = qtn.tensor_core.TensorNetwork.from_TN(qtn.MatrixProductState, target, bond_dim=bond_dim, dtype=np.float64)

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
                                     optimizer='L-BFGS-B')

    def optimize(self, num_steps: int):
        """Performs optimization for :num_steps:"""
        self.mps_optimal = self.optimizer.optimize(num_steps)

