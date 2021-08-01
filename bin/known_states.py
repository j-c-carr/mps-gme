import typing
import logging
import itertools
import numpy as np
from math import factorial
from typing import Optional, List

import quimb
import quimb.tensor as qtn


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

def binomial(n, k):
    return factorial(n) / (factorial(k) * factorial(n-k))

class PermutationInvariantState():

    """Wrapper for PermutationInvariantState. Analytic solutions are based in (Wei, 2003)"""

    def __init__(self, n: int, k: int, dims: List[int]):
       
        assert k <= n and n > 0

        self.dims = dims
        self.state = self.create_state(n, k)
        self.theoretical_gme = self.compute_theoretical_gme(n, k)

        self.mps = qtn.MatrixProductState.from_dense(quimb.qu(self.state, 
                                                     qtype="ket",
                                                     dtype=self.state.dtype), 
                                                     self.dims)


    def create_state(self, n: int, k: int) -> np.ndarray:
        """Creates a 1-D wavefunction"""

        c = np.sqrt(1 / binomial(n, k))
        logger.debug(f"coefficient: {c}")

        psi = np.zeros(tuple([2 for _ in range(n)]))

        I = [0 if i < k else 1 for i in range(n)]
        count = 0
        for ix in itertools.permutations(I):
            count += 1
            psi[ix] = max(psi[ix], c)

        logger.debug(f"There are {count} permutations, should be {binomial(n,k)}")

        # TODO: make sure flattening is consistent with vector identification
        psi = psi.flatten()

        assert abs(np.dot(psi, psi.T) - 1.0) < 1e-8, \
                f"Expected unit norm but psi has norm {np.dot(psi, psi.T)}."

        return psi.astype(np.float64)


    def compute_theoretical_gme(self, n: int, k: int) -> float:
        """Computes geometric measure of entanglement from (Wei, 2003)"""
        # Maximal eigenvalue associated to the state
        l_max = np.sqrt(binomial(n, k)) * np.power(k/n, k/2) * np.power((n-k)/n, (n-k)/2)

        return 1 - l_max**2
