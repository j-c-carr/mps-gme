import quimb
import typing
import logging
import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from typing import Optional
from mps_optimizer import MPSOptimizer
from known_states import PermutationInvariantState
from utils import negative_overlap, normalize_state

plt.style.use("highres")

# Fix seed to build the same tensors each time random is called
np.random.seed(0) 

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


if __name__ == "__main__":

    # Constants
    phys_dim = 2

    for N in range(10, 11):
        dims = tuple(phys_dim for _ in range(N))

        t_gme = []
        n_gme = []
        for k in range(N+1):

            psi = PermutationInvariantState(N, k, dims)

            MPSO = MPSOptimizer(N, psi.mps)

            MPSO.optimize(100)

            t_gme.append(psi.theoretical_gme)
            psi.numerical_gme = 1 + negative_overlap(psi.mps, MPSO.mps_optimal)
            n_gme.append(psi.numerical_gme)

        # plt.plot([i for i in range(N+1)], t_gme, "--", label=f"$N=${N}")
        # plt.plot([i for i in range(N+1)], n_gme, label=f"$N=${N}")

    # plt.legend()
    # plt.savefig("mps_gme_n.png", dpi=400)
        
