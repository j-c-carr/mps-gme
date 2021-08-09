import quimb
import typing
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from typing import Optional
from mps_optimizer import MPSOptimizer
from known_states import PermutationInvariantState, GHZState, WWbarSuperposition
from utils import negative_overlap, normalize_state, approximate_mps

plt.style.use("highres")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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


def make_parser() -> argparse.ArgumentParser:
    """Makes command line argument parser. Returns ArgumentParser"""

    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("datetime", help="datetime of exectution")
    parser.add_argument("root_dir", help="name project root directory")
    return parser.parse_args()


def sample():
    """Generates sample plot of GME"""

    for N in range(4, 11):
        dims = tuple(phys_dim for _ in range(N))

        t_gme = []
        n_gme = []
        for k in range(N+1):

            psi = GHZState(N, dims)

            # mps0 = approximate_mps(psi.mps, rank=1)

            MPSO = MPSOptimizer(N, psi.mps) #, optimizer=optimizer)

            MPSO.optimize(100)

            t_gme.append(psi.theoretical_gme)
            psi.numerical_gme = 1 + negative_overlap(psi.mps, MPSO.mps_optimal)
            n_gme.append(psi.numerical_gme)

        plt.plot([i for i in range(N+1)], t_gme, "--", c=COLORS[N-4])
        plt.plot([i for i in range(N+1)], n_gme, c=COLORS[N-4], label=f"$N=${N}")

    plt.legend()
    plt.ylabel(r"$E_{\sin^{2}}$")
    plt.xlabel("k")
    plt.title(r"$E_{\sin^{2}}$ for nGHZ State")
    plt.savefig(f"mps_gme_ghz.png", dpi=400)


def plot_w_wbar():
    """Recreates Fig (1) of (Wei, GME) paper, plotting GME as a function of s"""
    S = np.linspace(0, 1, 10)
    GME = np.empty(S.shape)
    for i, s in enumerate(S):
        print(f"s = {s}")
        w = WWbarSuperposition(s)
        GME[i] = w.theoretical_gme

    plt.scatter(S, GME, s=5)
    plt.xlabel("$s$")
    plt.ylabel("$E_{\sin^{2}}$")
    plt.savefig(f"{FIG_DIR}/{args.datetime}_wei_fig1.png", dpi=400)

if __name__ == "__main__":

    args = make_parser()
    FIG_DIR = args.root_dir + "/figures"

    plot_w_wbar()

        
