import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from mps_optimizer import MPSOptimizer
from known_states import (QuantumState, 
                          SymmetricQuditState,
                          PermutationInvariantState,
                          GHZState,
                          WWbarSuperposition)
from utils import negative_overlap, normalize_state, approximate_mps


plt.style.use("highres")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def recreate_wei_fig_1(filename: str,
                       steps: Optional[int] = 100):
    """Plots the numerical GME vs s for the WWbarSuperposition state."""

    S = np.linspace(0, 1, steps)
    n_gme = []
    for s in S:
        psi = WWbarSuperposition(s)
        MPSO = MPSOptimizer(3, psi.mps)

        MPSO.optimize(100)

        psi.numerical_gme = 1 + negative_overlap(psi.mps, MPSO.mps_optimal)

        n_gme.append(psi.numerical_gme)

    plt.scatter(S, n_gme, s=5)
    plt.ylabel(r"$E_{\sin^{2}}$")
    plt.xlabel("s")
    plt.title(r"$E_{\sin^{2}}$ of $\sqrt{s}|W\rangle + \sqrt{1-s}|\tilde{W}\rangle$")
    plt.savefig(f"{filename}.png", dpi=400)


def recreate_wei_fig_2(filename: str, 
                       steps: Optional[int] = 100):
    """Plots the numerical GME vs s for the GHZ W state."""

    dims = [2, 2, 2]
    S = np.linspace(0, 1, steps)
    n_gme_0 = []
    n_gme_pi = []
    for s in S:
        GHZ = GHZState(3, dims)
        W = PermutationInvariantState(3, 2, dims)

        psi = QuantumState.superposition([GHZ, W], 
                                         [np.sqrt(s), np.sqrt(1-s)])

        MPSO = MPSOptimizer(3, psi.mps)

        MPSO.optimize(100)

        psi.numerical_gme = 1 + negative_overlap(psi.mps, MPSO.mps_optimal)

        n_gme_0.append(psi.numerical_gme)

    plt.scatter(S, n_gme_0, s=5)
    plt.ylabel(r"$E_{\sin^{2}}$")
    plt.xlabel("s")
    plt.title(r"$E_{\sin^{2}}$ of $\sqrt{s}|$GHZ$\rangle + \sqrt{1-s}|W\rangle$")
    plt.savefig(f"{filename}.png", dpi=400)


def sample():
    """Generates sample plot of GME"""

    for N in range(4, 11):
        dims = tuple(phys_dim for _ in range(N))

        t_gme = []
        n_gme = []
        for k in range(N+1):

            psi = GHZState(N, dims)

            # mps0 = approximate_mps(psi.mps, rank=1)

            MPSO = MPSOptimizer(N, psi.mps)

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


def plot_qudit_states():

    n = 10

    random_K = {"2" : [5, 5],
                "3" : [3, 4, 3],
                "4" : [2, 3, 3, 2],
                "5" : [2, 2, 2, 2, 2],
                "6" : [1, 2, 2, 2, 2, 1],
                "7" : [1, 1, 2, 2, 2, 1, 1],
                "8" : [1, 1, 1, 2, 2, 1, 1, 1],
                "9" : [1, 1, 1, 1, 2, 1, 1, 1, 1]}

    n_gme = []
    t_gme = []
    for d, K in random_K.items():

        psi = SymmetricQuditState(n, K)

        MPSO = MPSOptimizer(n, psi.mps)

        MPSO.optimize(100)

        psi.numerical_gme = 1 + negative_overlap(psi.mps, MPSO.mps_optimal)

        n_gme.append(psi.numerical_gme)
        t_gme.append(psi.theoretical_gme)

    plt.plot([i for i in range(2, n)], t_gme, "--", c=COLORS[0])
    plt.plot([i for i in range(2, n)], n_gme, c=COLORS[0])
    plt.xlabel("d")
    plt.title(r"$E_{\sin^{2}}$ for Symmetric Qudit States")
    plt.savefig(f"{FIG_DIR}/{args.datetime}_mps_sym_qudit.png", dpi=400)
