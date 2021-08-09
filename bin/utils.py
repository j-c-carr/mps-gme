import quimb
import numpy as np
import quimb.tensor as qtn
from math import factorial

def normalize_state(psi):
    """Normalizes quimb quantum object"""
    return psi / (psi.H @ psi) ** 0.5

def binomial(n, k):
    return factorial(n) / (factorial(k) * factorial(n-k))

def negative_overlap(psi, target):
    return - (psi.H @ target) ** 2

def approximate_mps(mps, rank=1):
    """Returns a truncated mps with the first :rank: bonds"""

    assert rank <= min(mps.bond_sizes())
    assert len(mps.tensors) >= 2

    # Truncate each component tensor
    approx_tensors = []

    approx_tensors.append(mps.tensors[0].data[:, :rank])
    for a in mps.tensors[1:-1]:
        approx_tensors.append(a.data[:rank, :, :rank])

    approx_tensors.append(mps.tensors[-1].data[:rank, :])

    mps0 = qtn.MatrixProductState(approx_tensors, shape="lpr")

    return mps0
