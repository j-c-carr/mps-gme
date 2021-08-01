import quimb
import numpy as np

def normalize_state(psi):
    return psi / (psi.H @ psi) ** 0.5


def negative_overlap(psi, target):
    return - (psi.H @ target) ** 2
