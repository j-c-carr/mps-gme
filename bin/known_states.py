import typing
import logging
import itertools
import numpy as np
from math import factorial
from typing import Optional, List, Any, Union
from utils import binomial

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

LOGGER = init_logger(__name__)


class QuantumState():
    """Super class for all quantum states
    ----------
    Attributes
    :n: (int) Number of quantum bodies in the state
    :dims: (tuple) Physical dimensions of the quantum system
    :state_vector: (np.ndarray) 1-d numpy array state vector of quantum system
    :theoretical_gme: (float) Theoretical value of GME derived in (Wei, 2003)
    :mps: (quimb.tensor.MartixProductState) Matrix Product State representation
    """

    def __init__(self, n: int, dims: List[int]):

        self.n = n
        self.dims = tuple(dims)
        self.state_vector = None
        self.mps = None
        self.theoretical_gme = 0


    @classmethod
    def superposition(cls, psi_list: List[Any], c_list: List[Union[float, complex]]):
        """Returns superimposed Quantum state"""

        assert len(psi_list) >= 2
        assert len(c_list) == len(psi_list)

        # Assert all dims and values for N are identical
        #assert len({psi.dims for psi in psi_list}) == 1
        assert len({psi.n for psi in psi_list}) == 1
        #assert len({psi.state_vector.shape for psi in psi_list}) == 1

        assert abs(sum([abs(c)**2 for c in c_list]) - 1.0) <= 1e-8

        print("Old shape: ", psi_list[0].state_vector.shape)
        new_state_vector = np.sum([c * psi.state_vector for c, psi in
            zip(c_list, psi_list)], axis=0)
        print("new shape: ", new_state_vector.shape)

        assert abs(np.dot(new_state_vector, new_state_vector.T) - 1.0) < 1e-8

        state = cls(psi_list[0].n, psi_list[0].dims)
        state.state_vector = new_state_vector
        state.mps = qtn.MatrixProductState.from_dense(quimb.qu(new_state_vector, 
                                                     qtype="ket",
                                                     dtype=new_state_vector.dtype), 
                                                     psi_list[0].dims)
        return state


class PermutationInvariantState(QuantumState):

    """Wrapper for PermutationInvariantState. Analytic solutions are based in (Wei, 2003)
    ----------
    Parameters
    :n: (int) Number of quantum bodies in the state
    :state_vector: (np.ndarray) 1-d numpy array state vector of quantum system
    :theoretical_gme: (float) Theoretical value of GME derived in (Wei, 2003)
    :mps: (quimb.tensor.MartixProductState) Matrix Product State representation
    """

    def __init__(self, n: int, k: int, dims: List[int]):
       
        assert k <= n and n > 0

        super(PermutationInvariantState, self).__init__(n, dims)

        self.state_vector = self.create_state_vector(n, k)
        self.theoretical_gme = self.compute_theoretical_gme(n, k)

        self.mps = qtn.MatrixProductState.from_dense(quimb.qu(self.state_vector, 
                                                     qtype="ket",
                                                     dtype=self.state_vector.dtype), 
                                                     self.dims)


    def create_state_vector(self, n: int, k: int) -> np.ndarray:
        """Creates a 1-D wavefunction"""

        c = np.sqrt(1 / binomial(n, k))
        LOGGER.debug(f"coefficient: {c}")

        psi = np.zeros(tuple([2 for _ in range(n)]))

        I = [0 if i < k else 1 for i in range(n)]
        count = 0
        for ix in itertools.permutations(I):
            count += 1
            psi[ix] = max(psi[ix], c)

        LOGGER.debug(f"There are {count} permutations, should be {binomial(n,k)}")

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


class GHZState(QuantumState):

    """Wrapper for GHZ states. Analytic solutions to GME are based in (Wei, 2003)
    ----------
    Parameters
    :n: (int) Number of quantum bodies in the state
    :state_vector: (np.ndarray) 1-d numpy array state vector of quantum system
    :theoretical_gme: (float) Theoretical value of GME derived in (Wei, 2003)
    :mps: (quimb.tensor.MartixProductState) Matrix Product State representation
    """

    def __init__(self, n: int, dims: List[int]):
       
        assert n > 0

        super(GHZState, self).__init__(n, dims)

        self.state_vector = self.create_state_vector(n)
        self.theoretical_gme = self.compute_theoretical_gme()

        self.mps = qtn.MatrixProductState.from_dense(quimb.qu(self.state_vector, 
                                                     qtype="ket",
                                                     dtype=self.state_vector.dtype), 
                                                     self.dims)


    def create_state_vector(self, n: int) -> np.ndarray:
        """Creates a 1-D wavefunction
        -------
        Returns: state_vector vector as a 1d numpy array."""

        # Create S(m, 0)
        c0 = np.sqrt(1 / binomial(n, 0))
        LOGGER.debug(f"coefficient: {c0}")

        psi0 = np.zeros(tuple([2 for _ in range(n)]))

        I = [1 for _ in range(n)]

        for ix in itertools.permutations(I):
            psi0[ix] = max(psi0[ix], c0)

        psi0 = psi0.flatten()


        # Create S(m, m)
        c1 = np.sqrt(1 / binomial(n, n))
        LOGGER.debug(f"coefficient: {c1}")

        psi1 = np.zeros(tuple([2 for _ in range(n)]))

        I = [0 for _ in range(n)]
        count = 0
        for ix in itertools.permutations(I):
            count += 1
            psi1[ix] = max(psi1[ix], c1)
        psi1 = psi1.flatten()

        # Create GHZ state
        psi = (psi0 + psi1) / np.sqrt(2)


        psi = psi.flatten()

        assert abs(np.dot(psi, psi.T) - 1.0) < 1e-8, \
                f"Expected unit norm but psi has norm {np.dot(psi, psi.T)}."

        # scipy optimization requires np.float64 data entries
        return psi.astype(np.float64)


    def compute_theoretical_gme(self) -> float:
        """Computes geometric measure of entanglement from (Wei, 2003)"""

        return 0.5


class WWbarSuperposition():

    """Wrapper for  Analytic solutions are based in (Wei, 2003). The states
    superimposed are:

               W = |S(3,2)> = (|001> + |010> + |100>)/sqrt(3)
            Wbar = |S(3,1)> = (|110> + |101> + |011>)/sqrt(3)

    The WWbarSuperposition state is given by:
            |psi(s)> = sqrt(s)*|W> + sqrt(1-s)*|Wbar>

    ----------
    Attributes
    :n: (int) Number of quantum bodies in the state
    :state_vector: (np.ndarray) 1-d numpy array state vector of quantum system
    :theoretical_gme: (float) Theoretical value of GME derived in (Wei, 2003)
    :mps: (quimb.tensor.MartixProductState) Matrix Product State representation
    """

    def __init__(self, s: float):
       
        # Note that s could also be complex, we only need 0 <= |s| <= 1
        assert 0 <= s <= 1, \
            f"expected :s: to be in [0,1] but got {s}"

        self.dims = (2, 2, 2)
        self.state_vector = self.create_state_vector(s)
        self.theoretical_gme = self.compute_theoretical_gme(s)

        self.mps = qtn.MatrixProductState.from_dense(quimb.qu(self.state_vector, 
                                                     qtype="ket",
                                                     dtype=self.state_vector.dtype), 
                                                     self.dims)


    def create_state_vector(self, s: float) -> np.ndarray:
        """Creates a 1-D wavefunction
        -------
        Returns: state_vector vector as a 1d numpy array."""

        n = 3
        k0 = 2
        k1 = 1
        psi0 = np.zeros(self.dims)
        psi1 = np.zeros(self.dims)

        # Create S(3,2) and S(3,1)
        for _psi, k in zip([psi0, psi1], [k0, k1]):
            c = np.sqrt(1 / binomial(n, k))
            LOGGER.debug(f"coefficient: {c}")

            I = [0 if i < k else 1 for i in range(n)]

            # Max is taken to avoid multiplicities
            for ix in itertools.permutations(I):
                _psi[ix] = max(_psi[ix], c) 
            _psi = _psi.flatten()

            assert abs(np.dot(_psi, _psi.T) - 1.0) < 1e-8, \
                    f"Expected unit norm but psi has norm {np.dot(psi, psi.T)}."

        # # Create Wbar = S(3, 1)
        # c1 = np.sqrt(1 / binomial(n, k1))
        # LOGGER.debug(f"coefficient: {c1}")

        # psi1 = np.zeros(self.dims)

        # I = [0 if i < k1 else 1 for i in range(n)]
        # count = 0
        # for ix in itertools.permutations(I):
        #     count += 1
        #     psi1[ix] = max(psi1[ix], c1)
        # psi1 = psi1.flatten()

        # assert abs(np.dot(psi1, psi1.T) - 1.0) < 1e-8, \
        #         f"Expected unit norm but psi has norm {np.dot(psi, psi.T)}."

        # Create GHZ state
        psi = (np.sqrt(s) * psi0) + (np.sqrt(1-s) * psi1)

        psi = psi.flatten()

        assert abs(np.dot(psi, psi.T) - 1.0) < 1e-8, \
                f"Expected unit norm but psi has norm {np.dot(psi, psi.T)}."

        # scipy optimization requires np.float64 data entries
        return psi.astype(np.float64)


    def compute_theoretical_gme(self, s: float) -> float:
        """Computes geometric measure of entanglement from (Wei, 2003), eq. 19"""

        tan_ode = np.polynomial.polynomial.Polynomial([-np.sqrt(s), 
                                                       -2*np.sqrt(1-s),
                                                       2*np.sqrt(s),
                                                       np.sqrt(1-s)])
        _t = tan_ode.roots()

        LOGGER.debug(f"roots: \n{_t}")

        # Particular solution is always the last root
        t = _t[-1]
        assert np.sqrt(0.5) <= t <= np.sqrt(2), \
                f"expected t in range [sqrt(1/2), sqrt(2)] but got {t}"

        theta = np.arctan(t)
        print("\ttheta: ", theta)

        l_max = (np.sin(2*theta)/np.sqrt(2)) * (np.sqrt(s) * np.cos(theta) 
                                       + np.sqrt(1-s) * np.sin(theta))

        print("\tl_max: ", l_max)
        return 1 - l_max**2


class SymmetricQuditState(QuantumState):

    """Wrapper for SymmetricQuditState Analytic solutions are based in (Wei, 2003)
    ----------
    Parameters
    :n: (int) Number of quantum bodies in the state
    :state_vector: (np.ndarray) 1-d numpy array state vector of quantum system
    :theoretical_gme: (float) Theoretical value of GME derived in (Wei, 2003)
    :mps: (quimb.tensor.MartixProductState) Matrix Product State representation
    """

    def __init__(self, n: int, K: int):
       
        assert n > 0 and sum(K) == n

        dims = [len(K) for _ in range(n)]
        super(SymmetricQuditState, self).__init__(n, dims)

        self.state_vector = self.create_state_vector(n, K)
        self.theoretical_gme = self.compute_theoretical_gme(n, K)

        self.mps = qtn.MatrixProductState.from_dense(quimb.qu(self.state_vector, 
                                                     qtype="ket",
                                                     dtype=self.state_vector.dtype), 
                                                     self.dims)


    def create_state_vector(self, n: int, K: List[int]) -> np.ndarray:
        """Creates a 1-D wavefunction"""

        c = np.sqrt(np.prod(np.array([factorial(k) for k in K])) / factorial(n))
        LOGGER.debug(f"coefficient: {c}")

        psi = np.zeros(tuple(self.dims))

        # Create index set
        I = []
        for i in range(len(K)):
            I += [i for _ in range(K[i])]
        LOGGER.debug(f"Index set: {I}")


        for ix in itertools.permutations(I):
            # Take max to avoid multiplicities
            psi[ix] = max(psi[ix], c) 


        # TODO: make sure flattening is consistent with vector identification
        psi = psi.flatten()

        assert abs(np.dot(psi, psi.T) - 1.0) < 1e-8, \
                f"Expected unit norm but psi has norm {np.dot(psi, psi.T)}."

        return psi.astype(np.float64)


    def compute_theoretical_gme(self, n: int, K: int) -> float:
        """Computes geometric measure of entanglement from (Wei, 2003)"""

        a = np.sqrt(factorial(n) / np.prod(np.array([factorial(k) for k in K])))
        # Maximal eigenvalue associated to the state
        l_max = a * np.prod(np.array([np.power(k/n, k/2) for k in K])) 

        return 1 - l_max**2


