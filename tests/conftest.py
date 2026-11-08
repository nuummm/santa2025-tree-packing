import numpy as np
import pytest

from treepack.lattice import LatticeParams, search


@pytest.fixture(scope="session")
def small_lattice() -> LatticeParams:
    """A coarse lattice search: quick, overlap-free by construction, not dense."""
    return search(
        thetas=np.array([0.0, 15.0]),
        u_grid=np.linspace(0.0, 1.0, 6),
        v_grid=np.linspace(-1.0, 1.0, 9),
        n_shear=4,
        refine_iters=3,
    )
