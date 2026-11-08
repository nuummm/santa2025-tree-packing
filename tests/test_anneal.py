import numpy as np

from treepack.anneal import anneal
from treepack.compact import compact
from treepack.ensemble import is_valid
from treepack.initializers import remove_one
from treepack.lattice import lattice_layout
from treepack.scoring import side_of


def test_lattice_layouts_are_valid(small_lattice):
    for n in (1, 2, 7, 30, 101):
        lay = lattice_layout(n, small_lattice)
        assert lay.shape == (n, 3)
        assert is_valid(n, lay)


def test_anneal_is_valid_not_worse_and_deterministic(small_lattice):
    start = lattice_layout(25, small_lattice)
    a, acc = anneal(start, 50_000, seed=3)
    b, _ = anneal(start, 50_000, seed=3)
    c, _ = anneal(start, 50_000, seed=4)
    assert acc > 0
    assert is_valid(25, a)
    assert side_of(a) <= side_of(start)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_anneal_finds_rotated_single_tree():
    lay, _ = anneal(np.array([[0.0, 0.0, 0.0]]), 50_000, seed=0)
    # Upright the side is 1.0; tilting the tree shrinks its bounding square.
    assert side_of(lay) < 0.82


def test_compact_valid_and_not_worse(small_lattice):
    start = lattice_layout(40, small_lattice)
    spread = start.copy()
    spread[:, :2] *= 1.3
    out = compact(spread)
    assert is_valid(40, out)
    assert side_of(out) < side_of(spread)


def test_remove_one_keeps_validity(small_lattice):
    lay = lattice_layout(20, small_lattice)
    out = remove_one(lay)
    assert out.shape == (19, 3)
    assert is_valid(19, out)
    assert side_of(out) <= side_of(lay)
