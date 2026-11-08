import numpy as np
import pytest

from treepack.geometry import TREE_AREA
from treepack.scoring import band_scores, density, side_of, term, total_score


def test_term_and_total():
    assert term(4, 2.0) == pytest.approx(1.0)
    assert total_score({1: 1.0, 4: 2.0}) == pytest.approx(2.0)


def test_density_upper_bound():
    assert density(1, 1.0) == pytest.approx(TREE_AREA)
    assert density(1, 1.0) <= 1.0


def test_bands_partition_the_score():
    sides = {n: 1.0 + 0.01 * n for n in range(1, 201)}
    assert sum(band_scores(sides).values()) == pytest.approx(total_score(sides))


def test_side_of_two_trees_side_by_side():
    lay = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    assert side_of(lay) == pytest.approx(1.4)
