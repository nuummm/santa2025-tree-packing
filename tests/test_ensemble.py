import numpy as np

from treepack.ensemble import ensemble


def test_picks_smallest_valid_per_n():
    wide = {2: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    tight = {2: np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])}
    overlapping = {2: np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])}
    chosen, source = ensemble([wide, overlapping, tight])
    assert source == {2: 2}
    assert np.array_equal(chosen[2], tight[2])


def test_skips_n_without_valid_candidate():
    overlapping = {2: np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])}
    single = {1: np.array([[0.0, 0.0, 0.0]])}
    chosen, _ = ensemble([overlapping, single])
    assert set(chosen) == {1}
