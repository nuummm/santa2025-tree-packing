import numpy as np

from treepack.ensemble import ensemble
from treepack.scoring import side_of


def test_picks_smallest_valid_per_n():
    wide = {2: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    tight = {2: np.array([[0.0, 0.0, 0.0], [0.71, 0.0, 0.0]])}
    overlapping = {2: np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])}
    chosen, source = ensemble([wide, overlapping, tight])
    assert source == {2: 2}
    assert np.isclose(side_of(chosen[2]), side_of(tight[2]))
    assert np.allclose(chosen[2][1] - chosen[2][0], tight[2][1] - tight[2][0])


def test_skips_n_without_valid_candidate():
    overlapping = {2: np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])}
    single = {1: np.array([[0.0, 0.0, 0.0]])}
    chosen, _ = ensemble([overlapping, single])
    assert set(chosen) == {1}


def test_drifted_layout_is_centred_and_kept():
    far = {1: np.array([[150.0, -120.0, 45.0]]), 2: np.array([[90.0, 0.0, 0.0], [90.71, 0.0, 0.0]])}
    chosen, _ = ensemble([far])
    assert set(chosen) == {1, 2}
    assert np.all(np.abs(chosen[1][:, :2]) < 1.0)
    assert np.all(np.abs(chosen[2][:, :2]) < 1.0)
    assert np.isclose(side_of(chosen[2]), side_of(far[2]))
