import numpy as np

from treepack.io import layouts_to_rows, write_submission
from treepack.lattice import lattice_layout
from treepack.validate import validate_file, validate_rows


def _grid(n):
    k = int(np.ceil(np.sqrt(n)))
    return np.array([[(i % k) * 0.8, (i // k) * 1.1, 0.0] for i in range(n)])


def test_full_file_passes_and_scores(tmp_path):
    layouts = {n: _grid(n) for n in range(1, 201)}
    path = tmp_path / "sub.csv"
    write_submission(path, layouts)
    report = validate_file(path)
    assert report.ok, report.errors
    assert len(report.sides) == 200
    assert float(report.sides[1]) == 1.0


def test_detects_overlap_in_large_n(small_lattice):
    lay = lattice_layout(180, small_lattice)
    assert validate_rows(layouts_to_rows({180: lay}), require_all=False).ok
    bad = lay.copy()
    bad[150, :2] = bad[151, :2] + [0.05, 0.0]
    report = validate_rows(layouts_to_rows({180: bad}), require_all=False)
    assert not report.ok
    assert "overlapping" in report.errors[0]


def test_touching_is_allowed():
    lay = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    assert validate_rows(layouts_to_rows({2: lay}), require_all=False).ok


def test_detects_out_of_range_and_wrong_count():
    lay = np.array([[100.5, 0.0, 0.0]])
    assert not validate_rows(layouts_to_rows({1: lay}), require_all=False).ok
    rows = layouts_to_rows({3: _grid(3)})
    rows[3] = rows[3][:2]
    assert not validate_rows(rows, require_all=False).ok


def test_missing_n_is_an_error():
    rows = layouts_to_rows({1: _grid(1)})
    assert not validate_rows(rows).ok
    assert validate_rows(rows, require_all=False).ok


def test_non_finite_values_are_reported_not_raised():
    for bad in ("snan", "nan", "inf", "-Infinity", "1e400"):
        for col in range(3):
            rec = ["s0.0", "s0.0", "s0.0"]
            rec[col] = "s" + bad
            report = validate_rows({1: [tuple(rec)]}, require_all=False)
            assert not report.ok
            assert "not a finite number" in report.errors[0]
