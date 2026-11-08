import numpy as np
import pytest

from treepack.io import format_value, read_rows, read_submission, write_submission


def test_format_has_prefix_and_round_trips():
    for v in (0.0, -1.5, 0.1 + 0.2, 1e-17, np.float64(123.456789012345678)):
        s = format_value(v)
        assert s.startswith("s")
        assert float(s[1:]) == float(v)


def test_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    layouts = {n: rng.uniform(-50, 50, (n, 3)) for n in (1, 2, 7)}
    path = tmp_path / "sub.csv"
    write_submission(path, layouts)
    back = read_submission(path)
    assert set(back) == {1, 2, 7}
    for n in layouts:
        assert np.array_equal(back[n], layouts[n])
    first = path.read_text().splitlines()[:3]
    assert first[0] == "id,x,y,deg"
    assert first[1].startswith("001_0,s")
    assert first[2].startswith("002_0,s")


def test_rejects_bad_ids(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,x,y,deg\n002_0,s0,s0,s0\n002_0,s1,s0,s0\n")
    with pytest.raises(ValueError, match="out of sequence"):
        read_rows(path)


def test_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,x,y\n")
    with pytest.raises(ValueError, match="header"):
        read_rows(path)
