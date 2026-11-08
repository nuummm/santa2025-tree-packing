import json

from treepack.cli import main
from treepack.io import read_submission
from treepack.pipeline import SolveConfig, solve
from treepack.validate import validate_file


def test_end_to_end_small(tmp_path, small_lattice):
    cfg = SolveConfig(n_max=10, moves_per_tree=500, seeds=(0, 1), workers=2, lattice=small_lattice)
    m = solve(cfg, tmp_path)
    report = validate_file(tmp_path / "submission.csv", require_all=False)
    assert report.ok
    assert set(read_submission(tmp_path / "submission.csv")) == set(range(1, 11))
    stages = m["stages"]
    assert stages["ensemble"]["score"] <= stages["seed0/remove_one"]["score"] + 1e-9
    assert stages["seed0/remove_one"]["score"] <= stages["seed0/compact"]["score"] + 1e-9
    assert stages["seed0/compact"]["score"] <= stages["seed0/anneal"]["score"] + 1e-9
    assert stages["seed0/anneal"]["score"] <= stages["seed0/lattice"]["score"] + 1e-9
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["seeds"] == [0, 1]


def test_same_seed_same_file(tmp_path, small_lattice):
    cfg = SolveConfig(n_max=6, moves_per_tree=300, seeds=(0,), workers=1, lattice=small_lattice)
    solve(cfg, tmp_path / "a")
    cfg.workers = 3
    solve(cfg, tmp_path / "b")
    a = (tmp_path / "a" / "submission.csv").read_text()
    assert a == (tmp_path / "b" / "submission.csv").read_text()


def test_cli_validate(tmp_path, small_lattice, capsys):
    cfg = SolveConfig(n_max=3, moves_per_tree=200, seeds=(0,), workers=1, lattice=small_lattice)
    solve(cfg, tmp_path)
    assert main(["validate", "--partial", str(tmp_path / "submission.csv")]) == 0
    assert "OK: 3 configurations" in capsys.readouterr().out
    assert main(["validate", str(tmp_path / "submission.csv")]) == 1
