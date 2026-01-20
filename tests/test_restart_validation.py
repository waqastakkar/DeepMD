from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "io"))

from restart import read_restart, write_restart, RestartRecord  # noqa: E402


def _base_restart_lines():
    return [
        "#Parameters\tValues(kcal/mol)",
        "(0)Steps:\t0",
        "(1)Boosted VminD:\t1.0",
        "(2)Boosted VmaxD:\t2.0",
        "(3)DihedralRefEnergy:\t3.0",
        "(4)Final DihedralBoost:\t4.0",
        "(5)Final k0D:\t0.5",
        "(6)Boosted VminP:\t5.0",
        "(7)Boosted VmaxP:\t6.0",
        "(8)TotalRefEnergy:\t7.0",
        "(9)Final TotalBoost:\t8.0",
        "(10)Final k0P:\t0.6",
    ]


@pytest.mark.parametrize(
    "idx,value",
    [
        (1, "nan"),
        (2, "inf"),
    ],
)
def test_read_rejects_non_finite(tmp_path, idx, value):
    lines = _base_restart_lines()
    lines[idx + 1] = lines[idx + 1].split("\t")[0] + f"\t{value}"
    path = tmp_path / "restart.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_restart(path)


def test_read_rejects_vmax_below_vmin(tmp_path):
    lines = _base_restart_lines()
    lines[3] = "(2)Boosted VmaxD:\t0.5"
    path = tmp_path / "restart.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_restart(path)


def test_read_clamps_k0_bounds(tmp_path):
    lines = _base_restart_lines()
    lines[6] = "(5)Final k0D:\t2.5"
    lines[11] = "(10)Final k0P:\t-1.0"
    path = tmp_path / "restart.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rec = read_restart(path, k0_bounds=(0.1, 0.9))
    assert rec.k0D == pytest.approx(0.9)
    assert rec.k0P == pytest.approx(0.1)


def test_write_clamps_k0_bounds(tmp_path):
    rec = RestartRecord(
        steps=5,
        VminD_kJ=1.0,
        VmaxD_kJ=2.0,
        DihedralRef_kJ=3.0,
        DihedralBoost_kJ=4.0,
        k0D=2.5,
        VminP_kJ=5.0,
        VmaxP_kJ=6.0,
        TotalRef_kJ=7.0,
        TotalBoost_kJ=8.0,
        k0P=-0.2,
    )
    path = tmp_path / "restart.dat"
    write_restart(path, rec, k0_bounds=(0.1, 0.9))

    reread = read_restart(path)
    assert reread.k0D == pytest.approx(0.9)
    assert reread.k0P == pytest.approx(0.1)
