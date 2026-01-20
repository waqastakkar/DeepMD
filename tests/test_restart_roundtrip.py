from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "io"))

from restart import read_restart, write_restart, RestartRecord  # noqa: E402


def test_restart_roundtrip_preserves_content(tmp_path):
    content = "\n".join(
        [
            "#CustomHeader\tValues(kcal/mol)",
            "# comment before steps",
            "(0)Steps:\t10",
            "",
            "# comment before VminD",
            "(1)Boosted VminD:\t1.0",
            "(2)Boosted VmaxD:\t2.0",
            "(3)DihedralRefEnergy:\t3.0",
            "(4)Final DihedralBoost:\t4.0",
            "(5)Final k0D:\t0.5",
            "(6)Boosted VminP:\t5.0",
            "(7)Boosted VmaxP:\t6.0",
            "(8)TotalRefEnergy:\t7.0",
            "(9)Final TotalBoost:\t8.0",
            "(10)Final k0P:\t0.9",
            "# trailing comment",
        ]
    )
    path = tmp_path / "restart.dat"
    path.write_text(content + "\n", encoding="utf-8")

    rec = read_restart(path)
    out_path = tmp_path / "restart_out.dat"
    write_restart(out_path, rec)

    assert out_path.read_text(encoding="utf-8") == content + "\n"


def test_restart_roundtrip_values(tmp_path):
    rec = RestartRecord(
        steps=42,
        VminD_kJ=10.0,
        VmaxD_kJ=20.0,
        DihedralRef_kJ=30.0,
        DihedralBoost_kJ=40.0,
        k0D=0.25,
        VminP_kJ=50.0,
        VmaxP_kJ=60.0,
        TotalRef_kJ=70.0,
        TotalBoost_kJ=80.0,
        k0P=0.75,
    )
    path = tmp_path / "restart.dat"
    write_restart(path, rec)

    reread = read_restart(path)
    assert reread.steps == rec.steps
    assert reread.VminD_kJ == pytest.approx(rec.VminD_kJ)
    assert reread.VmaxD_kJ == pytest.approx(rec.VmaxD_kJ)
    assert reread.DihedralRef_kJ == pytest.approx(rec.DihedralRef_kJ)
    assert reread.DihedralBoost_kJ == pytest.approx(rec.DihedralBoost_kJ)
    assert reread.k0D == pytest.approx(rec.k0D)
    assert reread.VminP_kJ == pytest.approx(rec.VminP_kJ)
    assert reread.VmaxP_kJ == pytest.approx(rec.VmaxP_kJ)
    assert reread.TotalRef_kJ == pytest.approx(rec.TotalRef_kJ)
    assert reread.TotalBoost_kJ == pytest.approx(rec.TotalBoost_kJ)
    assert reread.k0P == pytest.approx(rec.k0P)
