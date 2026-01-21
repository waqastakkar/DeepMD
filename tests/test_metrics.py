from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")

from paddle.validate.metrics import gaussianity_report


def test_gaussianity_report_on_gaussian():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=50_000)
    report = gaussianity_report(samples)

    assert abs(report["skewness"]) < 0.1
    assert abs(report["excess_kurtosis"]) < 0.1
    assert report["tail_risk"] < 0.01
