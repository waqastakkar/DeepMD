import numpy as np

from validate.metrics import gaussianity_report


def test_gaussianity_report_on_gaussian():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=50_000)
    report = gaussianity_report(samples)

    assert abs(report["skewness"]) < 0.1
    assert abs(report["excess_kurtosis"]) < 0.1
    assert report["tail_risk"] < 0.01
