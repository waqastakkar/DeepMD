import pytest

from config import SimulationConfig


def test_simulation_config_determinism_and_precision():
    cfg = SimulationConfig.from_dict(
        {"precision": "double", "deterministic_forces": True}
    )

    assert cfg.precision == "double"
    assert cfg.deterministic_forces is True


def test_get_platform_cpu_properties():
    pytest.importorskip("openmm")
    from core.engine import get_platform

    plat, props = get_platform(name="CPU", precision="double", deterministic_forces=True)

    assert plat.getName() == "CPU"
    assert props == {}


def test_get_platform_fallback_to_cpu():
    pytest.importorskip("openmm")
    from core.engine import get_platform

    plat, props = get_platform(name="NotAPlatform", precision="mixed", deterministic_forces=True)

    assert plat.getName() == "CPU"
    assert props == {}


def test_get_platform_cuda_properties_when_available():
    openmm = pytest.importorskip("openmm")
    from core.engine import get_platform

    names = {
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    }
    if "CUDA" not in names:
        pytest.skip("CUDA platform not available for testing")

    plat, props = get_platform(name="CUDA", precision="double", deterministic_forces=True)

    assert plat.getName() == "CUDA"
    assert props["Precision"] == "double"
    assert props["DeterministicForces"] == "true"
