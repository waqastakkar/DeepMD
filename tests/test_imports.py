from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.config import SimulationConfig
import paddle


def test_imports_smoke():
    assert paddle is not None
    assert SimulationConfig is not None
