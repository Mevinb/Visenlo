import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visenlo-mpl")


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(0)


@pytest.fixture
def tmp_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VISENLO_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config"
