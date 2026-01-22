from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.learn.data import prepare_pca_inputs, project_pca


def test_prepare_pca_inputs_projects_with_matching_features():
    feature_columns = [
        "E_potential_kJ",
        "E_bond_kJ",
        "E_angle_kJ",
        "E_dihedral_kJ",
        "E_nonbonded_kJ",
        "T_K",
    ]
    payload = {
        "feature_names": list(feature_columns),
        "original_dim": 6,
        "kept_feature_indices": [0, 1, 2, 3, 4, 5],
    }
    X = np.arange(12, dtype=float).reshape(2, 6)
    mean = np.zeros(6, dtype=float)
    components = np.eye(6, dtype=float)[:2]

    X_prepared, kept = prepare_pca_inputs(X, feature_columns, payload)
    Z = project_pca(X_prepared, mean, components, kept)

    assert Z.shape == (2, 2)


def test_prepare_pca_inputs_raises_on_mismatch():
    payload = {
        "feature_names": [
            "E_potential_kJ",
            "E_bond_kJ",
            "E_angle_kJ",
            "E_dihedral_kJ",
            "E_nonbonded_kJ",
            "T_K",
        ],
        "original_dim": 6,
        "kept_feature_indices": [0, 1, 2, 3, 4, 5],
    }
    X = np.zeros((2, 3), dtype=float)
    feature_columns = ["E_potential_kJ", "E_bond_kJ", "E_angle_kJ"]

    with pytest.raises(ValueError, match="feature_columns"):
        prepare_pca_inputs(X, feature_columns, payload)
