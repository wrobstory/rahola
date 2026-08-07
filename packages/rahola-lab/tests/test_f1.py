from __future__ import annotations

import json

import numpy as np
from rahola_lab.campaigns.f1 import F1_TEST_SLICES, verify_f1_test_slices
from rahola_lab.campaigns.load import _REFERENCE_CHECKSUMS
from rahola_lab.experiments.f1_common import fit_logistic


def test_f1_seed_slices_are_predeclared_and_disjoint(tmp_path) -> None:
    campaign = tmp_path / "existing"
    campaign.mkdir()
    (campaign / "manifest.json").write_text(
        json.dumps({"splits": {"test": {"offset": 0, "count": 1_000}}}),
        encoding="utf-8",
    )
    verification = verify_f1_test_slices((tmp_path,))
    assert verification["pairwise_disjoint"]
    assert verification["declared_count"] == 9_400
    assert F1_TEST_SLICES["softening_step_v02"] == (3_000, 38_000)
    assert set(f"{name}_f1" for name in F1_TEST_SLICES) <= _REFERENCE_CHECKSUMS.keys()


def test_f1_logistic_fit_is_frozen_and_monotone_on_separable_example() -> None:
    features = np.array([[-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]])
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    model = fit_logistic(features, labels)
    predictions = model.predict(features)
    assert np.all(np.diff(predictions) > 0.0)
    restored = type(model).from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.predict(features), predictions)
