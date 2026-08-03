from __future__ import annotations

import numpy as np
from rahola_lab.detectors import GrayBoxDetector


def test_graybox_fits_physical_posterior_and_hazard_heads() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(32, 64, 2)).astype(np.float32)
    states = rng.normal(scale=0.1, size=(32, 2)).astype(np.float32)
    labels = np.tile(np.array([0, 1], dtype=np.int8), 16)
    latents = rng.normal(size=(32, 7)).astype(np.float32)
    model = GrayBoxDetector(auxiliary_weight=0.25, epochs=1, batch_size=16)
    model.fit(features, states, labels, latents)
    assert model.parameter_count() < 100_000
    assert model.predict_scores(features, states).shape == (32,)
    assert model.predict_latents(features, states).shape == (32, 7)
