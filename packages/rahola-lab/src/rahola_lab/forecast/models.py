"""Three increasingly expressive CPU-sized quantile forecasters."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]


def _pinball(target: FloatArray, prediction: FloatArray, quantile: float) -> float:
    error = target - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def summary_features(histories: FloatArray) -> FloatArray:
    """Causal motion summaries used by the linear baseline."""
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("histories must have shape (samples, time, [roll, rate])")
    length = values.shape[1]
    roll = values[:, :, 0]
    rate = values[:, :, 1]
    fractions = (1 / 12, 1 / 4, 1 / 2, 1.0)
    maxima = [
        np.max(np.abs(roll[:, -max(1, round(length * fraction)) :]), axis=1)
        for fraction in fractions
    ]
    return np.column_stack(
        (
            *maxima,
            np.sqrt(np.mean(roll**2, axis=1)),
            np.sqrt(np.mean(rate**2, axis=1)),
            np.max(np.abs(rate), axis=1),
            roll[:, -1] - roll[:, 0],
        )
    )


@dataclass
class EnvelopePersistenceForecaster:
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95)
    scales_: FloatArray | None = None

    def fit(self, histories: FloatArray, targets: FloatArray) -> EnvelopePersistenceForecaster:
        recent_max = np.maximum(np.max(np.abs(histories[:, :, 0]), axis=1), 1e-8)
        fitted: list[float] = []
        for quantile in self.quantiles:
            result = minimize_scalar(
                lambda scale, q=quantile: _pinball(targets, scale * recent_max, q),
                bounds=(0.0, 10.0),
                method="bounded",
            )
            fitted.append(float(result.x))
        self.scales_ = np.asarray(fitted, dtype=np.float64)
        return self

    def predict(self, histories: FloatArray) -> FloatArray:
        if self.scales_ is None:
            raise RuntimeError("fit must be called before predict")
        recent_max = np.max(np.abs(histories[:, :, 0]), axis=1)
        return recent_max[:, None] * self.scales_[None, :]


@dataclass
class LinearQuantileForecaster:
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95)
    learning_rate: float = 0.03
    iterations: int = 1500
    coefficients_: FloatArray | None = None
    feature_mean_: FloatArray | None = None
    feature_scale_: FloatArray | None = None

    def fit(self, histories: FloatArray, targets: FloatArray) -> LinearQuantileForecaster:
        features = summary_features(histories)
        self.feature_mean_ = features.mean(axis=0)
        self.feature_scale_ = np.maximum(features.std(axis=0), 1e-8)
        design = np.column_stack(
            (np.ones(len(features)), (features - self.feature_mean_) / self.feature_scale_)
        )
        coefficients = np.zeros((design.shape[1], len(self.quantiles)), dtype=np.float64)
        coefficients[0] = np.quantile(targets, self.quantiles)
        first = np.zeros_like(coefficients)
        second = np.zeros_like(coefficients)
        for iteration in range(1, self.iterations + 1):
            prediction = design @ coefficients
            error = targets[:, None] - prediction
            gradient = -(design.T @ (np.asarray(self.quantiles)[None, :] - (error < 0))) / len(
                targets
            )
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            corrected_first = first / (1.0 - 0.9**iteration)
            corrected_second = second / (1.0 - 0.999**iteration)
            coefficients -= (
                self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            )
        self.coefficients_ = coefficients
        return self

    def predict(self, histories: FloatArray) -> FloatArray:
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("fit must be called before predict")
        features = summary_features(histories)
        design = np.column_stack(
            (np.ones(len(features)), (features - self.feature_mean_) / self.feature_scale_)
        )
        return np.maximum.accumulate(design @ self.coefficients_, axis=1)


def _parameter_count(parameters: dict[str, jax.Array]) -> int:
    return sum(math.prod(value.shape) for value in parameters.values())


@dataclass
class JaxLSTMQuantileForecaster:
    """Single-layer LSTM quantile regressor trained with pinball loss and Adam."""

    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95)
    hidden_size: int = 32
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 0.003
    seed: int = 731
    parameters_: dict[str, jax.Array] | None = None
    input_mean_: FloatArray | None = None
    input_scale_: FloatArray | None = None

    def _initialize(self) -> dict[str, jax.Array]:
        key1, key2 = jax.random.split(jax.random.PRNGKey(self.seed))
        recurrent_width = 4 * self.hidden_size
        parameters = {
            "kernel": 0.08 * jax.random.normal(key1, (2 + self.hidden_size, recurrent_width)),
            "bias": jnp.zeros((recurrent_width,)),
            "output": 0.08 * jax.random.normal(key2, (self.hidden_size, len(self.quantiles))),
            "output_bias": jnp.zeros((len(self.quantiles),)),
        }
        if _parameter_count(parameters) > 100_000:
            raise ValueError("LSTM configuration exceeds the 100k parameter budget")
        return parameters

    def parameter_count(self) -> int:
        parameters = self.parameters_ or self._initialize()
        return _parameter_count(parameters)

    def _predict_jax(self, parameters: dict[str, jax.Array], histories: jax.Array) -> jax.Array:
        batch = histories.shape[0]

        def step(carry: tuple[jax.Array, jax.Array], sample: jax.Array):
            hidden, cell = carry
            gates = jnp.concatenate((sample, hidden), axis=1) @ parameters["kernel"]
            gates = gates + parameters["bias"]
            forget, update, candidate, output = jnp.split(gates, 4, axis=1)
            cell = jax.nn.sigmoid(forget) * cell + jax.nn.sigmoid(update) * jnp.tanh(candidate)
            hidden = jax.nn.sigmoid(output) * jnp.tanh(cell)
            return (hidden, cell), None

        initial = (
            jnp.zeros((batch, self.hidden_size)),
            jnp.zeros((batch, self.hidden_size)),
        )
        (hidden, _), _ = jax.lax.scan(step, initial, jnp.swapaxes(histories, 0, 1))
        raw = hidden @ parameters["output"] + parameters["output_bias"]
        return jnp.maximum.accumulate(raw, axis=1)

    def fit(self, histories: FloatArray, targets: FloatArray) -> JaxLSTMQuantileForecaster:
        self.input_mean_ = histories.mean(axis=(0, 1))
        self.input_scale_ = np.maximum(histories.std(axis=(0, 1)), 1e-8)
        inputs = np.asarray((histories - self.input_mean_) / self.input_scale_, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.float32)
        parameters = self._initialize()
        first = jax.tree.map(jnp.zeros_like, parameters)
        second = jax.tree.map(jnp.zeros_like, parameters)
        quantiles = jnp.asarray(self.quantiles)[None, :]

        @jax.jit
        def train_step(parameters, first, second, batch_x, batch_y, step_number):
            def loss_fn(current):
                prediction = self._predict_jax(current, batch_x)
                error = batch_y[:, None] - prediction
                return jnp.mean(jnp.maximum(quantiles * error, (quantiles - 1.0) * error))

            loss, gradient = jax.value_and_grad(loss_fn)(parameters)
            first = jax.tree.map(lambda old, grad: 0.9 * old + 0.1 * grad, first, gradient)
            second = jax.tree.map(lambda old, grad: 0.999 * old + 0.001 * grad**2, second, gradient)
            first_hat = jax.tree.map(lambda value: value / (1 - 0.9**step_number), first)
            second_hat = jax.tree.map(lambda value: value / (1 - 0.999**step_number), second)
            parameters = jax.tree.map(
                lambda value, mean, variance: (
                    value - self.learning_rate * mean / (jnp.sqrt(variance) + 1e-8)
                ),
                parameters,
                first_hat,
                second_hat,
            )
            return parameters, first, second, loss

        rng = np.random.default_rng(self.seed)
        step_number = 0
        for _ in range(self.epochs):
            for start in range(0, len(inputs), self.batch_size):
                indices = rng.permutation(len(inputs))[start : start + self.batch_size]
                step_number += 1
                parameters, first, second, _ = train_step(
                    parameters, first, second, inputs[indices], labels[indices], step_number
                )
        self.parameters_ = parameters
        return self

    def predict(self, histories: FloatArray) -> FloatArray:
        if self.parameters_ is None or self.input_mean_ is None or self.input_scale_ is None:
            raise RuntimeError("fit must be called before predict")
        inputs = np.asarray((histories - self.input_mean_) / self.input_scale_, dtype=np.float32)
        prediction = self._predict_jax(self.parameters_, jnp.asarray(inputs))
        return np.asarray(prediction, dtype=np.float64)
