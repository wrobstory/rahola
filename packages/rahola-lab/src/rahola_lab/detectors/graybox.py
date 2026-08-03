"""Amortized physical-state filter with an analytic capsize-margin head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def _parameter_count(parameters: dict[str, jax.Array]) -> int:
    return sum(math.prod(value.shape) for value in parameters.values())


@dataclass
class GrayBoxDetector:
    """Temporal encoder -> Gaussian physical latents -> analytic margin hazard."""

    auxiliary_weight: float
    channels: tuple[int, int] = (16, 32)
    kernel_size: int = 7
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 0.002
    seed: int = 34_901
    parameters_: dict[str, jax.Array] | None = None
    latent_mean_: NDArray[np.float32] | None = None
    latent_scale_: NDArray[np.float32] | None = None

    def _initialize(self, latent_count: int) -> dict[str, jax.Array]:
        keys = jax.random.split(jax.random.PRNGKey(self.seed), 5)
        first, second = self.channels
        parameters = {
            "conv1": 0.08
            * jax.random.normal(keys[0], (self.kernel_size, 2, first), dtype=jnp.float32),
            "bias1": jnp.zeros(first, dtype=jnp.float32),
            "conv2": 0.08
            * jax.random.normal(keys[1], (self.kernel_size, first, second), dtype=jnp.float32),
            "bias2": jnp.zeros(second, dtype=jnp.float32),
            "latent_mean": 0.08
            * jax.random.normal(keys[2], (second, latent_count), dtype=jnp.float32),
            "latent_mean_bias": jnp.zeros(latent_count, dtype=jnp.float32),
            "latent_log_scale": 0.02
            * jax.random.normal(keys[3], (second, latent_count), dtype=jnp.float32),
            "latent_log_scale_bias": jnp.full(latent_count, -1.0, dtype=jnp.float32),
            "hazard": 0.08 * jax.random.normal(keys[4], (latent_count + 3, 1), dtype=jnp.float32),
            "hazard_bias": jnp.zeros(1, dtype=jnp.float32),
        }
        if _parameter_count(parameters) > 100_000:
            raise ValueError("gray-box encoder exceeds the 100k parameter budget")
        return parameters

    def parameter_count(self) -> int:
        return _parameter_count(self.parameters_ or self._initialize(7))

    @staticmethod
    def _convolve(inputs: jax.Array, kernel: jax.Array, bias: jax.Array) -> jax.Array:
        return jax.nn.silu(
            jax.lax.conv_general_dilated(
                inputs,
                kernel,
                window_strides=(2,),
                padding="SAME",
                dimension_numbers=("NWC", "WIO", "NWC"),
            )
            + bias
        )

    def _forward(
        self,
        parameters: dict[str, jax.Array],
        inputs: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hidden = self._convolve(inputs, parameters["conv1"], parameters["bias1"])
        hidden = self._convolve(hidden, parameters["conv2"], parameters["bias2"])
        pooled = jnp.mean(hidden, axis=1)
        latent_mean = pooled @ parameters["latent_mean"] + parameters["latent_mean_bias"]
        latent_log_scale = jnp.clip(
            pooled @ parameters["latent_log_scale"] + parameters["latent_log_scale_bias"],
            -4.0,
            2.0,
        )
        # Latent 0 is standardized current stiffness and latent 1 is standardized
        # stiffness drift. Decoding stays differentiable and feeds a split-time-
        # inspired outward-rate margin rather than an unconstrained pattern head.
        assert self.latent_mean_ is not None and self.latent_scale_ is not None
        decoded = latent_mean * jnp.asarray(self.latent_scale_) + jnp.asarray(self.latent_mean_)
        stiffness = jnp.maximum(decoded[:, 0], 0.02)
        x, velocity = state[:, 0], state[:, 1]
        remaining_distance = jnp.maximum(1.0 - jnp.abs(x), 0.0)
        critical_rate = jnp.sqrt(stiffness) * remaining_distance
        outward_margin = jnp.abs(velocity) - critical_rate
        hazard_input = jnp.column_stack(
            (latent_mean, outward_margin, jnp.abs(x), jnp.abs(velocity))
        )
        logits = (hazard_input @ parameters["hazard"] + parameters["hazard_bias"])[:, 0]
        return logits, latent_mean, latent_log_scale

    def fit(
        self,
        features: NDArray[np.floating],
        states: NDArray[np.floating],
        labels: NDArray[np.integer],
        latent_targets: NDArray[np.floating],
    ) -> GrayBoxDetector:
        inputs = np.asarray(features, dtype=np.float32)
        state_values = np.asarray(states, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.float32)
        latents = np.asarray(latent_targets, dtype=np.float32)
        if (
            inputs.ndim != 3
            or inputs.shape[-1] != 2
            or state_values.shape != (len(inputs), 2)
            or targets.shape != (len(inputs),)
            or latents.ndim != 2
            or len(latents) != len(inputs)
        ):
            raise ValueError("gray-box training arrays have incompatible shapes")
        self.latent_mean_ = np.mean(latents, axis=0)
        self.latent_scale_ = np.maximum(np.std(latents, axis=0), 1e-3)
        normalized_latents = (latents - self.latent_mean_) / self.latent_scale_
        parameters = self._initialize(latents.shape[1])
        first = jax.tree.map(jnp.zeros_like, parameters)
        second = jax.tree.map(jnp.zeros_like, parameters)
        positive_weight = float(np.sum(targets == 0) / max(np.sum(targets == 1), 1.0))

        @jax.jit
        def train_step(parameters, first, second, batch_x, batch_state, batch_y, batch_z, step):
            def loss_fn(current):
                logits, mean, log_scale = self._forward(current, batch_x, batch_state)
                weights = jnp.where(batch_y > 0.5, positive_weight, 1.0)
                binary = jnp.mean(
                    weights
                    * (
                        jnp.maximum(logits, 0.0)
                        - logits * batch_y
                        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
                    )
                )
                gaussian_nll = jnp.mean(
                    0.5 * ((batch_z - mean) / jnp.exp(log_scale)) ** 2 + log_scale
                )
                return binary + self.auxiliary_weight * gaussian_nll

            loss, gradient = jax.value_and_grad(loss_fn)(parameters)
            first = jax.tree.map(lambda old, grad: 0.9 * old + 0.1 * grad, first, gradient)
            second = jax.tree.map(lambda old, grad: 0.999 * old + 0.001 * grad**2, second, gradient)
            first_hat = jax.tree.map(lambda value: value / (1.0 - 0.9**step), first)
            second_hat = jax.tree.map(lambda value: value / (1.0 - 0.999**step), second)
            parameters = jax.tree.map(
                lambda value, mean_value, variance: (
                    value - self.learning_rate * mean_value / (jnp.sqrt(variance) + 1e-8)
                ),
                parameters,
                first_hat,
                second_hat,
            )
            return parameters, first, second, loss

        rng = np.random.default_rng(self.seed)
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(inputs))
            for start in range(0, len(inputs), self.batch_size):
                selected = order[start : start + self.batch_size]
                step += 1
                parameters, first, second, _ = train_step(
                    parameters,
                    first,
                    second,
                    inputs[selected],
                    state_values[selected],
                    targets[selected],
                    normalized_latents[selected],
                    step,
                )
        self.parameters_ = parameters
        return self

    def _predict(
        self,
        features: NDArray[np.floating],
        states: NDArray[np.floating],
        *,
        batch_size: int = 2048,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.parameters_ is None or self.latent_mean_ is None or self.latent_scale_ is None:
            raise RuntimeError("fit must be called before prediction")
        values = np.asarray(features, dtype=np.float32)
        state_values = np.asarray(states, dtype=np.float32)
        scores = []
        latents = []
        for start in range(0, len(values), batch_size):
            logits, mean, _ = self._forward(
                self.parameters_,
                values[start : start + batch_size],
                state_values[start : start + batch_size],
            )
            scores.append(np.asarray(logits))
            latents.append(
                np.asarray(mean) * self.latent_scale_[None, :] + self.latent_mean_[None, :]
            )
        return np.concatenate(scores), np.concatenate(latents)

    def predict_scores(
        self, features: NDArray[np.floating], states: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        return self._predict(features, states)[0].astype(np.float64)

    def predict_latents(
        self, features: NDArray[np.floating], states: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        return self._predict(features, states)[1].astype(np.float64)
