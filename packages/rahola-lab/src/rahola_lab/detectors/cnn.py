"""Small native-JAX temporal CNN for motion-history danger detection."""

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
class JaxTemporalCNN:
    """Two strided temporal convolutions with optional family and physics heads."""

    channels: tuple[int, int] = (12, 24)
    kernel_size: int = 9
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 0.002
    seed: int = 2901
    family_head_weight: float = 0.0
    physics_head_weight: float = 0.0
    parameters_: dict[str, jax.Array] | None = None

    def _initialize(self) -> dict[str, jax.Array]:
        keys = jax.random.split(jax.random.PRNGKey(self.seed), 5)
        first, second = self.channels
        parameters = {
            "conv1": 0.08
            * jax.random.normal(keys[0], (self.kernel_size, 2, first), dtype=jnp.float32),
            "bias1": jnp.zeros((first,), dtype=jnp.float32),
            "conv2": 0.08
            * jax.random.normal(keys[1], (self.kernel_size, first, second), dtype=jnp.float32),
            "bias2": jnp.zeros((second,), dtype=jnp.float32),
            "danger": 0.08 * jax.random.normal(keys[2], (second, 1), dtype=jnp.float32),
            "danger_bias": jnp.zeros((1,), dtype=jnp.float32),
            "family": 0.08 * jax.random.normal(keys[3], (second, 3), dtype=jnp.float32),
            "family_bias": jnp.zeros((3,), dtype=jnp.float32),
            "physics": 0.08 * jax.random.normal(keys[4], (second, 1), dtype=jnp.float32),
            "physics_bias": jnp.zeros((1,), dtype=jnp.float32),
        }
        if _parameter_count(parameters) > 100_000:
            raise ValueError("CNN configuration exceeds the 100k parameter budget")
        return parameters

    def parameter_count(self) -> int:
        return _parameter_count(self.parameters_ or self._initialize())

    @staticmethod
    def _convolve(inputs: jax.Array, kernel: jax.Array, bias: jax.Array) -> jax.Array:
        return jax.nn.relu(
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
        self, parameters: dict[str, jax.Array], inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hidden = self._convolve(inputs, parameters["conv1"], parameters["bias1"])
        hidden = self._convolve(hidden, parameters["conv2"], parameters["bias2"])
        pooled = jnp.mean(hidden, axis=1)
        danger = (pooled @ parameters["danger"] + parameters["danger_bias"])[:, 0]
        family = pooled @ parameters["family"] + parameters["family_bias"]
        physics = (pooled @ parameters["physics"] + parameters["physics_bias"])[:, 0]
        return danger, family, physics

    def fit(
        self,
        features: NDArray[np.floating],
        labels: NDArray[np.integer],
        *,
        family_labels: NDArray[np.integer] | None = None,
        physics_targets: NDArray[np.floating] | None = None,
    ) -> JaxTemporalCNN:
        inputs = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.float32)
        if inputs.ndim != 3 or inputs.shape[-1] != 2 or targets.shape != (len(inputs),):
            raise ValueError("features and labels have incompatible shapes")
        families = np.zeros(len(inputs), dtype=np.int32)
        if self.family_head_weight:
            if family_labels is None or np.asarray(family_labels).shape != (len(inputs),):
                raise ValueError("enabled family head requires one family label per window")
            families = np.asarray(family_labels, dtype=np.int32)
        physics = np.zeros(len(inputs), dtype=np.float32)
        if self.physics_head_weight:
            if physics_targets is None or np.asarray(physics_targets).shape != (len(inputs),):
                raise ValueError("enabled physics head requires one target per window")
            physics = np.asarray(physics_targets, dtype=np.float32)
        parameters = self._initialize()
        first = jax.tree.map(jnp.zeros_like, parameters)
        second = jax.tree.map(jnp.zeros_like, parameters)
        positive_weight = float(np.sum(targets == 0) / max(np.sum(targets == 1), 1.0))

        @jax.jit
        def train_step(
            parameters, first, second, batch_x, batch_y, batch_family, batch_physics, step
        ):
            def loss_fn(current):
                logits, family_logits, physics_prediction = self._forward(current, batch_x)
                weights = jnp.where(batch_y > 0.5, positive_weight, 1.0)
                binary = jnp.mean(
                    weights
                    * (
                        jnp.maximum(logits, 0.0)
                        - logits * batch_y
                        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
                    )
                )
                family_loss = -jnp.mean(
                    jax.nn.log_softmax(family_logits)[jnp.arange(len(batch_family)), batch_family]
                )
                physics_loss = jnp.mean((physics_prediction - batch_physics) ** 2)
                return (
                    binary
                    + self.family_head_weight * family_loss
                    + self.physics_head_weight * physics_loss
                )

            loss, gradient = jax.value_and_grad(loss_fn)(parameters)
            first = jax.tree.map(lambda old, grad: 0.9 * old + 0.1 * grad, first, gradient)
            second = jax.tree.map(lambda old, grad: 0.999 * old + 0.001 * grad**2, second, gradient)
            first_hat = jax.tree.map(lambda value: value / (1.0 - 0.9**step), first)
            second_hat = jax.tree.map(lambda value: value / (1.0 - 0.999**step), second)
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
                    targets[selected],
                    families[selected],
                    physics[selected],
                    step,
                )
        self.parameters_ = parameters
        return self

    def predict_scores(
        self, features: NDArray[np.floating], *, batch_size: int = 2048
    ) -> NDArray[np.float64]:
        if self.parameters_ is None:
            raise RuntimeError("fit must be called before prediction")
        values = np.asarray(features, dtype=np.float32)
        pieces = [
            np.asarray(self._forward(self.parameters_, values[start : start + batch_size])[0])
            for start in range(0, len(values), batch_size)
        ]
        return np.concatenate(pieces).astype(np.float64) if pieces else np.empty(0)
