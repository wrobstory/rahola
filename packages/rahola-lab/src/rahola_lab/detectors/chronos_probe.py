"""Pinned Chronos-T5 embedding and encoder-fine-tune classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from chronos import ChronosPipeline
from numpy.typing import NDArray

from rahola_lab.constants import (
    B2_CONTEXT_SAMPLES,
    B2_FINETUNE_EPOCHS,
    B2_FINETUNE_MAX_WINDOWS,
    B2_FROZEN_HEAD_EPOCHS,
    CHRONOS_CHECKPOINT,
    CHRONOS_REVISION,
)


@dataclass
class ChronosClassifier:
    """Binary head over two univariate Chronos encoder passes."""

    mode: str
    seed: int = 71_903
    pipeline_: ChronosPipeline | None = None
    head_: torch.nn.Linear | None = None

    def _load(self) -> None:
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.pipeline_ = ChronosPipeline.from_pretrained(
            CHRONOS_CHECKPOINT,
            revision=CHRONOS_REVISION,
            device_map="cpu",
        )
        self.head_ = torch.nn.Linear(512, 1)

    @staticmethod
    def _contexts(features: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(features, dtype=np.float32)
        indices = np.linspace(0, values.shape[1] - 1, B2_CONTEXT_SAMPLES, dtype=np.int64)
        return values[:, indices]

    def _frozen_embeddings(
        self, features: NDArray[np.floating], *, batch_size: int = 32
    ) -> torch.Tensor:
        assert self.pipeline_ is not None
        contexts = self._contexts(features)
        output = []
        for start in range(0, len(contexts), batch_size):
            batch = torch.from_numpy(contexts[start : start + batch_size])
            channels = []
            for channel in range(2):
                embedding, _ = self.pipeline_.embed(batch[:, :, channel])
                channels.append(embedding.mean(dim=1))
            output.append(torch.cat(channels, dim=1))
        return torch.cat(output) if output else torch.empty((0, 512))

    def _encoded(self, context: torch.Tensor) -> torch.Tensor:
        assert self.pipeline_ is not None
        prepared = self.pipeline_._prepare_and_validate_context(context=context)
        token_ids, attention_mask, _ = self.pipeline_.tokenizer.context_input_transform(prepared)
        embedding = self.pipeline_.model.encode(
            input_ids=token_ids.to(self.pipeline_.model.device),
            attention_mask=attention_mask.to(self.pipeline_.model.device),
        )
        weights = attention_mask.to(embedding.device, dtype=embedding.dtype)[:, :, None]
        return torch.sum(embedding * weights, dim=1) / torch.clamp(torch.sum(weights, dim=1), 1.0)

    def _train_head(
        self,
        embeddings: torch.Tensor,
        labels: NDArray[np.integer],
        *,
        epochs: int,
        learning_rate: float,
    ) -> None:
        assert self.head_ is not None
        targets = torch.from_numpy(np.asarray(labels, dtype=np.float32))[:, None]
        positive_weight = torch.tensor(
            [float(np.sum(targets.numpy() == 0) / max(np.sum(targets.numpy() == 1), 1))]
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        optimizer = torch.optim.AdamW(self.head_.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            order = torch.randperm(len(embeddings), generator=generator)
            for start in range(0, len(order), 128):
                selected = order[start : start + 128]
                optimizer.zero_grad()
                loss = loss_fn(self.head_(embeddings[selected]), targets[selected])
                loss.backward()
                optimizer.step()

    def fit(self, features: NDArray[np.floating], labels: NDArray[np.integer]) -> ChronosClassifier:
        if self.mode not in {"frozen", "finetune"}:
            raise ValueError("Chronos mode must be frozen or finetune")
        self._load()
        assert self.pipeline_ is not None and self.head_ is not None
        if self.mode == "frozen":
            embeddings = self._frozen_embeddings(features)
            self._train_head(
                embeddings,
                labels,
                epochs=B2_FROZEN_HEAD_EPOCHS,
                learning_rate=0.003,
            )
            return self
        labels_array = np.asarray(labels, dtype=np.int8)
        rng = np.random.default_rng(self.seed)
        positives = np.flatnonzero(labels_array == 1)
        negatives = np.flatnonzero(labels_array == 0)
        positive_count = min(len(positives), B2_FINETUNE_MAX_WINDOWS // 2)
        negative_count = min(len(negatives), B2_FINETUNE_MAX_WINDOWS - positive_count)
        selected = np.concatenate(
            (
                rng.choice(positives, positive_count, replace=False),
                rng.choice(negatives, negative_count, replace=False),
            )
        )
        contexts = self._contexts(np.asarray(features)[selected])
        targets = torch.from_numpy(labels_array[selected].astype(np.float32))[:, None]
        negative_count = np.sum(labels_array[selected] == 0)
        positive_count = np.sum(labels_array[selected] == 1)
        positive_weight = torch.tensor([float(negative_count / max(positive_count, 1))])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        encoder = self.pipeline_.model.model.encoder
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(self.head_.parameters()), lr=1e-5
        )
        generator = torch.Generator().manual_seed(self.seed)
        encoder.train()
        for _ in range(B2_FINETUNE_EPOCHS):
            order = torch.randperm(len(selected), generator=generator)
            for start in range(0, len(order), 8):
                batch = order[start : start + 8]
                values = torch.from_numpy(contexts[batch])
                optimizer.zero_grad()
                channels = [self._encoded(values[:, :, channel]) for channel in range(2)]
                logits = self.head_(torch.cat(channels, dim=1))
                loss = loss_fn(logits, targets[batch])
                loss.backward()
                optimizer.step()
        encoder.eval()
        return self

    def predict_scores(
        self, features: NDArray[np.floating], *, batch_size: int = 32
    ) -> NDArray[np.float64]:
        if self.pipeline_ is None or self.head_ is None:
            raise RuntimeError("fit must be called before prediction")
        if not len(features):
            return np.empty(0, dtype=np.float64)
        self.head_.eval()
        with torch.no_grad():
            if self.mode == "frozen":
                embeddings = self._frozen_embeddings(features, batch_size=batch_size)
                return self.head_(embeddings)[:, 0].numpy().astype(np.float64)
            contexts = self._contexts(features)
            output = []
            for start in range(0, len(contexts), batch_size):
                values = torch.from_numpy(contexts[start : start + batch_size])
                channels = [self._encoded(values[:, :, channel]) for channel in range(2)]
                output.append(self.head_(torch.cat(channels, dim=1))[:, 0].numpy())
        return np.concatenate(output).astype(np.float64)
