"""Train SASRec on one allowed slice and score it with the harness (ADR-0019)."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from movielens_recommender.baselines.common import top_n_indices
from movielens_recommender.evaluate import (
    _prepare_eval,
    collect_recommendations,
    evaluate_recommender,
    ndcg_point_estimate,
)
from movielens_recommender.metrics import ndcg_at_k
from movielens_recommender.sasrec.model import SASRecModel, zero_padding_row
from movielens_recommender.sasrec.plan import EPOCH_SHUFFLE
from movielens_recommender.sasrec.sequences import SequenceData, build_sequence_data
from movielens_recommender.split import SplitResult

ENCODE_BATCH = 256


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainResult:
    best_epoch: int
    epochs_trained: int
    best_val_ndcg10: float | None
    history: list[dict[str, Any]]
    wall_time_sec: float


class SASRecRecommender:
    """Exact top-n over the slice catalog. Seen items are masked, then ``top_n_indices``."""

    def __init__(
        self,
        model: SASRecModel,
        data: SequenceData,
        *,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.data = data
        self.device = torch.device(device)
        self._scores: np.ndarray | None = None
        self._row = {int(uid): i for i, uid in enumerate(data.infer_user_ids.tolist())}

    def _ensure_scores(self) -> np.ndarray:
        if self._scores is not None:
            return self._scores
        self.model.eval()
        tokens = self.data.infer_tokens
        pieces: list[torch.Tensor] = []
        weight = self.model.item_emb.weight[1:]
        with torch.no_grad():
            for start in range(0, len(tokens), ENCODE_BATCH):
                batch = torch.from_numpy(tokens[start : start + ENCODE_BATCH]).to(self.device)
                last = self.model(batch)[:, -1, :]
                pieces.append(torch.nn.functional.linear(last, weight).cpu())
        self._scores = torch.cat(pieces, dim=0).numpy().astype(np.float64, copy=False)
        return self._scores

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0:
            return []
        row = self._row.get(int(user_id))
        if row is None:
            return []
        scores = self._ensure_scores()[row].copy()
        seen = self.data.seen.get(int(user_id))
        if seen is not None and len(seen):
            columns = [
                self.data.item_index[int(item)]
                for item in seen
                if int(item) in self.data.item_index
            ]
            if columns:
                scores[np.asarray(columns, dtype=np.int64)] = -np.inf
        chosen = top_n_indices(scores, n)
        return [int(self.data.item_ids[i]) for i in chosen]


def _val_view(train: pd.DataFrame, val: pd.DataFrame, template: SplitResult) -> SplitResult:
    return SplitResult(
        train=train,
        test=val,
        config=template.config,
        n_users_kept=template.n_users_kept,
        n_users_dropped=template.n_users_dropped,
    )


def _run_epochs(
    model: SASRecModel,
    data: SequenceData,
    *,
    hyperparams: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int | None,
    val_frame: pd.DataFrame | None,
    val_template: SplitResult | None,
    relevance_threshold: float,
    show_progress: bool,
) -> tuple[SASRecModel, TrainResult]:
    """Train ``max_epochs`` steps. Early stop only when ``val_frame`` is set."""
    device = torch.device("cpu")
    model = model.to(device)
    batch_size = int(hyperparams["batch_size"])
    lr = float(hyperparams["learning_rate"])
    weight_decay = float(hyperparams["weight_decay"])
    dataset = TensorDataset(
        torch.from_numpy(data.tokens),
        torch.from_numpy(data.targets),
        torch.from_numpy(data.loss_mask),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val = float("-inf")
    stale = 0
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(int(seed) + epoch * EPOCH_SHUFFLE)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )
        losses: list[float] = []
        for tokens, targets, mask in loader:
            opt.zero_grad(set_to_none=True)
            hidden = model(tokens)
            loss = model.full_softmax_loss(hidden, targets, mask)
            if loss is None:
                continue
            opt.step()
            zero_padding_row(model)
            losses.append(float(loss))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        record: dict[str, Any] = {"epoch": epoch, "train_loss": round(mean_loss, 6)}

        if val_frame is not None and val_template is not None:
            model.eval()
            recommender = SASRecRecommender(model, data, device="cpu")
            score = ndcg_point_estimate(
                recommender.recommend,
                val_template.train,
                val_frame,
                relevance_threshold=relevance_threshold,
                k=10,
                split=_val_view(val_template.train, val_frame, val_template),
            )
            record["val_ndcg@10"] = round(float(score), 6)
            if show_progress:
                print(
                    f"    epoch {epoch}/{max_epochs} loss={mean_loss:.4f} "
                    f"val_ndcg@10={score:.4f}",
                    flush=True,
                )
            if score > best_val + 1e-6:
                best_val = float(score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                history.append(record)
                if patience is not None and stale >= patience:
                    break
                continue
        elif show_progress:
            print(f"    epoch {epoch}/{max_epochs} loss={mean_loss:.4f}", flush=True)

        history.append(record)
        if val_frame is None:
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    zero_padding_row(model)
    wall = time.perf_counter() - t0
    result = TrainResult(
        best_epoch=int(best_epoch),
        epochs_trained=int(history[-1]["epoch"] if history else 0),
        best_val_ndcg10=None if best_val == float("-inf") else float(best_val),
        history=history,
        wall_time_sec=round(wall, 3),
    )
    return model, result


def train_sasrec(
    frame: pd.DataFrame,
    *,
    hyperparams: dict[str, Any],
    seed: int,
    relevance_threshold: float = 4.0,
    val_frame: pd.DataFrame | None = None,
    val_template: SplitResult | None = None,
    n_epochs: int | None = None,
    shuffle_seed: int | None = None,
    show_progress: bool = False,
) -> tuple[SASRecRecommender, SequenceData, TrainResult]:
    """Fit on ``frame``. Pass ``val_frame`` to early-stop; pass ``n_epochs`` to refit.

    ``n_epochs`` disables the validation pass, including when ``val_frame`` is set.
    ``shuffle_seed`` permutes each window (the shuffled-order check).
    """
    set_torch_seed(seed)
    max_seq_len = int(hyperparams["max_seq_len"])
    data = build_sequence_data(
        frame,
        max_seq_len=max_seq_len,
        relevance_threshold=relevance_threshold,
        shuffle_seed=shuffle_seed,
    )
    if len(data.train_user_ids) == 0:
        raise ValueError("no user has a positive next-item target in this slice")
    model = SASRecModel(
        n_items=int(len(data.item_ids)),
        d_model=int(hyperparams["d_model"]),
        n_layers=int(hyperparams["n_layers"]),
        n_heads=int(hyperparams["n_heads"]),
        dropout=float(hyperparams["dropout"]),
        max_seq_len=max_seq_len,
    )
    if n_epochs is not None:
        trained, result = _run_epochs(
            model,
            data,
            hyperparams=hyperparams,
            seed=seed,
            max_epochs=int(n_epochs),
            patience=None,
            val_frame=None,
            val_template=None,
            relevance_threshold=relevance_threshold,
            show_progress=show_progress,
        )
    else:
        trained, result = _run_epochs(
            model,
            data,
            hyperparams=hyperparams,
            seed=seed,
            max_epochs=int(hyperparams["max_epochs"]),
            patience=int(hyperparams["patience"]),
            val_frame=val_frame,
            val_template=val_template,
            relevance_threshold=relevance_threshold,
            show_progress=show_progress,
        )
    return SASRecRecommender(trained, data, device="cpu"), data, result


def per_user_ndcg(
    recommend_fn: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    relevance_threshold: float,
    split: SplitResult,
    k: int = 10,
) -> tuple[list[int], list[float]]:
    """Harness user order and per-user NDCG@k. The order is ``relevant`` insertion order."""
    _split, _seen, relevant = _prepare_eval(
        train,
        test,
        relevance_threshold=relevance_threshold,
        split=split,
    )
    recommendations = collect_recommendations(recommend_fn, _seen, relevant, max_k=k)
    user_ids = [int(uid) for uid in relevant]
    values = [float(ndcg_at_k(recommendations[uid], relevant[uid], k)) for uid in user_ids]
    return user_ids, values


def evaluate_split(
    recommend_fn: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    relevance_threshold: float,
    split: SplitResult,
    n_bootstrap: int,
    bootstrap_alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Existing harness on one split. Coverage stays a point estimate."""
    from movielens_recommender.evaluate import format_metrics

    metrics = evaluate_recommender(
        recommend_fn,
        train,
        test,
        relevance_threshold=relevance_threshold,
        ks=(10, 20),
        n_bootstrap=n_bootstrap,
        bootstrap_alpha=bootstrap_alpha,
        seed=seed,
        split=split,
        include_segments=False,
    )
    return format_metrics(metrics)


def time_one_epoch(
    frame: pd.DataFrame,
    val_users_frame: pd.DataFrame,
    *,
    hyperparams: dict[str, Any],
    seed: int,
    relevance_threshold: float,
) -> float:
    """One training epoch plus one validation-user encode. No ranking metric.

    The model is discarded by the caller. This is the ADR-0019 probe.
    """
    set_torch_seed(seed)
    max_seq_len = int(hyperparams["max_seq_len"])
    data = build_sequence_data(
        frame,
        max_seq_len=max_seq_len,
        relevance_threshold=relevance_threshold,
        shuffle_seed=None,
    )
    if len(data.train_user_ids) == 0:
        raise ValueError("probe frame has no training targets")
    model = SASRecModel(
        n_items=int(len(data.item_ids)),
        d_model=int(hyperparams["d_model"]),
        n_layers=int(hyperparams["n_layers"]),
        n_heads=int(hyperparams["n_heads"]),
        dropout=float(hyperparams["dropout"]),
        max_seq_len=max_seq_len,
    )
    # Encode the users who appear in the validation frame, from fit-train windows.
    val_ids = set(int(uid) for uid in val_users_frame["user_id"].unique())
    keep = [i for i, uid in enumerate(data.infer_user_ids.tolist()) if int(uid) in val_ids]
    if not keep:
        raise ValueError("probe encode found no validation user in the fit-train windows")
    encode_tokens = data.infer_tokens[np.asarray(keep, dtype=np.int64)]
    t0 = time.perf_counter()
    _run_epochs(
        model,
        data,
        hyperparams=hyperparams,
        seed=seed,
        max_epochs=1,
        patience=None,
        val_frame=None,
        val_template=None,
        relevance_threshold=relevance_threshold,
        show_progress=False,
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encode_tokens), ENCODE_BATCH):
            batch = torch.from_numpy(encode_tokens[start : start + ENCODE_BATCH])
            model(batch)
    return time.perf_counter() - t0
