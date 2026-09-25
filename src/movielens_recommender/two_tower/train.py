"""Train / tune the two-tower model (validation only; never test)."""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from movielens_recommender.evaluate import ndcg_point_estimate
from movielens_recommender.split import SplitResult
from movielens_recommender.two_tower.features import (
    TwoTowerFeatures,
    build_features,
    pad_histories,
)
from movielens_recommender.two_tower.model import TwoTowerModel
from movielens_recommender.two_tower.recommender import TwoTowerRecommender

# Small, documented grid (ADR-0006). Primary selection metric: validation NDCG@10.
TWO_TOWER_GRID: list[dict[str, Any]] = [
    {
        "embedding_dim": d,
        "learning_rate": lr,
        "temperature": t,
        "batch_size": 1024,
        "weight_decay": 1e-4,
        "max_epochs": 20,
        "patience": 3,
        "max_history": 50,
    }
    for d, lr, t in itertools.product([32, 64], [1e-3, 3e-3], [0.05, 0.1])
]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - import already failed
        raise ImportError(
            "PyTorch is required for the two-tower model. "
            "Install with: pip install '.[deep]' "
            "(CPU: pip install torch==2.6.0 --index-url "
            "https://download.pytorch.org/whl/cpu)."
        )


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sampled_softmax_loss(
    logits: torch.Tensor,
    log_q: torch.Tensor,
) -> torch.Tensor:
    """In-batch sampled softmax with log-q correction.

    ``logits[b, j]`` = score(user_b, item_j) for items in the batch.
    Positives lie on the diagonal. ``log_q`` is log sampling prob per batch item.
    """
    corrected = logits - log_q.unsqueeze(0)
    # Diagonal positives.
    targets = torch.arange(corrected.size(0), device=corrected.device)
    return nn.functional.cross_entropy(corrected, targets)


def _batch_loss(
    model: TwoTowerModel,
    features: TwoTowerFeatures,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    device: torch.device,
    max_history: int,
) -> torch.Tensor:
    u_np = user_idx.cpu().numpy()
    i_np = item_idx.cpu().numpy()
    hist, mask = pad_histories(
        u_np, features, exclude_item_idx=i_np, max_history=max_history
    )
    hist_t = torch.from_numpy(hist).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    user_vec = model.encode_users(user_idx, hist_t, mask_t)

    genres = torch.from_numpy(features.genres[i_np]).to(device)
    years = torch.from_numpy(features.years[i_np]).to(device)
    item_vec = model.encode_items(item_idx, genres, years)

    # (B, B) scores: each user against every in-batch item.
    logits = (user_vec @ item_vec.T) / model.temperature
    q = torch.from_numpy(features.item_q[i_np].astype(np.float32)).to(device)
    log_q = torch.log(q.clamp(min=1e-12))
    return sampled_softmax_loss(logits, log_q)


@dataclass
class TrainResult:
    """Outcome of a single training run (with optional early stopping)."""

    best_epoch: int
    epochs_trained: int
    best_val_ndcg10: float | None
    history: list[dict[str, Any]]
    wall_time_sec: float


def train_two_tower(
    features: TwoTowerFeatures,
    *,
    hyperparams: Mapping[str, Any],
    seed: int = 42,
    val_split: SplitResult | None = None,
    relevance_threshold: float = 4.0,
    device: str | None = None,
    show_progress: bool = False,
) -> tuple[TwoTowerModel, TrainResult]:
    """Train on ``features``; optionally early-stop on validation NDCG@10.

    When ``val_split`` is provided, ``val_split.train`` must be the same matrix
    used to build ``features`` (fit-train). Validation rows are never folded
    into history.
    """
    _require_torch()
    set_torch_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    emb = int(hyperparams.get("embedding_dim", 64))
    lr = float(hyperparams.get("learning_rate", 1e-3))
    temperature = float(hyperparams.get("temperature", 0.1))
    batch_size = int(hyperparams.get("batch_size", 1024))
    weight_decay = float(hyperparams.get("weight_decay", 1e-4))
    max_epochs = int(hyperparams.get("max_epochs", 20))
    patience = int(hyperparams.get("patience", 3))
    max_history = int(hyperparams.get("max_history", 50))

    model = TwoTowerModel(
        n_users=features.n_users,
        n_items=features.n_items,
        n_genres=features.n_genres,
        embedding_dim=emb,
        temperature=temperature,
    ).to(dev)

    dataset = TensorDataset(
        torch.from_numpy(features.pos_user_idx),
        torch.from_numpy(features.pos_item_idx),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(dataset) > batch_size else False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val = float("-inf")
    stale = 0
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for user_idx, item_idx in loader:
            user_idx = user_idx.to(dev)
            item_idx = item_idx.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model, features, user_idx, item_idx, dev, max_history
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        record: dict[str, Any] = {"epoch": epoch, "train_loss": round(mean_loss, 6)}

        if val_split is not None and val_split.val is not None:
            model.eval()
            rec = TwoTowerRecommender.from_trained(
                model, features, max_history=max_history, device=str(dev)
            )
            score = ndcg_point_estimate(
                rec.recommend,
                val_split.train,
                val_split.val,
                relevance_threshold=relevance_threshold,
                k=10,
                split=SplitResult(
                    train=val_split.train,
                    test=val_split.val,
                    config=val_split.config,
                    n_users_kept=val_split.n_users_kept,
                    n_users_dropped=val_split.n_users_dropped,
                ),
            )
            record["val_ndcg@10"] = round(score, 6)
            if show_progress:
                print(
                    f"    epoch {epoch}/{max_epochs} "
                    f"loss={mean_loss:.4f} val_ndcg@10={score:.4f}",
                    flush=True,
                )
            if score > best_val + 1e-6:
                best_val = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    history.append(record)
                    break
        elif show_progress:
            print(
                f"    epoch {epoch}/{max_epochs} loss={mean_loss:.4f}",
                flush=True,
            )
            best_epoch = epoch
        else:
            best_epoch = epoch

        history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)
    elif val_split is None:
        # Fixed-epoch refit: last epoch is the intended stopping point.
        best_epoch = max_epochs if not history else history[-1]["epoch"]

    wall = time.perf_counter() - t0
    result = TrainResult(
        best_epoch=int(best_epoch),
        epochs_trained=int(history[-1]["epoch"] if history else 0),
        best_val_ndcg10=None if best_val == float("-inf") else float(best_val),
        history=history,
        wall_time_sec=round(wall, 3),
    )
    model.eval()
    return model, result


def fit_two_tower_recommender(
    train: pd.DataFrame,
    *,
    dataset: str,
    data_dir: str = "data",
    movies: pd.DataFrame | None = None,
    hyperparams: Mapping[str, Any],
    seed: int = 42,
    relevance_threshold: float = 4.0,
    n_epochs: int | None = None,
    val_split: SplitResult | None = None,
    device: str | None = None,
    show_progress: bool = False,
) -> tuple[TwoTowerRecommender, TwoTowerFeatures, TrainResult]:
    """Build features, train, wrap as a recommender.

    For refit on full-train, pass ``n_epochs`` (from early stopping) and leave
    ``val_split=None`` so training runs a fixed epoch count with no val peeking.
    """
    hp = dict(hyperparams)
    if n_epochs is not None:
        hp["max_epochs"] = int(n_epochs)
        hp["patience"] = int(n_epochs) + 1  # disable early stop

    features = build_features(
        train,
        dataset=dataset,
        data_dir=data_dir,
        movies=movies,
        relevance_threshold=relevance_threshold,
        max_history=int(hp.get("max_history", 50)),
    )
    model, result = train_two_tower(
        features,
        hyperparams=hp,
        seed=seed,
        val_split=val_split,
        relevance_threshold=relevance_threshold,
        device=device,
        show_progress=show_progress,
    )
    rec = TwoTowerRecommender.from_trained(
        model,
        features,
        max_history=int(hp.get("max_history", 50)),
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return rec, features, result


@dataclass(frozen=True)
class TwoTowerTuningResult:
    model: str
    primary_metric: str
    grid: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    best_hyperparams: dict[str, Any]
    best_val_score: float
    best_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "primary_metric": self.primary_metric,
            "grid": self.grid,
            "trials": self.trials,
            "best_hyperparams": self.best_hyperparams,
            "best_val_score": round(float(self.best_val_score), 6),
            "best_epoch": int(self.best_epoch),
            "early_stopping": (
                "Train on fit-train; stop when validation NDCG@10 does not "
                "improve for `patience` epochs. Refit on full-train uses "
                "best_epoch as the fixed epoch count (no validation)."
            ),
        }


def tune_two_tower(
    split: SplitResult,
    *,
    dataset: str,
    data_dir: str = "data",
    movies: pd.DataFrame | None = None,
    relevance_threshold: float = 4.0,
    seed: int = 42,
    grid: Sequence[Mapping[str, Any]] | None = None,
    show_progress: bool = True,
) -> TwoTowerTuningResult:
    """Grid-search two-tower on validation NDCG@10; never touches test."""
    if split.val is None or split.val.empty:
        raise ValueError("tune_two_tower requires a non-empty validation split")

    configs = [dict(c) for c in (grid if grid is not None else TWO_TOWER_GRID)]
    trials: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_hp = dict(configs[0])
    best_epoch = 1

    val_split = SplitResult(
        train=split.train,
        test=split.val,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )

    for i, hp in enumerate(configs, start=1):
        if show_progress:
            print(f"  [two_tower] trial {i}/{len(configs)}: {hp}", flush=True)
        _rec, _feat, result = fit_two_tower_recommender(
            split.train,
            dataset=dataset,
            data_dir=data_dir,
            movies=movies,
            hyperparams=hp,
            seed=seed,
            relevance_threshold=relevance_threshold,
            val_split=val_split,
            show_progress=show_progress,
        )
        score = (
            float("-inf")
            if result.best_val_ndcg10 is None
            else float(result.best_val_ndcg10)
        )
        trials.append(
            {
                "hyperparams": hp,
                "val_ndcg@10": round(score, 6),
                "best_epoch": result.best_epoch,
                "epochs_trained": result.epochs_trained,
                "wall_time_sec": result.wall_time_sec,
            }
        )
        if score > best_score:
            best_score = score
            best_hp = dict(hp)
            best_epoch = int(result.best_epoch)

    return TwoTowerTuningResult(
        model="two_tower",
        primary_metric="ndcg@10",
        grid=configs,
        trials=trials,
        best_hyperparams=best_hp,
        best_val_score=best_score,
        best_epoch=best_epoch,
    )
