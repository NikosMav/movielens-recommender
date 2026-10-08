"""Train / tune the two-tower model (validation only; never test)."""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from movielens_recommender.evaluate import ndcg_point_estimate
from movielens_recommender.scale import release_memory
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


# ``in_batch`` is the ADR-0006 objective and the default. The other two are
# ADR-0013. Omitting ``loss`` must keep the in-batch path.
LOSS_IN_BATCH = "in_batch"
LOSS_FULL_SOFTMAX = "full_softmax"
LOSS_SAMPLED_SOFTMAX = "sampled_softmax"
TWO_TOWER_LOSSES: tuple[str, ...] = (
    LOSS_IN_BATCH,
    LOSS_FULL_SOFTMAX,
    LOSS_SAMPLED_SOFTMAX,
)
DEFAULT_N_NEGATIVES = 256


def resolve_two_tower_loss(hyperparams: Mapping[str, Any] | None) -> str:
    """Return the training loss name. Missing means the in-batch reference."""
    if not hyperparams or hyperparams.get("loss") in (None, ""):
        return LOSS_IN_BATCH
    name = str(hyperparams["loss"])
    if name not in TWO_TOWER_LOSSES:
        raise ValueError(
            f"two-tower loss must be one of {TWO_TOWER_LOSSES}; got {name!r}"
        )
    return name


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


def full_softmax_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over every catalog item.

    ``logits`` is ``(batch, n_items)`` and already temperature-scaled.
    ``targets`` is the positive catalog index per row. There is no log-q
    term: the denominator is the whole catalog.
    """
    return nn.functional.cross_entropy(logits, targets)


def sampled_softmax_logq_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    positive_log_q: torch.Tensor,
    negative_log_q: torch.Tensor,
    *,
    positive_index: torch.Tensor | None = None,
    negative_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sampled softmax with a log-q popularity correction.

    ``positive_logits`` is ``(batch,)``. ``negative_logits`` is ``(batch, K)``.
    ``positive_log_q`` is ``(batch,)``. ``negative_log_q`` is ``(K,)`` or
    ``(batch, K)``. The corrected logit is ``score - log q``. The positive is
    class 0. When both index tensors are set, a negative that equals that
    row's positive is masked so the label is not copied into the denominator.
    """
    if positive_logits.ndim != 1:
        raise ValueError("positive_logits must have shape (batch,)")
    if negative_logits.ndim != 2 or negative_logits.shape[0] != positive_logits.shape[0]:
        raise ValueError("negative_logits must have shape (batch, K)")
    pos = (positive_logits - positive_log_q).unsqueeze(1)
    neg_log_q = negative_log_q.unsqueeze(0) if negative_log_q.ndim == 1 else negative_log_q
    neg = negative_logits - neg_log_q
    if positive_index is not None and negative_index is not None:
        collision = negative_index.unsqueeze(0) == positive_index.unsqueeze(1)
        neg = neg.masked_fill(collision, torch.finfo(neg.dtype).min)
    logits = torch.cat([pos, neg], dim=1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return nn.functional.cross_entropy(logits, targets)


@dataclass
class CatalogTensors:
    """Catalog-side tensors reused by full softmax and sampled softmax."""

    idx: torch.Tensor
    genres: torch.Tensor
    years: torch.Tensor
    q: torch.Tensor
    log_q: torch.Tensor


def build_catalog_tensors(
    features: TwoTowerFeatures, device: torch.device
) -> CatalogTensors:
    """Move catalog features and the popularity distribution onto ``device``.

    ``q`` is renormalized after the float32 cast so multinomial gets a
    probability vector. The in-batch loss does not use this helper.
    """
    q = torch.from_numpy(np.ascontiguousarray(features.item_q, dtype=np.float32))
    q = q.clamp(min=1e-12)
    q = q / q.sum()
    return CatalogTensors(
        idx=torch.arange(features.n_items, device=device),
        genres=torch.from_numpy(
            np.ascontiguousarray(features.genres, dtype=np.float32)
        ).to(device),
        years=torch.from_numpy(
            np.ascontiguousarray(features.years, dtype=np.float32)
        ).to(device),
        q=q.to(device),
        log_q=torch.log(q).to(device),
    )


def _encode_batch_users(
    model: TwoTowerModel,
    features: TwoTowerFeatures,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    device: torch.device,
    max_history: int,
    user_id_dropout: float,
) -> torch.Tensor:
    """User vectors for one batch, positive item removed from history."""
    u_np = user_idx.detach().cpu().numpy()
    i_np = item_idx.detach().cpu().numpy()
    hist, mask = pad_histories(
        u_np, features, exclude_item_idx=i_np, max_history=max_history
    )
    zero_user_id = None
    if user_id_dropout > 0.0 and model.training and not model.history_only:
        zero_user_id = torch.bernoulli(
            torch.full((user_idx.shape[0],), float(user_id_dropout), device=device)
        )
    return model.encode_users(
        user_idx,
        torch.from_numpy(hist).to(device),
        torch.from_numpy(mask).to(device),
        zero_user_id=zero_user_id,
    )


def _full_softmax_batch_loss(
    model: TwoTowerModel,
    features: TwoTowerFeatures,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    device: torch.device,
    max_history: int,
    user_id_dropout: float,
    catalog: CatalogTensors,
) -> torch.Tensor:
    user_vec = _encode_batch_users(
        model, features, user_idx, item_idx, device, max_history, user_id_dropout
    )
    item_vec = model.encode_items(catalog.idx, catalog.genres, catalog.years)
    logits = (user_vec @ item_vec.T) / model.temperature
    return full_softmax_loss(logits, item_idx)


def _sampled_softmax_batch_loss(
    model: TwoTowerModel,
    features: TwoTowerFeatures,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    device: torch.device,
    max_history: int,
    user_id_dropout: float,
    catalog: CatalogTensors,
    n_negatives: int,
) -> torch.Tensor:
    """Negatives are shared across the batch and drawn from ``q``."""
    k = min(int(n_negatives), int(features.n_items) - 1)
    if k < 1:
        raise ValueError("sampled softmax needs at least two catalog items")
    user_vec = _encode_batch_users(
        model, features, user_idx, item_idx, device, max_history, user_id_dropout
    )
    neg_idx = torch.multinomial(catalog.q, k, replacement=False)
    all_idx = torch.cat([item_idx, neg_idx], dim=0)
    vecs = model.encode_items(all_idx, catalog.genres[all_idx], catalog.years[all_idx])
    n_pos = int(item_idx.shape[0])
    pos_vec = vecs[:n_pos]
    neg_vec = vecs[n_pos:]
    pos_logit = (user_vec * pos_vec).sum(dim=-1) / model.temperature
    neg_logit = (user_vec @ neg_vec.T) / model.temperature
    return sampled_softmax_logq_loss(
        pos_logit,
        neg_logit,
        catalog.log_q[item_idx],
        catalog.log_q[neg_idx],
        positive_index=item_idx,
        negative_index=neg_idx,
    )


def _batch_loss(
    model: TwoTowerModel,
    features: TwoTowerFeatures,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    device: torch.device,
    max_history: int,
    user_id_dropout: float = 0.0,
    *,
    loss: str = LOSS_IN_BATCH,
    n_negatives: int = DEFAULT_N_NEGATIVES,
    catalog: CatalogTensors | None = None,
) -> torch.Tensor:
    # The default branch below is the ADR-0006 in-batch loss, unchanged.
    if loss != LOSS_IN_BATCH:
        if catalog is None:
            raise ValueError(f"loss={loss!r} requires catalog tensors")
        if loss == LOSS_FULL_SOFTMAX:
            return _full_softmax_batch_loss(
                model,
                features,
                user_idx,
                item_idx,
                device,
                max_history,
                user_id_dropout,
                catalog,
            )
        if loss == LOSS_SAMPLED_SOFTMAX:
            return _sampled_softmax_batch_loss(
                model,
                features,
                user_idx,
                item_idx,
                device,
                max_history,
                user_id_dropout,
                catalog,
                n_negatives,
            )
        raise ValueError(
            f"two-tower loss must be one of {TWO_TOWER_LOSSES}; got {loss!r}"
        )
    u_np = user_idx.cpu().numpy()
    i_np = item_idx.cpu().numpy()
    hist, mask = pad_histories(
        u_np, features, exclude_item_idx=i_np, max_history=max_history
    )
    hist_t = torch.from_numpy(hist).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    zero_user_id = None
    if user_id_dropout > 0.0 and model.training and not model.history_only:
        zero_user_id = torch.bernoulli(
            torch.full(
                (user_idx.shape[0],),
                float(user_id_dropout),
                device=device,
            )
        )
    user_vec = model.encode_users(
        user_idx, hist_t, mask_t, zero_user_id=zero_user_id
    )

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


def _epoch_payload(
    epoch: int,
    model: TwoTowerModel,
    opt: torch.optim.Optimizer,
    best_state: dict[str, Any] | None,
    best_epoch: int,
    best_val: float,
    stale: int,
    history: list[dict[str, Any]],
    prior_wall: float,
    started: float,
    *,
    finished: bool,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "best_state": best_state,
        "best_epoch": int(best_epoch),
        "best_val": None if best_val == float("-inf") else float(best_val),
        "stale": int(stale),
        "history": history,
        "wall_time_sec": round(prior_wall + (time.perf_counter() - started), 3),
        "finished": bool(finished),
    }


def _save_epoch_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_epoch_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def train_two_tower(
    features: TwoTowerFeatures,
    *,
    hyperparams: Mapping[str, Any],
    seed: int = 42,
    val_split: SplitResult | None = None,
    relevance_threshold: float = 4.0,
    device: str | None = None,
    show_progress: bool = False,
    score_without_user_id: bool = False,
    epoch_checkpoint: str | Path | None = None,
) -> tuple[TwoTowerModel, TrainResult]:
    """Train on ``features``; optionally early-stop on validation NDCG@10.

    When ``val_split`` is provided, ``val_split.train`` must be the same matrix
    used to build ``features`` (fit-train). Validation rows are never folded
    into history.

    ``epoch_checkpoint`` is rewritten after every completed epoch. A resume
    loads weights and optimizer state and continues at the next epoch. The
    DataLoader is shuffled again, so a resumed run is not bit-exact.
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
    user_id_dropout = float(hyperparams.get("user_id_dropout", 0.0))
    history_only = bool(hyperparams.get("history_only", False))
    cold_eval = bool(score_without_user_id) or history_only
    loss_name = resolve_two_tower_loss(hyperparams)
    n_negatives = int(hyperparams.get("n_negatives", DEFAULT_N_NEGATIVES))
    if n_negatives < 1:
        raise ValueError("n_negatives must be >= 1")

    model = TwoTowerModel(
        n_users=features.n_users,
        n_items=features.n_items,
        n_genres=features.n_genres,
        embedding_dim=emb,
        temperature=temperature,
        history_only=history_only,
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
    # Built only for the ADR-0013 losses. The in-batch path does not read it.
    catalog = None if loss_name == LOSS_IN_BATCH else build_catalog_tensors(features, dev)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val = float("-inf")
    stale = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    prior_wall = 0.0
    ckpt_path = None if epoch_checkpoint is None else Path(epoch_checkpoint)
    saved = _load_epoch_checkpoint(ckpt_path) if ckpt_path is not None else None
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        best_epoch = int(saved["best_epoch"])
        best_val = float("-inf") if saved["best_val"] is None else float(saved["best_val"])
        stale = int(saved["stale"])
        history = list(saved["history"])
        best_state = saved["best_state"]
        prior_wall = float(saved.get("wall_time_sec", 0.0))
        start_epoch = int(saved["epoch"]) + 1
        if show_progress:
            print(f"    resumed after epoch {saved['epoch']}", flush=True)
        if saved.get("finished"):
            start_epoch = max_epochs + 1
    t0 = time.perf_counter()

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for step, (user_idx, item_idx) in enumerate(loader, start=1):
            user_idx = user_idx.to(dev)
            item_idx = item_idx.to(dev)
            opt.zero_grad(set_to_none=True)
            loss_kwargs: dict[str, Any] = {}
            if loss_name != LOSS_IN_BATCH:
                loss_kwargs = {
                    "loss": loss_name,
                    "n_negatives": n_negatives,
                    "catalog": catalog,
                }
            loss = _batch_loss(
                model,
                features,
                user_idx,
                item_idx,
                dev,
                max_history,
                user_id_dropout=user_id_dropout,
                **loss_kwargs,
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if show_progress and step % 2000 == 0:
                print(f"    epoch {epoch} batch {step}", flush=True)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        record: dict[str, Any] = {"epoch": epoch, "train_loss": round(mean_loss, 6)}

        if val_split is not None and val_split.val is not None:
            model.eval()
            rec = TwoTowerRecommender.from_trained(
                model, features, max_history=max_history, device=str(dev)
            )
            rec.score_without_user_id = cold_eval
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
                if ckpt_path is not None:
                    _save_epoch_checkpoint(
                        ckpt_path,
                        _epoch_payload(
                            epoch,
                            model,
                            opt,
                            best_state,
                            best_epoch,
                            best_val,
                            stale,
                            history,
                            prior_wall,
                            t0,
                            finished=True,
                        ),
                    )
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
        if ckpt_path is not None:
            _save_epoch_checkpoint(
                ckpt_path,
                _epoch_payload(
                    epoch,
                    model,
                    opt,
                    best_state,
                    best_epoch,
                    best_val,
                    stale,
                    history,
                    prior_wall,
                    t0,
                    finished=epoch == max_epochs,
                ),
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    elif val_split is None:
        # Fixed-epoch refit: last epoch is the intended stopping point.
        best_epoch = max_epochs if not history else history[-1]["epoch"]

    wall = prior_wall + (time.perf_counter() - t0)
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
    epoch_checkpoint: str | Path | None = None,
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
    cold_eval = bool(hp.get("score_without_user_id", False)) or bool(
        hp.get("history_only", False)
    )
    model, result = train_two_tower(
        features,
        hyperparams=hp,
        seed=seed,
        val_split=val_split,
        relevance_threshold=relevance_threshold,
        device=device,
        show_progress=show_progress,
        score_without_user_id=cold_eval,
        epoch_checkpoint=epoch_checkpoint,
    )
    rec = TwoTowerRecommender.from_trained(
        model,
        features,
        max_history=int(hp.get("max_history", 50)),
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        hyperparams=hp,
    )
    rec.score_without_user_id = cold_eval or bool(model.history_only)
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
        del _rec, _feat, result
        release_memory()

    return TwoTowerTuningResult(
        model="two_tower",
        primary_metric="ndcg@10",
        grid=configs,
        trials=trials,
        best_hyperparams=best_hp,
        best_val_score=best_score,
        best_epoch=best_epoch,
    )
