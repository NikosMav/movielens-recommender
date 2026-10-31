"""SASRec mask, leakage, determinism, grid, and README panel (ADR-0019)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from movielens_recommender.sasrec.model import (  # noqa: E402
    SASRecModel,
    build_attention_mask,
)
from movielens_recommender.sasrec.plan import (  # noqa: E402
    base_grid,
    edge_extension_configs,
    find_edges,
    lock_budget,
    select_winner,
)
from movielens_recommender.sasrec.sequences import (  # noqa: E402
    build_sequence_data,
    events_by_user,
    take_window,
    training_target_ids,
    window_item_ids,
    windows_by_user,
)
from movielens_recommender.sasrec.train import train_sasrec  # noqa: E402
from movielens_recommender.split import SplitConfig, time_based_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _ratings() -> tuple[pd.DataFrame, int]:
    """Two users, unique items, plus one shared id that is user 1's test item.

    User 2 rates that shared id first, so it is in user 2's fit-train and must
    stay out of user 1's windows.
    """
    rows: list[dict[str, float | int]] = []
    shared = 10_000
    next_item = 1
    user1_items: list[int] = []
    for user, n in ((1, 12), (2, 12)):
        for t in range(n):
            if user == 2 and t == 0:
                item = shared
            else:
                item = next_item
                next_item += 1
            if user == 1:
                user1_items.append(item)
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "rating": 5.0 if t % 2 == 0 else 2.0,
                    "timestamp": t,
                }
            )
    # User 1's newest row is the test tail. Point user 2's oldest rating at it.
    user1_test_item = user1_items[-1]
    frame = pd.DataFrame(rows)
    frame.loc[(frame["user_id"] == 2) & (frame["timestamp"] == 0), "item_id"] = user1_test_item
    return frame, user1_test_item


def _hyperparams(**overrides: object) -> dict:
    row = {
        "batch_size": 4,
        "d_model": 8,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "loss": "full_softmax",
        "max_epochs": 1,
        "max_seq_len": 4,
        "n_heads": 1,
        "n_layers": 1,
        "patience": 3,
        "weight_decay": 0.0,
    }
    row.update(overrides)
    return row


def test_causal_mask_blocks_future_and_padding() -> None:
    tokens = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]])
    blocked = build_attention_mask(tokens)
    assert blocked.dtype == torch.bool
    assert blocked.shape == (2, 4, 4)
    for i in range(4):
        for j in range(4):
            assert bool(blocked[1, i, j].item()) is (j > i)
    assert blocked[0, 0].tolist() == [False, True, True, True]
    assert blocked[0, 1].tolist() == [True, False, True, True]
    assert blocked[0, 2].tolist() == [True, True, False, True]
    assert blocked[0, 3].tolist() == [True, True, False, False]
    # A pad query whose past is also pad can attend to itself, so softmax is defined.
    assert not bool(blocked[0].all(dim=-1).any())


def test_future_token_does_not_change_earlier_hidden_state() -> None:
    for n_heads in (1, 2):
        torch.manual_seed(0)
        model = SASRecModel(
            n_items=6,
            d_model=8,
            n_layers=2,
            n_heads=n_heads,
            dropout=0.0,
            max_seq_len=4,
        )
        model.eval()
        base = torch.tensor([[0, 1, 2, 3]])
        future = torch.tensor([[0, 1, 2, 4]])
        middle = torch.tensor([[0, 1, 5, 3]])
        with torch.no_grad():
            hidden = model(base)
            changed_future = model(future)
            changed_middle = model(middle)
        assert torch.isfinite(hidden).all()
        assert torch.equal(hidden[:, :-1], changed_future[:, :-1])
        assert not torch.equal(hidden[:, -1], changed_future[:, -1])
        assert torch.equal(hidden[:, :2], changed_middle[:, :2])
        assert not torch.equal(hidden[:, 2], changed_middle[:, 2])


def test_full_softmax_chunks_match_the_mean_and_the_step() -> None:
    tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    targets = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    mask = torch.tensor([[False, False, True, True], [False, True, False, True]])
    torch.manual_seed(0)
    model = SASRecModel(
        n_items=5, d_model=8, n_layers=1, n_heads=1, dropout=0.0, max_seq_len=4
    )
    model.eval()
    with torch.no_grad():
        hidden = model(tokens)
        whole = model.full_softmax_loss(hidden, targets, mask, chunk_positions=10)
        chunks = model.full_softmax_loss(hidden, targets, mask, chunk_positions=1)
        assert whole is not None and chunks is not None
        assert torch.allclose(whole, chunks)
        weight = model.item_emb.weight[1:]
        index = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        logits = torch.nn.functional.linear(hidden.reshape(-1, 8).index_select(0, index), weight)
        manual = torch.nn.functional.cross_entropy(
            logits, targets.reshape(-1).index_select(0, index), reduction="mean"
        )
        assert torch.allclose(whole, manual)
        assert model.full_softmax_loss(hidden, targets, torch.zeros_like(mask)) is None

    def _grads(chunk_positions: int) -> list[torch.Tensor]:
        torch.manual_seed(1)
        stepped = SASRecModel(
            n_items=5, d_model=8, n_layers=1, n_heads=1, dropout=0.0, max_seq_len=4
        )
        loss = stepped.full_softmax_loss(
            stepped(tokens), targets, mask, chunk_positions=chunk_positions
        )
        assert loss is not None
        grads = []
        for param in stepped.parameters():
            assert param.grad is not None
            grads.append(param.grad.detach().clone())
        return grads

    for left, right in zip(_grads(1), _grads(100), strict=True):
        assert torch.allclose(left, right, rtol=1e-5, atol=1e-5)


def test_fit_and_full_windows_exclude_later_slices() -> None:
    ratings, shared = _ratings()
    split = time_based_split(
        ratings,
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.1, relevance_threshold=4.0),
    )
    assert split.val is not None

    def _ids(frame: pd.DataFrame) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for uid, group in frame.groupby("user_id"):
            out[int(uid)] = set(int(item) for item in group["item_id"])
        return out

    fit_ids = _ids(split.train)
    val_ids = _ids(split.val)
    test_ids = _ids(split.test)
    full_ids = _ids(split.full_train)
    assert shared in test_ids[1]
    assert shared in fit_ids[2]

    for length in (4, 50):
        fit_windows = window_item_ids(split.train, length)
        full_windows = window_item_ids(split.full_train, length)
        fit_targets = training_target_ids(split.train, length)
        full_targets = training_target_ids(split.full_train, length)
        for uid, window in fit_windows.items():
            assert set(window).isdisjoint(val_ids[uid])
            assert set(window).isdisjoint(test_ids[uid])
            assert set(window) <= fit_ids[uid]
            assert set(fit_targets[uid]) <= set(window)
            assert set(fit_targets[uid]).isdisjoint(val_ids[uid])
            assert set(fit_targets[uid]).isdisjoint(test_ids[uid])
        for uid, window in full_windows.items():
            assert set(window).isdisjoint(test_ids[uid])
            assert set(window) <= full_ids[uid]
            assert set(full_targets[uid]) <= set(window)
            assert set(full_targets[uid]).isdisjoint(test_ids[uid])
        # The shared id is user 2's oldest fit-train item, so the long window keeps it
        # and user 1's window still does not.
        if length == 50:
            assert shared in fit_windows[2]
            assert shared not in fit_windows[1]
            assert shared not in full_windows[1]

        data = build_sequence_data(
            split.train, max_seq_len=length, relevance_threshold=4.0
        )
        catalog = {int(item): idx for idx, item in enumerate(data.item_ids)}
        assert shared not in set(data.item_ids) or shared in fit_ids[2]
        for row, uid in enumerate(data.train_user_ids.tolist()):
            positions = np.flatnonzero(data.loss_mask[row])
            for pos in positions:
                item = int(data.item_ids[int(data.targets[row, pos])])
                assert item in fit_ids[int(uid)]
                assert item not in val_ids[int(uid)]
                assert item not in test_ids[int(uid)]
                token = int(data.tokens[row, pos])
                if token != 0:
                    assert catalog[item] >= 0


def test_shuffle_follows_one_numpy_stream_in_user_order() -> None:
    ratings, _shared = _ratings()
    plain = windows_by_user(ratings, 5)
    shuffled = windows_by_user(ratings, 5, shuffle_seed=42)
    again = windows_by_user(ratings, 5, shuffle_seed=42)
    assert shuffled == again
    rng = np.random.default_rng(42)
    events = events_by_user(ratings)
    for uid in sorted(events):
        window = take_window(events[uid], 5)
        order = rng.permutation(len(window))
        expected = [window[int(i)] for i in order]
        assert shuffled[uid] == expected
        assert sorted(shuffled[uid]) == sorted(plain[uid])
    assert shuffled != plain


def test_one_epoch_is_deterministic() -> None:
    ratings, _shared = _ratings()
    hp = _hyperparams(dropout=0.2)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first, _data, first_result = train_sasrec(
            ratings, hyperparams=hp, seed=7, relevance_threshold=4.0, n_epochs=1
        )
        second, _data2, second_result = train_sasrec(
            ratings, hyperparams=hp, seed=7, relevance_threshold=4.0, n_epochs=1
        )
    finally:
        torch.set_num_threads(threads)
    assert first_result.history == second_result.history
    assert first_result.best_epoch == 1
    for key in first.model.state_dict():
        assert torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])
    for uid in sorted(ratings["user_id"].unique()):
        assert first.recommend(int(uid), 3) == second.recommend(int(uid), 3)
    assert torch.count_nonzero(first.model.item_emb.weight[0]) == 0


def test_train_trial_uses_the_requested_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sasrec

    seen: dict[str, int] = {}

    def fake_train(*_args: object, **kwargs: object) -> tuple[object, object, object]:
        seen["seed"] = int(kwargs["seed"])
        return object(), object(), object()

    monkeypatch.setattr(run_sasrec, "train_sasrec", fake_train)
    run_sasrec._train_trial(None, None, None, {}, relevance=4.0, n_epochs=2, seed=43)
    assert seen["seed"] == 43


def test_grid_lock_edges_and_ties() -> None:
    grid = base_grid()
    assert len(grid) == 8
    axes = [(row["max_seq_len"], row["n_layers"], row["dropout"]) for row in grid]
    assert axes == sorted(axes)
    assert axes[0] == (50, 1, 0.2)
    assert axes[-1] == (200, 2, 0.5)

    full = lock_budget(1.0, 1.0)
    assert full["fits"] is True
    assert full["cut"] is None
    assert full["n_configs"] == 8
    assert full["max_epochs"] == 30
    assert full["test_seeds"] == [42, 43, 44]

    dropped = lock_budget(1.0, float("inf"))
    assert dropped["fits"] is True
    assert dropped["cut"] == "dropped max_seq_len=200"
    assert dropped["n_configs"] == 4
    assert {row["max_seq_len"] for row in dropped["grid"]} == {50}

    epochs = lock_budget(40.0, float("inf"))
    assert epochs["fits"] is True
    assert epochs["cut"] == "dropped max_seq_len=200; max_epochs=15"
    assert epochs["max_epochs"] == 15

    one_seed = lock_budget(70.0, float("inf"))
    assert one_seed["fits"] is True
    assert one_seed["cut"] == "dropped max_seq_len=200; max_epochs=15; test seeds 42"
    assert one_seed["test_seeds"] == [42]

    layers = lock_budget(100.0, float("inf"))
    assert layers["fits"] is True
    assert layers["cut"] == (
        "dropped max_seq_len=200; max_epochs=15; test seeds 42; dropped n_layers=2"
    )
    assert layers["n_configs"] == 2
    assert {row["n_layers"] for row in layers["grid"]} == {1}

    assert lock_budget(300.0, float("inf"))["fits"] is False

    high = dict(grid[-1])
    extensions = edge_extension_configs(high, grid)
    assert [row["n_layers"] for row in extensions] == [3, 2]
    assert [row["dropout"] for row in extensions] == [0.5, 0.8]
    assert all(row["max_seq_len"] == 200 for row in extensions)
    assert edge_extension_configs(dict(grid[0]), grid) == []
    # Length 200 is already the grid high end, so it is not stepped.
    assert edge_extension_configs(dict(grid[4]), grid) == []

    short = base_grid(max_epochs=15, drop_len_200=True, drop_layers_2=True)
    high_dropout = dict(short[-1])
    assert find_edges(high_dropout, short) == [
        {"parameter": "dropout", "side": "high", "value": 0.5}
    ]
    only = edge_extension_configs(high_dropout, short)
    assert len(only) == 1
    assert only[0]["dropout"] == 0.8
    assert only[0]["n_layers"] == 1
    assert only[0]["max_seq_len"] == 50

    trials = [
        {"name": "a", "val_ndcg@10": 0.1000004},
        {"name": "b", "val_ndcg@10": 0.1000006},
        {"name": "c", "val_ndcg@10": 0.2},
        {"name": "d", "val_ndcg@10": 0.2},
    ]
    assert select_winner(trials)["name"] == "c"


def _pair(mean: float, low: float, high: float) -> dict:
    return {
        "excludes_zero": bool(round(low, 6) > 0.0 or round(high, 6) < 0.0),
        "high": high,
        "low": low,
        "mean": mean,
        "n_users": 4,
    }


def _ci(mean: float, low: float, high: float) -> dict:
    return {"high": high, "low": low, "mean": mean}


def _results_payload(*, negative: bool, edges: list, truncated: bool = False) -> dict:
    return {
        "budget": {
            "cut": "dropped max_seq_len=200",
            "ease_configs": 8,
            "n_configs": 4,
            "n_edge_extensions": 1,
            "n_trials_scored": 3 if truncated else 4,
            "truncated": truncated,
            "two_tower_grid_configs": 8,
        },
        "decision": {"negative_result": negative, "order_helps": not negative},
        "edge_extension_trials": [
            {
                "hyperparams": {"dropout": 0.8, "max_seq_len": 50, "n_layers": 2},
                "val_ndcg@10": 0.05,
            }
        ],
        "models": {
            "lambdarank": {
                "best_iteration": 95,
                "ndcg@10": 0.44,
                "ndcg@10_ci": _ci(0.44, 0.41, 0.47),
            },
            "sasrec": {
                "ndcg@10_mean": 0.33,
                "ndcg@10_std": 0.01,
                "primary_ndcg@10_ci": _ci(0.331, 0.3, 0.36),
            },
            "sasrec_shuffled": {"ndcg@10": 0.22, "ndcg@10_ci": _ci(0.22, 0.19, 0.25)},
            "two_tower": {"ndcg@10": 0.55, "ndcg@10_ci": _ci(0.55, 0.51, 0.59)},
        },
        "paired": {
            "real_minus_shuffled": (
                _pair(0.11, 0.02, 0.2) if not negative else _pair(0.01, -0.02, 0.04)
            ),
            "sasrec_minus_lambdarank": _pair(-0.11, -0.16, -0.06),
            "sasrec_minus_two_tower": _pair(-0.22, -0.28, -0.16),
        },
        "published": {
            "lambdarank": {"ndcg@10": 0.127264},
            "two_tower": {"ndcg@10": 0.119246},
        },
        "seeds": {"bootstrap": 42, "shuffle": 42, "test": [42, 43, 44], "tuning": 42},
        "selected": {
            "base_grid_edges": [],
            "best_epoch": 4,
            "grid_edges": edges,
            "hyperparams": {
                "d_model": 64,
                "dropout": 0.2,
                "max_seq_len": 50,
                "n_heads": 1,
                "n_layers": 2,
            },
            "val_ndcg@10": 0.181,
            "val_ndcg@10_ci": _ci(0.181, 0.15, 0.21),
        },
    }


def test_readme_panel_is_three_sentences_from_the_json(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_results_table import _fmt, render_sasrec, sasrec_reading

    payload = _results_payload(
        negative=True,
        edges=[{"parameter": "dropout", "side": "low", "value": 0.2}],
    )
    reading = sasrec_reading(payload)
    assert len(reading) == 3
    assert all(sentence.endswith(".") for sentence in reading)
    text = "\n".join(reading)
    assert _fmt(payload["selected"]["val_ndcg@10"]) in text
    assert _fmt(payload["models"]["sasrec"]["ndcg@10_mean"]) in text
    assert _fmt(payload["models"]["sasrec"]["ndcg@10_std"]) in text
    assert _fmt(payload["models"]["sasrec"]["primary_ndcg@10_ci"]["mean"]) in text
    assert _fmt(payload["models"]["sasrec_shuffled"]["ndcg@10"]) in text
    assert _fmt(payload["models"]["two_tower"]["ndcg@10"]) in text
    assert _fmt(payload["models"]["lambdarank"]["ndcg@10"]) in text
    assert _fmt(payload["paired"]["real_minus_shuffled"]["mean"]) in text
    assert _fmt(payload["paired"]["sasrec_minus_two_tower"]["mean"]) in text
    assert _fmt(payload["paired"]["sasrec_minus_lambdarank"]["mean"]) in text
    assert _fmt(payload["published"]["two_tower"]["ndcg@10"]) in text
    assert _fmt(payload["published"]["lambdarank"]["ndcg@10"]) in text
    assert "best epoch 4" in reading[0]
    assert "max_seq_len 50" in reading[0]
    assert "Negative result" in reading[2]
    assert "low dropout edge (0.2)" in reading[2]
    assert "0.1192" not in reading[0]

    path = tmp_path / "ml-1m.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    panel = render_sasrec(path)
    assert panel == render_sasrec(path)
    for sentence in reading:
        assert sentence in panel
    assert "Validation configs in the locked grid: 4." in panel
    assert "The two-tower grid had 8 and EASE had 8." in panel
    assert "Budget cut: dropped max_seq_len=200." in panel
    assert "Seeds: tuning 42, test 42, 43 and 44, shuffle 42, bootstrap 42." in panel
    assert (
        "Edge-extension trials scored after the base grid, under the ADR-0019 "
        "one-step rule: 1 (the dropout step scored lower "
        "(validation NDCG@10 0.0500))."
    ) in panel
    assert "| SASRec | 0.1810 | 0.3300 | [0.3000, 0.3600] |" in panel
    assert render_sasrec(tmp_path / "missing.json") == ""
    rendered = render_sasrec()
    results_path = ROOT / "results" / "sasrec" / "ml-1m.json"
    if not results_path.is_file():
        assert rendered == ""
    else:
        assert rendered.strip()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        start = readme.index("### SASRec (`ml-1m`)")
        end = readme.index("<!-- END RESULTS TABLE -->", start)
        assert rendered.strip() == readme[start:end].strip()
        committed = json.loads(results_path.read_text(encoding="utf-8"))
        headline = json.loads((ROOT / "results" / "ml-1m.json").read_text(encoding="utf-8"))
        delta = float(committed["paired"]["real_minus_shuffled"]["mean"])
        shuffled_ndcg = float(committed["models"]["sasrec_shuffled"]["ndcg@10"])
        relative = int(round(100.0 * delta / shuffled_ndcg))
        assert f"+{_fmt(delta)} NDCG@10, about {relative}% relative" in rendered
        popular = headline["metrics"]["most_popular"]
        sasrec = committed["models"]["sasrec"]
        assert f"coverage@10 of {_fmt(float(sasrec['metrics']['coverage@10']))}" in rendered
        assert f"most-popular's {_fmt(float(popular['coverage@10']))}" in rendered
        assert f"most-popular's {_fmt(float(popular['ndcg@10']))}" in rendered
        assert "stays unresolved" in rendered
        assert "replaced the leader" in rendered
        assert "the dropout step scored lower" in rendered
        assert "under-trained" in rendered
        assert "twice the size" in rendered

    clear = _results_payload(negative=False, edges=[])
    assert sasrec_reading(clear)[2].startswith("Order helps on this split")
    assert "no edge of the searched axes" in sasrec_reading(clear)[2]
    stepped = _results_payload(negative=True, edges=[], truncated=True)
    stepped["selected"]["base_grid_edges"] = [
        {"parameter": "n_layers", "side": "high", "value": 2}
    ]
    stepped_reading = sasrec_reading(stepped)
    assert "after one step past the high n_layers edge (2)" in stepped_reading[2]
    assert "stopped after 3 of 4 configs" in stepped_reading[2]
