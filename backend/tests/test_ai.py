import pytest

import torch

from backend.game.board import Board, EMPTY, P1, P2
from backend.ml.model_agent import ModelMover, LookaheadMover
from backend.ml.predictor import OpponentPredictor
from training.torch_loader import load_model, TorchModelAdapter
from training.eval import model_move, evaluate_vs_random


@pytest.fixture
def tiny_model(tmp_path):
    """Train a tiny model quickly on random history, save it, reload."""
    from training.generate import random_games, to_tensors
    from training.model import policy_loss, value_loss, ValuePolicyTransformer
    from torch.utils.data import DataLoader, TensorDataset

    model = ValuePolicyTransformer(3, d_model=32, num_layers=1)
    data = random_games(3, 60, seed=1)
    boards, masks, moves, values = to_tensors(data)
    loader = DataLoader(TensorDataset(boards, masks, moves, values), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        for x, m, mv, v in loader:
            opt.zero_grad()
            vl, pl = model(x, m)
            (policy_loss(pl, mv) + value_loss(vl, v)).backward()
            opt.step()
    path = tmp_path / "ck.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"n": 3, "d_model": 32, "num_layers": 1}}, path)
    return str(path)


def test_model_mover_returns_legal_move(tiny_model):
    b = Board(3)
    model, err = load_model(tiny_model, 3)
    assert err is None
    mover = ModelMover(b, model=TorchModelAdapter(model), difficulty="hard")
    assert mover.error is None
    for _ in range(20):
        move = mover(P1)
        assert 0 <= move < 27
        assert b.cells[move] == EMPTY


def test_model_mover_fails_without_checkpoint():
    b = Board(3)
    mover = ModelMover(b, path="/nonexistent/model.bin")
    assert mover.error is not None
    with pytest.raises(RuntimeError):
        mover(P1)


def test_model_mover_respects_occupied(tiny_model):
    b = Board(3)
    b.set(0, 0, 0, P1)
    model, err = load_model(tiny_model, 3)
    assert err is None
    mover = ModelMover(b, model=TorchModelAdapter(model))
    move = mover(P2)
    assert b.cells[move] == EMPTY


def test_move_distribution_sums_to_one(tiny_model):
    b = Board(3)
    model, err = load_model(tiny_model, 3)
    assert err is None
    mover = ModelMover(b, model=TorchModelAdapter(model))
    dist = mover.move_distribution(P1)
    legal = [c == EMPTY for c in b.cells]
    assert abs(sum(dist[i] for i in range(27) if legal[i]) - 1.0) < 1e-4
    # illegal cells must be ~0 (empty board has none, so guard)
    if any(not l for l in legal):
        assert max(dist[i] for i in range(27) if not legal[i]) <= 1e-6


def test_predictor_returns_legal_distribution(tiny_model):
    b = Board(3)
    model, err = load_model(tiny_model, 3)
    assert err is None
    pred = OpponentPredictor(b, TorchModelAdapter(model))
    dist = pred.predict_distribution(P1)
    assert dist
    assert all(0.0 <= p <= 1.0 for _, p in dist)
    # top-8 is a truncated slice; full legal sum is 1.0
    full = pred.predict_distribution(P1, top_k=27)
    assert abs(sum(p for _, p in full) - 1.0) < 1e-4
    for idx, p in dist:
        assert b.cells[idx] == EMPTY


def test_eval_vs_random_runs(tiny_model):
    model, err = load_model(tiny_model, 3)
    assert err is None
    res = evaluate_vs_random(model, 3, games=6, seed=2)
    assert res["games"] == 6
    assert abs(res["wins"] + res["losses"] + res["draws"] - 1.0) < 1e-6


def _lookahead(path, board, difficulty="hard"):
    model, err = load_model(path, board.n)
    assert err is None
    engine = TorchModelAdapter(model)
    pred = OpponentPredictor(board, engine)
    return LookaheadMover(engine, board, pred, difficulty=difficulty)


def test_lookahead_takes_immediate_win(tiny_model):
    b = Board(3)
    b.set(0, 0, 0, P1)
    b.set(1, 0, 0, P1)
    mover = _lookahead(tiny_model, b)
    assert mover(P1) == b.idx(2, 0, 0)


def test_lookahead_blocks_immediate_loss(tiny_model):
    b = Board(3)
    b.set(0, 0, 0, P2)
    b.set(1, 0, 0, P2)
    mover = _lookahead(tiny_model, b)
    assert mover(P1) == b.idx(2, 0, 0)


def test_lookahead_returns_legal(tiny_model):
    b = Board(3)
    mover = _lookahead(tiny_model, b)
    for _ in range(3):
        m = mover(P1)
        assert b.cells[m] == EMPTY