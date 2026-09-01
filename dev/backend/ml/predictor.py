"""Model-based opponent predictor.

Predicts where the human is likely to move next, purely from the trained
network: candidate moves are scored by the net's value head (how good the
position becomes for the human if they play there), blended with the
player's learned affinity. Probabilities come from softmax, so the UI can
show "the model expects cell X at 34%".

Inference goes through the `eval_position(cells, player, n)` seam (C++ engine
in the server; torch adapter in lab scripts) — this module imports no PyTorch.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from backend.game.board import Board, EMPTY, P1, P2
from backend.ml.cpp_inference import sigmoid, softmax


class OpponentPredictor:
    def __init__(self, board: Board, model, decay: float = 0.9):
        from backend.ml.model_agent import _require_engine

        self.board = board
        self.model, self.error = _require_engine(model)
        self.decay = decay
        self.affinity: dict[int, dict[int, float]] = {}
        self.played: dict[int, list[int]] = {P1: [], P2: []}

    def record(self, player: int, cell: int) -> None:
        self.played.setdefault(player, []).append(cell)
        aff = self.affinity.setdefault(player, {})
        aff[cell] = aff.get(cell, 0.0) + 1.0

    def new_game(self) -> None:
        for player in self.affinity:
            for k in list(self.affinity[player].keys()):
                self.affinity[player][k] *= self.decay
        self.played = {P1: [], P2: []}

    def _value_of(self, board: Board, player: int) -> float:
        """Net's win-probability for `player` on this position."""
        if self.model is None:
            raise RuntimeError(self.error)
        value_logit, _ = self.model.eval_position(board.cells, player)
        return sigmoid(value_logit)

    def _score_cell(self, player: int, idx: int, board: Board) -> float:
        aff = self.affinity.get(player, {}).get(idx, 0.0)
        board.apply(idx, player)
        strength = self._value_of(board, player)
        board.cells[idx] = EMPTY
        return aff + strength

    def predict_distribution(
        self,
        player: int,
        temperature: float = 1.0,
        top_k: int = 8,
        board: Board | None = None,
    ) -> List[Tuple[int, float]]:
        board = board or self.board
        scored = [(idx, self._score_cell(player, idx, board)) for idx in board.moves()]
        if not scored:
            return []
        mx = max(s for _, s in scored)
        exps = [(idx, math.exp((s - mx) / temperature)) for idx, s in scored]
        total = sum(e for _, e in exps)
        ranked = sorted(((idx, e / total) for idx, e in exps), key=lambda x: -x[1])
        return ranked[:top_k]

    def predict_next(self, player: int, board: Board | None = None) -> Optional[int]:
        dist = self.predict_distribution(player, top_k=1, board=board)
        return dist[0][0] if dist else None

    def likely_moves(self, player: int, top_k: int = 5, board: Board | None = None):
        """Fast opponent-reply prediction straight from the model's policy head.

        One forward pass (vs scoring every move with the value head), so it
        is cheap enough to call many times inside lookahead search.
        """
        if self.model is None:
            raise RuntimeError(self.error)
        board = board or self.board
        _, plogits = self.model.eval_position(board.cells, player)
        probs = softmax(plogits, 1.0)
        scored = [(i, probs[i]) for i in range(len(probs)) if board.cells[i] == EMPTY]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]