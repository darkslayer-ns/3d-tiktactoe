"""Model-based mover: the ONLY game AI. Plays using the trained
transformer's policy head.

Inference runs on the C++ engine (libmodel.so) through the `eval_position`
seam — the server process loads no PyTorch. Movers accept any object
implementing `eval_position(cells, player, n) -> (value_logit, policy_logits)`:
the server passes `CppModel`; training/lab code wraps torch graphs with
`training.torch_loader.TorchModelAdapter`.

Difficulty maps to lookahead depth, a wrong-move budget, and sampling
temperatures. There is no search fallback — if the C++ weights are missing
the game cannot be created.
"""

from __future__ import annotations

import random as _random

from backend.game.board import Board, EMPTY, P1, P2
from backend.ml.cpp_inference import (
    argmax,
    load_cpp_model,
    sample_index,
    sigmoid,
    softmax,
)

DIFFICULTY_TEMPERATURE = {
    "easy": 1.5,
    "medium": 0.7,
    "hard": 0.0,
}

# Difficulty is pure RUNTIME config — all difficulties share the same strong
# base weights. "depth" = how many of the opponent's replies the AI predicts
# ahead (easy sees 2, hard sees 4). "wrong_move_budget" = how many random
# blunders the AI is allowed per game. "move_temp" = softmax temperature for
# sampling the final move from the top scores (keeps play varied on repeated
# positions; hard stays near-greedy). entry_temp varies the opening.
DIFFICULTY = {
    # blunder_kind: 'random' (EASY — blunders any non-forced move),
    # 'suboptimal' (MEDIUM — one mistake, only when about to win), 'none' (HARD).
    "easy": {
        "depth": 2, "top_k": 3, "sampling": False, "max_nodes": 220,
        "entry_temp": 1.0, "entry_moves": 2, "wrong_move_budget": 5, "mistake_rate": 0.5,
        "move_temp": 0.8, "blunder_kind": "random",
    },
    "medium": {
        "depth": 3, "top_k": 3, "sampling": False, "max_nodes": 220,
        "entry_temp": 0.6, "entry_moves": 1, "wrong_move_budget": 1, "mistake_rate": 0.35,
        "move_temp": 0.5, "blunder_kind": "suboptimal",
    },
    "hard": {
        "depth": 4, "top_k": 3, "sampling": False, "max_nodes": 220,
        "entry_temp": 0.3, "entry_moves": 1, "wrong_move_budget": 0, "mistake_rate": 0.0,
        "move_temp": 0.15, "blunder_kind": "none",
    },
}


def _require_engine(model):
    """Return (engine, error): the engine must implement eval_position().

    The server passes the C++ CppModel; training/lab code must wrap torch
    graphs with training.torch_loader.TorchModelAdapter first.
    """
    if model is None:
        return None, "no model"
    if not hasattr(model, "eval_position"):
        return None, "model must implement eval_position() — wrap torch graphs with training.torch_loader.TorchModelAdapter"
    return model, None


class ModelMover:
    """A callable agent that moves via the trained network (C++ engine)."""

    def __init__(
        self,
        board: Board,
        path=None,
        model=None,
        temperature: float = 0.0,
        difficulty: str = "hard",
    ):
        self.board = board
        self.temperature = DIFFICULTY_TEMPERATURE.get(difficulty, temperature)
        if model is not None:
            self.model, self.error = _require_engine(model)
        else:
            self.model, self.error = load_cpp_model(board.n, path)

    def __call__(self, player: int) -> int:
        if self.model is None:
            raise RuntimeError(self.error)
        _, plogits = self.model.eval_position(self.board.cells, player)
        if self.temperature <= 0:
            return argmax(plogits)
        probs = softmax(plogits, self.temperature)
        return sample_index(probs)

    def move_distribution(self, player: int) -> list:
        """Full policy probability over cells for the given side to move."""
        if self.model is None:
            raise RuntimeError(self.error)
        _, plogits = self.model.eval_position(self.board.cells, player)
        return softmax(plogits, 1.0)


class LookaheadMover:
    """Model-guided lookahead (expectimax over the predicted opponent).

    The AI does not just sample its policy. It:
      1. tries every legal move,
      2. for each, predicts the human's most likely replies (top_k from
         the opponent predictor) and the probability of each,
      3. rolls 2-3 plies deep (AI move -> human reply -> AI reply),
      4. evaluates the resulting position with the model's value head,
      5. picks the move with the highest expected value.

    This makes the model "think ahead" instead of acting randomly.
    """

    def __init__(self, model, board, predictor, difficulty: str = "hard"):
        self.model, self.error = _require_engine(model)
        self.board = board
        self.predictor = predictor
        cfg = DIFFICULTY.get(difficulty, DIFFICULTY["hard"])
        self.depth = cfg["depth"]
        self.top_k = cfg["top_k"]
        self.sampling = cfg["sampling"]
        self.max_nodes = cfg["max_nodes"]
        self.entry_temp = cfg.get("entry_temp", 0.0)
        self.entry_moves = cfg.get("entry_moves", 0)
        self._nodes = 0
        self.mistake_rate = cfg.get("mistake_rate", 0.0)
        self.shift_rate = 0.0
        self.wrong_move_budget = cfg.get("wrong_move_budget", 0)
        self.move_temp = cfg.get("move_temp", 0.0)
        self.blunder_kind = cfg.get("blunder_kind", "random")
        self._wrong_moves_used = 0

    def _forward(self, board: Board, player: int):
        if self.model is None:
            raise RuntimeError(self.error)
        value_logit, policy = self.model.eval_position(board.cells, player)
        return sigmoid(value_logit), policy

    def _evaluate(self, board: Board, player: int):
        """One forward: (win-prob for player, greedy move for player)."""
        value, policy = self._forward(board, player)
        return value, argmax(policy)

    def _greedy(self, board: Board, player: int) -> int:
        _, policy = self._forward(board, player)
        return argmax(policy)

    def _value_for(self, board: Board, player: int) -> float:
        v, _ = self._forward(board, player)
        return v

    def _terminal(self, board: Board, ai: int) -> float | None:
        w, _, over = board.outcome()
        if not over:
            return None
        if w == ai:
            return 1.0
        if w == EMPTY:
            return 0.5
        return 0.0

    def _opponent_expected(self, board: Board, ai: int, depth: int, top_k: int | None = None) -> float:
        """Expected value (for AI) after the opponent's likely replies.

        Operates on a board COPY — the live game board is never mutated, so
        a concurrent search can't corrupt it.

        `depth` = number of AI reply plies still to search. AI answers
        greedily at each level; branching spans the opponent's top-k likely
        moves, and top_k shrinks with depth so the total work stays bounded.
        """
        if top_k is None:
            top_k = self.top_k
        self._nodes += 1
        if self._nodes > self.max_nodes:
            v, _ = self._evaluate(board, ai)
            return v

        opp = P2 if ai == P1 else P1
        dist = self.predictor.likely_moves(opp, top_k=max(1, top_k), board=board)
        if not dist:
            v, _ = self._evaluate(board, ai)
            return v

        total = sum(p for _, p in dist)
        exp = 0.0
        next_top_k = max(1, top_k - 1)
        for r, pr in dist:
            w = pr / total
            board.apply(r, opp)
            t = self._terminal(board, ai)
            if t is not None:
                rv = t
            elif depth > 1:
                val, a = self._evaluate(board, ai)
                board.apply(a, ai)
                rv = self._opponent_expected(board, ai, depth - 1, next_top_k)
                board.cells[a] = EMPTY
            else:
                val, _ = self._evaluate(board, ai)
                rv = val
            exp += w * rv
            board.cells[r] = EMPTY
        return exp

    def _eval_move(self, board: Board, m: int, ai: int, depth: int | None = None) -> float:
        """Expected value of AI playing m on `board`, searching `depth` plies."""
        if depth is None:
            depth = self.depth
        self._nodes += 1
        if self._nodes > self.max_nodes:
            opp = P2 if ai == P1 else P1
            v = 1.0 - self._value_for(board, opp)
            board.cells[m] = EMPTY
            return v
        board.apply(m, ai)
        t = self._terminal(board, ai)
        if t is not None:
            board.cells[m] = EMPTY
            return t
        if depth <= 1:
            opp = P2 if ai == P1 else P1
            v = 1.0 - self._value_for(board, opp)
            board.cells[m] = EMPTY
            return v
        v = self._opponent_expected(board, ai, depth)
        board.cells[m] = EMPTY
        return v

    def _shifted_move(self, best: int) -> int:
        """An empty box adjacent (Chebyshev ≤ 1) to `best`, or `best` if none."""
        n = self.board.n
        bz, br = divmod(best, n * n)
        by, bx = divmod(br, n)
        cands = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = bx + dx, by + dy, bz + dz
                    if 0 <= nx < n and 0 <= ny < n and 0 <= nz < n:
                        i = nx + n * (ny + n * nz)
                        if self.board.cells[i] == EMPTY:
                            cands.append(i)
        if not cands:
            return best
        return _random.choice(cands)

    def _pick_blunder(self, moves, scored, best):
        """The deliberate-mistake cell. 'random' plays anywhere (EASY); 'suboptimal'
        (MEDIUM) plays a good-but-not-best move, never the winning cell itself."""
        if self.blunder_kind == "suboptimal":
            if scored and len(scored) > 1:
                rank = 1 + int(_random.random() * min(len(scored) - 1, 3))
                return scored[rank][0]
            others = [m for m in moves if m != best]
            return _random.choice(others) if others else _random.choice(moves)
        return _random.choice(moves)

    def __call__(self, ai: int) -> int:
        moves = self.board.moves()
        if not moves:
            return -1

        # fully random (no search) — the weakest possible player
        if self.mistake_rate >= 1.0:
            final = _random.choice(moves)
            self._record_decision(ai, final, None, "random")
            return final

        best = self._strong_move(ai, moves)
        scored = getattr(self, "_last_scored", None)
        forced = getattr(self, "_last_forced", None)

        # Deliberate mistakes, gated by difficulty rules:
        #  - easy (random): may blunder any move EXCEPT when it would win or must
        #    block (never throws away a win / never misses a block).
        #  - medium (suboptimal): `wrong_move_budget` (1) mistakes, only when it
        #    is ABOUT TO WIN — gifts the win once, but never fails to block.
        kind = getattr(self, "blunder_kind", "random")
        should_blunder = False
        if self.mistake_rate > 0 and self._wrong_moves_used < self.wrong_move_budget:
            if kind == "random":
                should_blunder = forced is None and _random.random() < self.mistake_rate
            elif kind == "suboptimal":
                should_blunder = forced == "win" and _random.random() < self.mistake_rate
        if should_blunder:
            self._wrong_moves_used += 1
            final = self._pick_blunder(moves, scored, best)
            self._record_decision(ai, final, scored, "blunder")
            return final

        # "shift by a box": statistically play an empty box adjacent to the
        # suggested one instead of the exact box — the AI stays in the right
        # neighbourhood but makes an off-by-one spatial mistake.
        if self.shift_rate > 0 and _random.random() < self.shift_rate:
            shifted = self._shifted_move(best)
            if shifted != best:
                self._record_decision(ai, shifted, scored, "shift")
                return shifted
        self._record_decision(ai, best, scored, "search")
        return best

    def _record_decision(self, ai, chosen, scored, kind: str) -> None:
        """Store the AI's chosen move + top candidate scores for the admin view."""
        top = [
            (int(m), round(float(v), 3))
            for m, v in (scored or [])[:5]
        ]
        self.last_decision = {
            "player": ai,
            "chosen": int(chosen),
            "coord": list(self.board.coord(chosen)),
            "kind": kind,
            "value": round(float(top[0][1]), 3) if top else None,
            "depth": self.depth,
            "scored": top,
            "line": self._predicted_line(ai, chosen),
        }

    def _predicted_line(self, ai: int, chosen: int) -> list:
        """Best-guess future line if the AI plays `chosen`.

        Alternates the opponent's most likely reply (from the predictor) with
        the AI's greedy counter for ~`depth` replies, so the admin can see
        exactly what sequence the AI is thinking ahead.
        """
        from backend.game.board import Board

        rb = Board(self.board.n, list(self.board.cells))
        line = [{"player": ai, "index": int(chosen), "coord": list(rb.coord(chosen))}]
        rb.apply(chosen, ai)
        player = P2 if ai == P1 else P1
        plies = 0
        while plies < self.depth * 2 and not rb.outcome()[2]:
            if player == ai:
                idx = self._greedy(rb, player)
            else:
                dist = self.predictor.likely_moves(player, top_k=1, board=rb)
                if not dist:
                    break
                idx = dist[0][0]
            line.append({"player": player, "index": int(idx), "coord": list(rb.coord(idx))})
            rb.apply(idx, player)
            player = P2 if player == P1 else P1
            plies += 1
        return line

    def _strong_move(self, ai: int, moves) -> int:
        """The intended strong move: immediate win, block, or search pick.

        All search happens on a COPY of the board so the live board is
        never mutated.
        """
        from backend.game.board import Board as _Board

        search = _Board(self.board.n, list(self.board.cells))
        opp = P2 if ai == P1 else P1
        self._nodes = 0
        self._last_scored = None
        self._last_forced = None

        # immediate win
        for m in moves:
            search.apply(m, ai)
            w, _, over = search.outcome()
            search.cells[m] = EMPTY
            if w == ai:
                self._last_forced = "win"
                return m

        # immediate block: opponent wins next move unless we take that cell
        for m in moves:
            search.apply(m, opp)
            w, _, over = search.outcome()
            search.cells[m] = EMPTY
            if w == opp:
                self._last_forced = "block"
                return m

        scored = sorted(self._scored(ai, moves, search), key=lambda x: -x[1])
        self._last_scored = scored

        # opening-phase temperature: sample among good moves so the AI does
        # not always open the same way (e.g. always center)
        moves_played = self.board.n**3 - self.board.empty_count()
        if self.entry_temp > 0 and self.entry_moves > 0 and moves_played < self.entry_moves and len(scored) > 1:
            top = scored[: min(5, len(scored))]
            probs = softmax([v for _, v in top], self.entry_temp)
            return top[sample_index(probs)][0]

        if self.sampling and len(scored) > 1:
            top = scored[:3]
            probs = softmax([v for _, v in top], 1.0 / 3.0)
            return top[sample_index(probs)][0]

        # sample the final move from the top scores so repeated positions still
        # produce varied play; hard stays near-greedy
        if self.move_temp > 0 and len(scored) > 1:
            top = scored[: min(3, len(scored))]
            probs = softmax([v for _, v in top], self.move_temp)
            return top[sample_index(probs)][0]
        return scored[0][0]

    def _scored(self, ai: int, moves, search: Board) -> list:
        """(move, expected-value) list from the lookahead search.

        Runs on a COPY (`search`) so the live board is never touched. The
        opening variety + wrong-move blunders are applied OUTSIDE this list.
        """
        return [(m, self._eval_move(search, m, ai)) for m in moves]