"""Game session manager: holds board state, runs agents, streams events."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Optional

from backend.game.board import Board, EMPTY, P1, P2
from backend.ml.model_agent import LookaheadMover
from backend.ml.predictor import OpponentPredictor

AGENTS = ("model",)
DIFFICULTIES = ("easy", "medium", "hard")


class GameSession:
    def __init__(
        self,
        game_id: str,
        size: int,
        mode: str,  # "pve" | "ave"
        difficulty: str = "medium",
        x_agent: str = "alphabeta",
        o_agent: str = "alphabeta",
        ai_delay: float = 0.4,
        predictor_enabled: bool = False,
        human_side: int = P1,
    ):
        self.id = game_id
        self.board = Board(size)
        self.mode = mode
        self.difficulty = difficulty
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.ai_delay = ai_delay
        self.predictor_enabled = predictor_enabled
        self.subscribers: List[asyncio.Queue] = []
        self.current_player = P1
        self.winner = EMPTY
        self.winning_line: Optional[List] = None
        self.over = False
        self.started_at = time.time()
        self.thinking = False
        self._human_side = human_side if mode == "pve" else None
        self._moves: List[Dict] = []
        self._logged = False

        self.x_mover, self.o_mover = self._build_movers()
        # predictor is always active: used for forecasts AND by the
        # lookahead mover to model the opponent's likely replies
        self.predictor = (
            OpponentPredictor(self.board, self.x_mover.model)
            if predictor_enabled or mode == "pve"
            else None
        )

    # -- agents ------------------------------------------------------------
    def _build_movers(self):
        """Both sides share one C++ engine; difficulty is runtime config."""
        from backend.ml.cpp_inference import load_cpp_model

        model, err = load_cpp_model(self.board.n)
        if err:
            raise ValueError(err)
        predictor = OpponentPredictor(self.board, model)
        mk = lambda: LookaheadMover(model, self.board, predictor, difficulty=self.difficulty)
        return mk(), mk()

    @property
    def agent_note(self) -> Optional[str]:
        """Explain what each side actually plays."""
        return f"C++ engine (cpp/model.bin), lookahead {self.x_mover.depth} plies, {self.x_mover.wrong_move_budget} mistake(s)/game"

    # -- events ------------------------------------------------------------
    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    async def _emit(self, event: Dict) -> None:
        for q in list(self.subscribers):
            await q.put(event)

    def snapshot(self) -> Dict:
        return {
            "id": self.id,
            "size": self.board.n,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "x_agent": self.x_agent,
            "o_agent": self.o_agent,
            "cells": list(self.board.cells),
            "current_player": self.current_player,
            "winner": self.winner,
            "winning_line": self.winning_line,
            "over": self.over,
            "thinking": self.thinking,
            "human_side": self._human_side,
            "agent_note": self.agent_note,
        }

    def admin_info(self) -> Dict:
        """Admin view: state + the AI's last decision on each side."""
        return {
            **self.snapshot(),
            "move_count": len(self._moves),
            "moves": self._moves[-8:],
            "x_decision": getattr(self.x_mover, "last_decision", None),
            "o_decision": getattr(self.o_mover, "last_decision", None),
        }

    # -- game actions ------------------------------------------------------
    async def start(self) -> None:
        await self._emit({"type": "started", "game": self.snapshot()})
        if self.mode == "ave":
            await self._run_ai_vs_ai()
        elif self._human_side is not None and self.current_player != self._human_side:
            # AI plays first (human chose O)
            await self._run_ai_turn()

    async def human_move(self, index: int) -> Dict:
        if self.mode != "pve":
            return {"ok": False, "error": "not a human game"}
        if self.over or self.thinking:
            return {"ok": False, "error": "game over or AI thinking"}
        if self.current_player != self._human_side:
            return {"ok": False, "error": "not your turn"}
        result = await self._place(index, self.current_player)
        if result.get("ok") and not self.over:
            asyncio.create_task(self._run_ai_turn())
        return result

    async def _run_ai_turn(self) -> None:
        """In PvE, let the AI respond exactly ONCE (fire-and-forget task)."""
        if self.over or self.thinking or self.current_player == self._human_side:
            return
        delay = max(0.0, self.ai_delay)
        await asyncio.sleep(delay)
        if self.over or self.thinking or self.current_player == self._human_side:
            return

        player = self.current_player
        mover = self.x_mover if player == P1 else self.o_mover
        self.thinking = True
        await self._emit({"type": "thinking", "player": player, "game": self.snapshot()})
        try:
            loop = asyncio.get_event_loop()
            index = await loop.run_in_executor(None, mover, player)
        finally:
            self.thinking = False
        if not (0 <= index < self.board.n**3) or self.board.cells[index] != EMPTY:
            # illegal/duplicate AI move: never place, never loop again
            await self._emit({"type": "thinking", "player": player, "game": self.snapshot()})
            return
        await self._place(index, player)

    async def _place(self, index: int, player: int) -> Dict:
        if not (0 <= index < self.board.n**3):
            return {"ok": False, "error": "index out of range"}
        if self.board.cells[index] != EMPTY:
            return {"ok": False, "error": "cell occupied"}
        self.board.apply(index, player)
        if self.predictor:
            self.predictor.record(player, index)
        self._moves.append(
            {"player": player, "index": index, "coord": list(self.board.coord(index))}
        )

        self._evaluate()
        await self._emit(
            {
                "type": "move",
                "player": player,
                "index": index,
                "coord": list(self.board.coord(index)),
                "game": self.snapshot(),
            }
        )
        if self.over:
            await self._emit({"type": "gameover", "game": self.snapshot()})
        return {"ok": True, "game": self.snapshot()}

    def _evaluate(self) -> None:
        self.winner, self.winning_line, self.over = self.board.outcome()
        if self.over:
            self._log_game()
        if not self.over:
            self.current_player = P2 if self.current_player == P1 else P1

    def _log_game(self) -> None:
        """Append the finished game to the JSONL log for later training."""
        if self._logged:
            return
        self._logged = True
        from backend.server.game_log import record_game

        record_game(
            {
                "id": self.id,
                "ts": round(time.time(), 3),
                "size": self.board.n,
                "mode": self.mode,
                "difficulty": self.difficulty,
                "human_side": self._human_side,
                "winner": self.winner,
                "winning_line": self.winning_line,
                "moves": self._moves,
            }
        )

    def _switch_side_if_needed(self) -> None:
        pass

    # -- AI-vs-AI ----------------------------------------------------------
    async def _run_ai_vs_ai(self) -> None:
        delay = max(0.0, self.ai_delay)
        while not self.over:
            await asyncio.sleep(delay)
            player = self.current_player
            mover = self.x_mover if player == P1 else self.o_mover
            self.thinking = True
            await self._emit({"type": "thinking", "player": player, "game": self.snapshot()})
            loop = asyncio.get_event_loop()
            index = await loop.run_in_executor(None, mover, player)
            self.thinking = False
            await self._place(index, player)
        # final gameover already emitted by _place

class GameRegistry:
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}

    def create(self, **kwargs) -> GameSession:
        game_id = f"g{int(time.time() * 1000)}"
        session = GameSession(game_id, **kwargs)
        self.sessions[game_id] = session
        return session

    def get(self, game_id: str) -> Optional[GameSession]:
        return self.sessions.get(game_id)

    def remove(self, game_id: str) -> None:
        self.sessions.pop(game_id, None)