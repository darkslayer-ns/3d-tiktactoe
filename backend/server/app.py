"""FastAPI application: REST + WebSocket."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.server.session import AGENTS, DIFFICULTIES, GameRegistry
from backend.game.board import SUPPORTED_SIZES

app = FastAPI(title="3D Tic-Tac-Toe")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = GameRegistry()


class CreateGame(BaseModel):
    size: int = Field(3, ge=3, le=5)
    mode: str = Field("pve", pattern="^(pve|ave)$")
    difficulty: str = "medium"
    x_agent: str = "model"
    o_agent: str = "model"
    ai_delay: float = 0.4
    predictor_enabled: bool = False
    human_side: int = Field(1, ge=1, le=2)


class MoveRequest(BaseModel):
    index: int = Field(..., ge=0)


@app.get("/")
def root():
    return {"status": "ok", "service": "3d-tic-tac-toe"}


@app.post("/games")
async def create_game(req: CreateGame):
    if req.size not in SUPPORTED_SIZES:
        raise HTTPException(400, "size must be 3, 4 or 5")
    if req.x_agent not in AGENTS or req.o_agent not in AGENTS:
        raise HTTPException(400, "unknown agent")
    if req.difficulty not in DIFFICULTIES:
        raise HTTPException(400, "unknown difficulty")
    try:
        session = registry.create(
            size=req.size,
            mode=req.mode,
            difficulty=req.difficulty,
            x_agent=req.x_agent,
            o_agent=req.o_agent,
            ai_delay=req.ai_delay,
            predictor_enabled=req.predictor_enabled,
            human_side=req.human_side,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"game_id": session.id}


@app.get("/games")
async def list_games():
    return {"games": [s.admin_info() for s in registry.sessions.values()]}


@app.get("/games/{game_id}")
async def get_game(game_id: str):
    session = registry.get(game_id)
    if not session:
        raise HTTPException(404, "game not found")
    return session.snapshot()


@app.post("/games/{game_id}/move")
async def make_move(game_id: str, req: MoveRequest):
    session = registry.get(game_id)
    if not session:
        raise HTTPException(404, "game not found")
    result = await session.human_move(req.index)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "bad move"))
    return result


@app.post("/games/{game_id}/stop")
async def stop_game(game_id: str):
    session = registry.get(game_id)
    if not session:
        raise HTTPException(404, "game not found")
    session.over = True
    await session._emit({"type": "stopped", "game": session.snapshot()})
    return {"ok": True}


@app.delete("/games/{game_id}")
async def delete_game(game_id: str):
    session = registry.get(game_id)
    if not session:
        raise HTTPException(404, "game not found")
    session.over = True
    registry.remove(game_id)
    return {"ok": True}


@app.websocket("/games/{game_id}/stream")
async def stream(websocket: WebSocket, game_id: str):
    session = registry.get(game_id)
    if not session:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    queue = await session.subscribe()
    try:
        send_task = asyncio.create_task(_pump(queue, websocket))
        if not session.over and not session.thinking:
            # new observer: start game if it hasn't begun
            pass
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif data.get("type") == "start" and not session.over:
                await session.start()
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        session.unsubscribe(queue)


async def _pump(queue: asyncio.Queue, websocket: WebSocket):
    try:
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event))
    except asyncio.CancelledError:
        pass