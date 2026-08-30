"""Append-only game log (JSONL), for later training on real games.

Each finished game is written as one JSON line:
  {id, ts, size, mode, difficulty, human_side, winner, winning_line, moves:[{player,index,coord}]}
"""

from __future__ import annotations

import json
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATH = os.path.join(ROOT, "data", "games.jsonl")
_lock = threading.Lock()


def record_game(record: dict) -> None:
    with _lock:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def path() -> str:
    return _PATH