"""A small transformer for 3D tic-tac-toe with policy + value heads.

Treats each cell of the n×n×n cube as a token (like a word in a
sentence). Self-attention lets every cell condition on every other,
which is exactly what you need to see lines that run diagonally
through 3D space.

  policy head -> logits over cells (masked to legal moves)
  value head  -> win probability for the player to move

Board is normalized so the side-to-move is always "X" (value 1).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Coordinate-based position encoding, size-agnostic.

    Instead of a learned embedding over n^3 flat token ids (which ties the
    model to one cube size), we encode each cell's 3D coordinate (x,y,z)
    normalized to [0,1] through a small MLP. Coordinates always live in
    [0,n) regardless of n, so the SAME model works for any cube size.
    """

    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        # x: (B, N, d); build coordinate embedding for an n×n×n cube
        B, N, d = x.shape
        z = torch.arange(n, device=x.device).float() / (n - 1)
        y = torch.arange(n, device=x.device).float() / (n - 1)
        xc = torch.arange(n, device=x.device).float() / (n - 1)
        coords = torch.stack(
            [
                xc.repeat_interleave(n * n),
                y.repeat(n).repeat_interleave(n),
                z.repeat(n * n),
            ],
            dim=1,
        )  # (N, 3)
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        pos = self.mlp(coords)  # (B, N, d)
        return x + pos


class ValuePolicyTransformer(nn.Module):
    def __init__(self, n: int = 3, d_model: int = 128, nhead: int = 8, num_layers: int = 3):
        super().__init__()
        self.n = n
        self.cell_embed = nn.Embedding(3, d_model)  # 0 empty, 1 X, 2 O
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.policy_head = nn.Linear(d_model, 1)

    def cfg(self) -> dict:
        return {"n": self.n, "d_model": self.pos.mlp[-1].out_features, "num_layers": len(self.encoder.layers)}

    def forward(
        self,
        board: torch.Tensor,      # (B, N) int values 0/1/2
        legal_mask: torch.Tensor, # (B, N) bool, True = empty & playable
        n: int | None = None,     # override cube size (default self.n)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (value_logits (B,1), policy_logits (B,N)).

        `n` can differ from construction size, letting ONE trained model
        handle any cube size (coordinate-based position encoding).
        """
        size = n or (round(board.shape[-1] ** (1 / 3)) if board.dim() >= 1 and board.shape[-1] > 0 else self.n)
        x = self.cell_embed(board)          # (B, N, d)
        x = self.pos(x, size)
        x = self.encoder(x)                 # (B, N, d)

        value = self.value_head(x.mean(dim=1))   # pool over cells -> (B, 1)
        policy = self.policy_head(x).squeeze(-1)  # (B, N)

        # push illegal moves to -inf so softmax ignores them
        policy = policy.masked_fill(~legal_mask, float("-inf"))
        return value, policy

    def predict_move(self, board: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Argmax legal move (used for self-play inference)."""
        _, policy = self.forward(board, legal_mask)
        return policy.argmax(dim=1)


def policy_loss(policy_logits: torch.Tensor, move_target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over legal moves only (teacher move target)."""
    return F.cross_entropy(policy_logits, move_target)


def value_loss(value_logits: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy; target 1 = side-to-move wins, 0 = loses."""
    return F.binary_cross_entropy_with_logits(value_logits.squeeze(-1), value_target)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)