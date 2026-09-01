# ISOCUBE — 3D Tic-Tac-Toe with an on-device neural network

A full-stack 3D tic-tac-toe (n×n×n) with a small transformer that plays on
device — no server, no data collection, works offline. The AI is a single
**~106k-parameter transformer** (`d_model=64`, 8 heads, 2 layers) trained by
self-play and solver distillation, ported to C++ and embedded straight into the
mobile binary.

```
                    ┌──────────────────────────────┐
                    │  TRAINING (offline, Python)   │
                    │  self-play  +  solver distill │
                    │  → PyTorch checkpoint (.pt)   │
                    └──────────────┬───────────────┘
                                   │ export_weights.py
                                   ▼
                    ┌──────────────────────────────┐
                    │  TFM1 binary  (cpp/model.bin) │
                    │  + embed_weights.py → C array │
                    │  tfm_model_data.h (in-app)    │
                    └──────────────┬───────────────┘
                                   │ C++ port (byte-for-byte parity)
                                   ▼
   ┌────────────┐   board → tokens   ┌──────────────────────────────┐
   │ BoardState │ ──────────────────► │  tfm::Model::forward(n)      │
   │  n×n×n     │  (normalize + mask) │  → value logit + policy logits│
   └────────────┘                     └──────────────┬───────────────┘
                                                     │ JSI / TurboModule
                                                     ▼
                                   ┌──────────────────────────────┐
                                   │  LookaheadMover (TS)         │
                                   │  shallow search + difficulty │
                                   └──────────────┬───────────────┘
                                                  ▼
                                            chosen move
```

---

## 1. Input representation ("tokenizer")

There is no text tokenizer — **every cell of the cube is one token**, like a
word in a sentence. A board with 27 cells becomes a sequence of 27 tokens.

The board is **normalized so the side-to-move is always "X" (token 1)**, so the
model only ever needs to answer one question: "how good is this position for
the player whose turn it is?".

| cell state         | token | mask (legal?) |
|--------------------|-------|---------------|
| empty              | `0`   | `1`           |
| player to move (X) | `1`   | `0`           |
| opponent (O)       | `2`   | `0`           |

```
cells  = [ 0, 1, 0, 2, 0, ... ]   n=3 → 27 tokens
player = 2                        side-to-move is O
norm   = [ 0, 2, 0, 1, 0, ... ]   every non-empty O (2) becomes X (1),
                                  every non-empty X (1) becomes O (2)
mask   = [ 1, 0, 1, 0, 1, ... ]   1 where empty (legal moves)
```

> `mobile-rn/src/ai/engine.ts` and `training/selfplay.py:_state` implement the
> same normalization; the C++ side re-checks it (`TfmEngine.cpp`).

---

## 2. Embedding

Two embeddings are added together to form the input to the transformer.

### 2a. Cell-type embedding

A learned `nn.Embedding(3, d_model)` table maps the token `{0,1,2}` to a
64-dim vector — the model learns a distinct vector for "empty", "mine", "their".

### 2b. Coordinate position encoding (size-agnostic)

Instead of learned position ids over `n³` flat indices (which would tie the
model to one cube size), each cell's 3D coordinate `(x, y, z)` is normalized
to `[0, 1]` and pushed through a small MLP:

```
coord_i = ( x/(n-1), y/(n-1), z/(n-1) )    3 inputs
          ┌────────────────────────────┐
pos_i  =  │ Linear(3→32) → ReLU        │
          │ Linear(32→64)              │   d_model outputs
          └────────────────────────────┘

input_i = cell_embed(token_i) + pos_i
```

Because coordinates always live in `[0,1]` regardless of `n`, **one trained
model works for any cube size** (3×3×3, 4×4×4, … 6×6×6). This is what makes
the "universal" model possible.

```
 tokens ──► Embedding(3→64) ──┐
                              ├── ( + ) ──► x  (N × 64)
 coords ──► CoordMLP(3→64) ──┘
```

---

## 3. Transformer encoder

Two stacked encoder blocks process the token sequence. Each cell attends to
every other cell, which is exactly what lets the model see **lines that run
diagonally through 3D space** (self-attention has no locality bias).

```
                    x (N × 64)
                        │
              ┌─────────▼─────────┐
              │  Multi-Head Self- │  8 heads, 64-dim, d_FF = 256
              │  Attention (8h)   │
              └─────────┬─────────┘
                        │  + (residual)
              ┌─────────▼─────────┐
              │  LayerNorm        │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │  Feed-Forward     │  Linear(64→256) → GELU
              │                   │  Linear(256→64)
              └─────────┬─────────┘
                        │  + (residual)
              ┌─────────▼─────────┐
              │  LayerNorm        │
              └─────────┬─────────┘
                        │
                 ┌──────┴──────┐
                 ▼             ▼
             value head    policy head
```

Each block implements the standard `norm_first=False` (post-norm) residual
pattern:

```
attn = MultiHead(x)
x    = LayerNorm(x + attn)
ff   = GELU(Linear(x)) → Linear
x    = LayerNorm(x + ff)
```

---

## 4. Heads

The encoder outputs one 64-dim vector **per cell**. Two small heads turn them
into the model's answers.

### Value head — "how good is this position?"

Mean-pool the per-cell vectors into a single vector, then an MLP:

```
pooled = mean over cells of x        (1 × 64)
value  = Linear(64→64) → ReLU → Linear(64→1)     → value logit v
P(win for side-to-move) = sigmoid(v)
```

### Policy head — "where should I play?"

A single `Linear(64→1)` per cell produces a raw logit per cell:

```
policy_i = Linear(x_i)   for every cell i     (N logits)
```

Illegal moves (occupied cells) are set to **−∞** so the softmax ignores them:

```
probs_i = softmax(policy)_i        over legal cells only
best move = argmax over legal cells
```

```
                    mean ──► MLP ──► value logit ──► sigmoid ──► win prob
                ┌───┘
 x (N×64) ──────┼───► Linear(64→1) per cell ──► logits
                └──────────────► mask(−∞ on occupied) ──► softmax ──► policy
```

---

## 5. One forward pass — end to end

```
board (n³ cells)                     mask (n³)
     │                                  │
     ▼                                  ▼
 [0,2,0,1,…]                     [1,0,1,0,…]
     │                                  │
     ▼                                  │
 cell_embed(token) ──┐                  │
 coord_mlp(xyz) ─────┼─► x (n³×64)     │
     ▼                │                 │
 2 × TransformerEncoder                  │
     ▼                 │                 │
 mean ──► value logit  │                 │
 per-cell ──► logits   │                 │
     │                  ▼                 ▼
 sigmoid ──► P(win)    mask logits(−∞) → softmax → policy over legal moves
     │                  │
     ▼                  ▼
 { value: 0.63 }   { policy: [0.001, 0.02, …] }   ← returned to the mover
```

The C++ implementation mirrors PyTorch **exactly** (eval mode, dropout off)
and is checked byte-for-byte against the reference graph by a parity test
(`cpp/tools/check_parity.py`, `mobile-rn` parity jest test).

---

## 6. Training

The model is trained offline (Python/PyTorch) with **two complementary
sources of supervision**:

### 6a. Self-play (policy-gradient)

```
┌──────────┐  sample move ┌──────────┐  play game ┌──────────┐
│ network  │ ────────────► │  game    │ ─────────► │ outcome  │
│ (policy) │   temperature │  self    │            │ win/loss │
└──────────┘   exploration └──────────┘            └────┬─────┘
                                                       │ value target
                                                       ▼
                    store every position with its game outcome
                    → value head learns "was this position winning?"
                    → policy head learns "what did the winner play?"
```

At every position the network samples a move from its **own** softmax policy
(`temperature` controls exploration); the game's outcome becomes the value
target for every stored position. The improved network generates better games
next round (`training/train_universal.py` — one size-agnostic model trained on
a mix of 3×3×3, 4×4×4, 6×6×6 games).

### 6b. Solver distillation

Alternative/auxiliary: the transformer learns to **imitate the Go alpha-beta
solver** (`backend/distill`). Each sample is a board position labeled with the
solver's best move and game value:

```
┌────────────┐  positions  ┌──────────────┐  best move ┌─────────────┐
│ alpha-beta │ ───────────►│  transformer │ ◄───────── │ train policy│
│  solver    │  + values   │    (net)     │   + value  │ + value     │
└────────────┘             └──────────────┘            └─────────────┘
```

Train/eval are split by **whole games** so eval positions never come from
training games (no leakage).

### Losses

```
L_policy = cross_entropy(policy_logits, best_move)      # over legal moves only
L_value  = binary_cross_entropy_with_logits(value, win?) # 1 = side-to-move wins
```

---

## 7. Export → on-device C++

```
PyTorch model (.pt)
      │
      ▼  cpp/tools/export_weights.py
TFM1 binary (self-describing)  ──►  cpp/model.bin
      │
      ▼  mobile-rn/scripts/embed_weights.py
tfm_model_data.h  (const unsigned char kModelBin[])  ── compiled into the app
      │
      ▼  mobile-rn/native/cpp/TfmEngine.cpp  (tfm::Model, layers, ops)
JS ⇄ C++ via JSI host functions:  load() / evalPosition(board, mask, n) / numel()
```

The weights ship **inside the app binary** — no filesystem I/O, no network,
no runtime framework. The C++ engine is registered as a TurboModule
(`TfmEngine`) and exposed to JS, with a byte-for-byte parity guarantee against
the PyTorch graph.

`numel()` = **106,690** parameters.

---

## 8. Search + difficulty (runtime behavior)

The raw network is combined with a shallow lookahead to decide moves:

```
evalPosition(board, side)
        │
        ▼
LookaheadMover.chooseMove(side)
  • immediate win / block checks
  • score every legal move by depth-limited expected-value search
    (uses OpponentPredictor — the net's own policy head — to model the
    opponent's replies)
  • final pick from the top-scored moves, temperature-tempered
```

**Difficulty is not a different model** — it's runtime search parameters over
the SAME weights (`src/ai/mover.ts:DIFFICULTY`):

|            | Easy (≈65% AI win) | Medium (≈80%) | Hard (≈95%) |
|------------|---------------------|---------------|-------------|
| search depth | 1                  | 2             | 4           |
| deliberate blunders | up to 6 @ 45% | up to 2 @ 30% | none       |
| move randomness | high (temp 1.1) | med (0.6)     | near-greedy (0.1) |

The `OpponentPredictor` additionally keeps a persistent, decaying affinity map
of the moves *you* play (boosted when you win) so the AI gradually learns your
tendencies across sessions — heuristic, on-device, no weight updates.

---

## Repo map (ML parts)

```
training/
  model.py            ValuePolicyTransformer (tokenizer, embeds, encoder, heads)
  selfplay.py         self-play game generation
  train_universal.py  size-agnostic self-play trainer
  train_distill.py    alpha-beta distillation trainer
  eval.py             strength evaluation vs random
  torch_loader.py     checkpoint loader + TorchModelAdapter (eval seam)
backend/
  distill/            Go alpha-beta solver (distillation teacher)
  ml/                 game-server AI wiring (torch-free)
cpp/
  include/tfm/*.hpp   C++ model: model.hpp, layers.hpp, ops.hpp, weights.hpp
  tools/export_weights.py   PyTorch → TFM1 binary
  tools/check_parity.py     C++ vs PyTorch parity
  tests/parity.cpp          on-host parity test
mobile-rn/
  native/cpp/TfmEngine.cpp  JSI/TurboModule host functions
  native/include/tfm_model_data.h   embedded weights (C array)
  scripts/embed_weights.py         bin → C header
  src/ai/engine.ts        normalization + eval seam
  src/ai/mover.ts         lookahead search + difficulties
  src/ai/predictor.ts     opponent modeling (persistent affinity)
```