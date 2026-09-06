# ISOCUBE — build a transformer from scratch, and ship it in a game

<img src="mobile-rn/assets/isocube_logo.png" width="360" alt="ISOCUBE logo">

> A free, open-source, end-to-end project on **how to build a transformer**:
> design a small transformer, teach it to play 3D tic-tac-toe (n×n×n) with
> **supervised distillation followed by self-play RL**, port it to **C++**,
> and embed it into an **iOS / Android game that runs 100% on-device** — no
> server, no data collection, works offline.

The AI is a single **~106k-parameter transformer** (`d_model=64`, 8 heads,
2 layers). It treats every cell of the cube as one token (like a word in a
sentence) and is **size-agnostic**: one trained model plays any cube size
from 3×3×3 up to 6×6×6. The same hand-written C++ engine that powers the
Python backend is compiled into the app and called through JSI.

```
                    ┌────────────────────────────────────────────┐
                    │  TRAINING (offline, Python/PyTorch)         │
                    │  Phase 1: supervised distillation           │
                    │           (imitate the alpha-beta solver)   │
                    │  Phase 2: RL self-play (policy gradient)    │
                    │  Phase 3: difficulty calibration            │
                    │  → PyTorch checkpoint (.pt)                 │
                    └──────────────────────┬──────────────────────┘
                                           │ cpp/tools/export_weights.py
                                           ▼
                    ┌────────────────────────────────────────────┐
                    │  TFM1 binary  (cpp/model.bin)              │
                    │  + embed_weights.py → C array              │
                    │  tfm_model_data.h (in-app)                 │
                    └──────────────────────┬──────────────────────┘
                                           │ C++ port (byte-for-byte parity)
                                           ▼
   ┌────────────┐   board → tokens   ┌────────────────────────────┐
   │ BoardState │ ──────────────────► │  tfm::Model::forward(n)   │
   │  n×n×n     │  (normalize + mask) │  → value logit + policy    │
   └────────────┘                     └────────────┬───────────────┘
                                                   │ JSI / TurboModule
                                                   ▼
                    ┌────────────────────────────────────────────┐
                    │  LookaheadMover (TS) — shallow search +     │
                    │  difficulty knobs over the SAME weights     │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                                       chosen move
```

---

## The whole thing in plain English

**The idea in one sentence.** The game's AI is a tiny "brain" of about
106,000 numbers. Nobody programmed it with rules like *"if the opponent has
two in a row, block the third"* — instead it *learned* to play by watching a
perfect player and then playing millions of games against itself. This project
is a complete, from-scratch walkthrough of how to build that brain, and how to
shrink it so it fits inside a phone and runs offline.

**Think of each step like cooking from a recipe:**

| Step | Jargon | What it really is |
|------|--------|-------------------|
| 1. Turn the board into a language | *tokenizer* (§1) | Transformers were invented for text, but they work on any sequence. We treat the 27 cells of a 3×3×3 cube as 27 "words" and read them all at once. |
| 2. Give every box a meaning and a position | *embeddings* (§2) | Each cell gets a learned "meaning vector" (empty / mine / theirs) plus its 3D coordinate, so the brain knows both *what* is in each box and *where* it sits. |
| 3. Let every word read every other word | *self-attention* (§3) | Each cell "looks at" all the others at the same time. This is how the brain spots lines running diagonally through 3D space — the ones humans routinely miss. |
| 4. Ask two questions | *value + policy heads* (§4) | After thinking, the brain answers *"how likely am I to win?"* and *"which cell should I play next?"* |
| 5. Teach it | *supervised + RL training* | First it imitates a perfect solver (like studying grandmaster games), then it plays thousands of games against itself and learns from each win and loss. |
| 6. Ship it | *export → C++ → embed* (§6) | The 106k numbers are translated into a C++ file and embedded straight into the iOS/Android app, so it plays fully offline with no server. |

Each technical section below has a **Layman's take** box that re-explains the
same idea without the math.

---

## How the AI was built (the training journey)

### Phase 1 — Supervised pretraining: learn from a strong teacher

Before it can improve itself, the model first learns to **imitate the Go
alpha-beta solver** (`backend/distill`). The solver plays millions of games
and every position is labelled with its **best move** and **game value**:

```
┌────────────┐  positions  ┌──────────────┐  best move ┌─────────────┐
│ alpha-beta │ ───────────►│  transformer │ ◄───────── │ train policy│
│  solver    │  + values   │    (net)     │   + value  │ + value     │
└────────────┘             └──────────────┘            └─────────────┘
```

This is plain **supervised learning** — `L_policy` = cross-entropy against the
solver's best move, `L_value` = binary cross-entropy against the game outcome.
Train/eval are split by **whole games**, so eval positions never leak from
training games. It gives the network a strong policy/value baseline quickly.

### Phase 2 — RL: self-play policy gradient

Then the network improves by **playing itself**. At every position it samples
a move from its **own** softmax policy (a `temperature` controls exploration),
plays a full game, and the game's outcome becomes the value target for every
position it visited:

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

Because a better network generates better games next round, this is the
classic **policy-gradient / self-play** loop
(`training/train_universal.py` — one size-agnostic model trained on a mix of
3×3×3, 4×4×4 and 6×6×6 games, so a 6×6-trained model transfers to 3×3).

### Phase 3 — Difficulty calibration

`retrain_selfplay.py` calibrates each difficulty's runtime knobs (mistake
rate, temperature, lookahead depth) against the **real on-device mover**, so a
casual human wins roughly the intended share per level.

The training code lives in `dev/training/` and `dev/scripts/` (see
[Repo map](#repo-map)).

> **Layman's take.** Learning to play is like learning a musical instrument in
> two stages. First you *copy a master* — the solver shows you the perfect move
> for millions of positions, and the brain copies it until it's competent
> (Phase 1, supervised). Then you *practice alone*: the brain plays thousands of
> games against itself; whenever it wins, it slightly reinforces the moves that
> led to the win, and whenever it loses, it weakens them (Phase 2, RL). Finally
> we tune how often the brain "accidentally" plays a bad move so the game has an
> Easy / Medium / Hard setting (Phase 3, calibration).

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

> `mobile-rn/src/ai/engine.ts` and `dev/training/selfplay.py:_state` implement
> the same normalization; the C++ side re-checks it (`TfmEngine.cpp`).

> **Layman's take — the "tokenizer".** Transformers were invented for text, but
> they don't actually care about words — they care about *sequences*. Here we
> simply call each cell of the cube a "word": a 3×3×3 board is a 27-word
> "sentence", a 5×5×5 board is a 125-word sentence. The brain reads the whole
> sentence at once. We also do a small trick so the brain only ever has to learn
> *one* question — "am I winning?" — by always re-labelling the board so the
> player whose turn it is is called "X". The `mask` is just the list of
> legal (empty) cells, like telling the brain which words it's allowed to change.

---

## 2. Embedding

Two embeddings are added together to form the input to the transformer.

### 2a. Cell-type embedding

A learned `nn.Embedding(3, d_model)` table maps the token `{0,1,2}` to a
64-dim vector — the model learns a distinct vector for "empty", "mine", "their".

> **Layman's take.** A "64-dim vector" is just a list of 64 numbers that stands
> in for a word, like a dictionary entry. The brain starts with three random
> entries (empty / mine / theirs) and, during training, slowly rewrites them so
> that "mine" and "their" end up *feeling* different to the network. It's the
> same idea as a word-embedding in a chatbot: "king" and "queen" live near each
> other in the map, and here "empty" and "occupied" live far apart.

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

> **Layman's take — position.** A word's *meaning* alone isn't enough; you also
> need to know where it sits in the sentence. ("the cat bit the dog" ≠ "the dog
> bit the cat".) So each cell also gets a small vector built from its 3D
> coordinate `(x, y, z)`. Because coordinates are squeezed into the range
> 0…1 regardless of the cube size, the *same* trained brain understands a 3×3×3
> board, a 4×4×4 board, and a 6×6×6 board — one model, every size.

---

## 3. Transformer encoder

Two stacked encoder blocks process the token sequence. Each cell attends to
every other cell, which is exactly what lets the model see **lines that run
diagonally through 3D space** (self-attention has no locality bias).

### How self-attention works (the Q/K/V story)

> **Layman's take.** Imagine each cell is a guest at a round-table meeting, and
> every guest wants to hear what every other guest has to say before making up
> their mind. That's exactly what "self-attention" does. Mechanically each cell
> writes three notes about itself:
>
> - a **Query** (Q) — "this is what I'm asking about",
> - a **Key** (K) — "this is what I have to offer",
> - a **Value** (V) — "this is my actual message".
>
> Every cell then looks at every other cell and scores "does your Key answer my
> Query?" — cells that match get more attention, and the final output for each
> cell is a weighted mix of everyone's Values. This is why the network sees
> winning lines that cut diagonally through the cube: the two cells at opposite
> corners of a diagonal pair up their Keys and Queries even though they're far
> apart in the flat array. The "8 heads" just means the meeting happens 8 times
> in parallel, each paying attention to a different kind of relationship
> (e.g. one head might track straight rows, another the space diagonals).

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

> **Why residual connections + LayerNorm?** The "+(residual)" is a shortcut that
> lets each cell keep its original information and only *adjust* it, rather than
> fully rewriting it — this makes deep networks far easier to train (a famous
> idea that made very deep transformers possible). LayerNorm just rescales the
> numbers so they stay in a healthy range as they flow through the network —
> think of it as "keep your volume reasonable" at each step.

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

> **Layman's take.** After the attention meeting, every cell has formed an
> opinion. The value head simply *averages all the opinions* into one number and
> converts it into a "chance of winning" between 0% and 100%. In a position
> where you're about to complete a line, it should say ~99%; in a hopeless
> position, ~1%.

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

> **Layman's take — policy.** The policy head is the "where do I move?" answer:
> each cell gets a raw score, occupied cells are disqualified (masked to −∞),
> and the scores are turned into a ranked list of probabilities. Playing the
> highest-probability cell is the "best" move; sampling from the list with some
> temperature is how the brain explores during training and how Easy/Medium
> levels add variety.

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

> **Layman's take — the whole forward pass.** Feeding a board through the network
> is like running a single question through the brain: the board becomes a
> "sentence" of cells → each cell gets a meaning + position → every cell reads
> every other cell (attention) → the value head says "you're 63% likely to win"
> and the policy head hands back a ranked list of legal moves. One forward pass,
> a handful of milliseconds, two answers.

---

## 6. Export → on-device C++

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

> **Layman's take — "translating the recipe".** The brain was trained in Python
> (PyTorch) as a bunch of layer definitions and 106,690 numbers. To run on a
> phone, we *translate* it by hand into C++ — the exact same operations, re-written
> in a language the phone can run natively. The weights are then baked into the
> app binary as a giant C array, so nothing is downloaded at runtime and there is
> no server anywhere. The "parity test" is our guarantee that the phone's C++
> brain makes byte-for-byte the same decisions the Python brain would — otherwise
> Easy/Medium/Hard behaviour would silently differ between desktop and phone.

---

## 7. Search + difficulty (runtime behavior)

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
the SAME weights (`mobile-rn/src/ai/mover.ts:DIFFICULTY`):

|            | Easy (≈65% AI win) | Medium (≈80%) | Hard (≈95%) |
|------------|---------------------|---------------|-------------|
| search depth | 1 | 3 | 4 |
| deliberate blunders | up to 6 **random** moves (~25% of neutral moves; ~90% when it's already clearly winning) | exactly **1 suboptimal** move, only when about to win | none |
| move randomness | high (temp 1.1) | medium (temp 0.5) | near-greedy (temp 0.1) |
| extra | defensive bias; adapts to the player's results | — | — |

### It learns your habits (opponent memory)

On top of the fixed transformer weights, the app keeps a small, persistent
**opponent memory** — a per-side map of *which cells you like to play*,
maintained entirely on-device (pure bookkeeping, no weight updates):

```
 you play cell i   ──►  affinity[you][i] += 1      (recorded live, each move)
 you WIN a game    ──►  your played cells × 1.25   (the AI learns what beat it)
 you LOSE a game   ──►  your played cells × 0.5    (those moves are punished)
 each new game     ──►  all weights × 0.9          (recency fade)
```

- **Recorded live** — every move you make increments that cell's affinity for
  your side (`OpponentPredictor.record`).
- **Rewarded / punished after every game** — win → your played cells are boosted
  1.25× (`WIN_BOOST`); lose → they are decayed 0.5× (`LOSS_DECAY`); a draw
  leaves them unchanged (`opponentMemory.applyResult`).
- **Decayed per game** (× 0.9) so recent sessions count more than old ones.
- **Persisted to on-device storage**, so the memory survives app restarts
  (`opponentStorage.ts`).

The search then uses this memory two ways while playing against you
(`LookaheadMover._strongMove`, `src/ai/mover.ts`):

1. **Denying your favourite cells.** Every move you've overplayed gets a
   "deny" bonus (`DENY_WEIGHT × affinity[you][cell]`) added to the AI's own
   move scores, so the AI prefers to take those cells itself instead of leaving
   them open — the more you play a square, the more the AI snatches it.
2. **Predicting your replies.** During its lookahead the AI models where you're
   likely to move next using the network's policy head
   (`OpponentPredictor.likelyMoves`), re-weighted by your **style profile**
   (attacker vs. defender — how often you build threats vs. block), so it
   spends its search budget on the replies you're most likely to make.

The affinity-blended move prediction (`predictDistribution`) is also exposed
for the internal model-knowledge panel, but the live engine leans on the
deny-bias + policy-head prediction. Files: `src/ai/predictor.ts` (in-game
prediction), `src/ai/opponentMemory.ts` (cross-game learning),
`src/ai/opponentStorage.ts` (persistence).

Difficulty also adapts to your recent results (win too much → it hardens; lose
too much → it eases up).

> **Layman's take — thinking ahead.** The brain alone is a one-shot "move
> guesser". To be stronger, the game combines it with a mini search: "if I play
> here, what's my expected chance of winning, assuming I keep using the brain for
> the rest of the game?" — then it picks the move with the best score. Hard
> *thinks 4 moves ahead* and never blunders; Medium thinks 3 ahead and makes one
> subtle slip when it's about to win; Easy barely thinks (1 move), blunders up to
> 6 times with fully random moves, and mostly *defends* instead of attacking. So
> "difficulty" isn't a different brain — it's just how deep it thinks and how
> often it deliberately plays badly.

---

## Build & run

Quick orientation — the shipping app is `mobile-rn/`, the shared engine is
`cpp/`, all offline ML/dev tooling is `dev/`. Full per-component docs:

- [mobile-rn/README.md](mobile-rn/README.md) — the iOS/Android app: build, run,
  APK, tests
- [mobile-rn/native/README.md](mobile-rn/native/README.md) — the C++ JSI
  engine + iOS/Android build wiring
- `dev/scripts/`, `dev/training/` — training, export and calibration scripts
- `dev/frontend/` — the web frontend (Vite/React)

### 0. One-time setup

**Host requirements**

- **Python 3 + PyTorch** (training only)
- **C++17** compiler + CMake (engine + parity)
- **iOS**: macOS + Xcode + CocoaPods
- **Android**: Android SDK + **NDK 27.1.12297006** + JDK 17

**Local build secrets (gitignored).** Signing values are never committed. Create
`mobile-rn/scripts/build-secrets.env` locally (see `mobile-rn/.gitignore`) and
fill in your Apple Developer Team ID and Android keystore passwords:

```bash
# mobile-rn/scripts/build-secrets.env   (gitignored — do not commit)
IOS_TEAM=XXXXXXXXXX             # Apple Developer team ID
KEY_STORE_PASS=your-store-pass   # Android release.keystore store password
KEY_PASS=your-key-pass           # Android release.keystore key password
```

The Makefile, `build_apk.sh` and the iOS export-plist generator all source this
file; an environment `export` of the same names overrides it. Without it, the
iOS/Android release targets refuse to run.

**Generate native projects (once).** `expo prebuild` recreates `mobile-rn/android/`
and `mobile-rn/ios/` from `app.json` (the `withTfmEngine` plugin also writes the
Android CMake/OnLoad wiring and the iOS registration file):

```bash
cd mobile-rn
npm install
npx expo prebuild            # runs the withTfmEngine plugin
cd ios && pod install        # iOS CocoaPods
```

> ⚠️ `expo prebuild` **clears `mobile-rn/android/`**, which also removes the
> local `release.keystore`. Keep a backup of your keystore outside the repo and
> restore it into `mobile-rn/android/` after prebuild. The iOS engine
> registration file (`native/TfmEngineRegistration.mm`) is regenerated by the
> plugin.

### 1. The engine (C++ → parity)

```bash
make cpp          # builds parity_test + tfm-cli + libmodel.so into cpp/build
make parity       # full C++ ⇄ PyTorch parity check (fixtures + random cross-check)
```

### 2. Release builds (embedded bundle — no Metro)

The Makefile provides **public release** and **internal** targets. "Internal"
builds bake `EXPO_PUBLIC_INTERNAL_DEBUG=1` into the JS bundle so the AI
debug / model-knowledge panel is included; public release builds compile it out.
Both are Release configurations — the JS bundle is embedded, so no Metro / dev
server is needed at runtime.

```bash
# iOS — build for a connected device (UDID is auto-detected)
make ios-release                # public release
make ios-internal               # internal (+ AI debug panel)

# iOS — App Store Connect IPA (archive + export)
make ios-release-ipa            # → /tmp/ISOCUBE-release-export/ISOCUBE.ipa
make ios-internal-ipa           # → /tmp/ISOCUBE-internal-export/ISOCUBE.ipa

# Android — signed release APK
make android-release            # → mobile-rn/dist-apk/neoncube-phone-release.apk
make android-internal           # internal (+ AI debug panel)

# Both platforms
make release                    # ios-release + android-release
make internal                   # ios-internal + android-internal
```

Notes:
- iOS device builds require a connected iPhone (or set `IOS_UDID=<udid>`).
- iOS signing uses `IOS_TEAM` from `build-secrets.env`, passed as
  `DEVELOPMENT_TEAM`; the App Store export plist is generated at build time.
- Android signing uses `mobile-rn/android/release.keystore` with
  `KEY_STORE_PASS` / `KEY_PASS` from `build-secrets.env`.

### 3. Development (Metro / dev server)

```bash
cd mobile-rn
npx expo run:ios               # builds + starts Metro (macOS + Xcode required)
npx expo run:android           # builds + installs a debug build on a device (adb)
```

### 4. Tests

```bash
make test                   # cpp parity + backend pytest + mobile jest
```

---

## Repo map

```
mobile-rn/                 the ISOCUBE app (Expo/React Native 0.86)
cpp/                       shared C++ transformer engine (compiled into the app)
dev/                       everything else: training, backend, web, tooling
dev/
  backend/                 game rules, ML agent wiring, server, alpha-beta distill
  training/                PyTorch training: model.py, selfplay, trainers
  scripts/                 training/export/calibration scripts
  frontend/                web frontend (Vite/React)
  model_universal.pt       trained checkpoint
```

ML parts in detail:

```
dev/training/
  model.py            ValuePolicyTransformer (tokenizer, embeds, encoder, heads)
  selfplay.py         self-play game generation
  train_universal.py  size-agnostic self-play (RL) trainer
  train_distill.py    alpha-beta distillation (supervised) trainer
  eval.py             strength evaluation vs random
  torch_loader.py     checkpoint loader + TorchModelAdapter (eval seam)
dev/backend/
  distill/            Go alpha-beta solver (distillation teacher)
  ml/                 game-server AI wiring (torch-free)
cpp/
  include/tfm/*.hpp   C++ model: model.hpp, layers.hpp, ops.hpp, weights.hpp
  src/                model.cpp, layers.cpp, ops.cpp, weights.cpp, search.cpp
  tools/export_weights.py   PyTorch → TFM1 binary
  tools/check_parity.py     C++ vs PyTorch parity
  tests/parity.cpp          on-host parity test
mobile-rn/
  native/cpp/TfmEngine.cpp  JSI/TurboModule host functions
  native/include/tfm_model_data.h   embedded weights (C array)
  native/TfmEngineRegistration.mm   iOS global registrar (add to target)
  scripts/embed_weights.py          bin → C header
  src/ai/engine.ts        normalization + eval seam
  src/ai/mover.ts         lookahead search + difficulties
  src/ai/predictor.ts     opponent modeling (persistent affinity)
  src/three/              Blender-baked 3D geometry (assets → TS)
```

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

This is a free, from-scratch project written to show how a transformer works
and how to take it all the way to a shipping product: tokenization, embedding,
attention, training (supervised → self-play RL), a byte-for-byte C++ port, and
on-device inference in a real mobile game.