/**
 * Model-based mover: the game AI. Byte-for-byte port of
 * backend/ml/model_agent.py (LookaheadMover + ModelMover).
 *
 * Plays using the trained transformer's policy head, plus model-guided
 * lookahead (expectimax over the predicted opponent). Difficulty maps to
 * lookahead depth, a wrong-move budget, and sampling temperatures.
 *
 * Python's `random.random()/random.choice` are replaced by a deterministic,
 * injectable RNG (default Math.random) so tests can seed it. The engine is the
 * EvalEngine seam (src/ai/engine.ts) whose evalPosition returns a VALUE LOGIT
 * and POLICY LOGITS — the mover sigmoids the value and softmax/argmaxes the
 * policy exactly like the Python.
 */

import { Board } from '../game/board'
import { EMPTY, P1, P2 } from '../game/types'
import type { Cell } from '../game/types'
import type { AiDecision, Difficulty, EvalEngine, LineStep, ScoredMove } from './types'
import { argmax, sampleIndex, sigmoid, softmax, pyRound } from './math'
import type { Rng } from './math'
import { defaultRng } from './math'
import type { OpponentPredictor } from './predictor'
import { hasWinInOne, wouldWin } from './profile'

export const DIFFICULTY_TEMPERATURE: Record<Difficulty, number> = {
  easy: 1.5,
  medium: 0.7,
  hard: 0.0,
}

/** Lookahead depth the Hint uses, independent of difficulty (strong hints). */
export const HINT_DEPTH = 5

/**
 * Denial weight: the AI prefers to occupy cells the player has overplayed
 * (from the persisted affinity heatmap), DENYING their favourite cells instead
 * of leaving them open. Map lookups are O(1) — faster than a BST.
 */
export const DENY_WEIGHT = 0.05

/** Defensive-bias strength per unit of `defensive` (EASY blocks > attacks). */
export const DEFENSIVE_WEIGHT = 0.15

/**
 * EASY "get-out-of-jail" card: when the AI's search says it's clearly winning
 * (predicted win-prob above this), it makes a mistake instead — giving the
 * player a comeback even though the position was lost.
 */
export const EASY_PREDICAMENT_THRESHOLD = 0.85
export const EASY_PREDICAMENT_RATE = 0.9

export interface DifficultyConfig {
  depth: number
  top_k: number
  sampling: boolean
  max_nodes: number
  entry_temp: number
  entry_moves: number
  wrong_move_budget: number
  mistake_rate: number
  move_temp: number
  /**
   * How the AI makes its deliberate mistakes:
   *   'random'      → plays a random empty cell (visibly dumb — EASY)
   *   'suboptimal'  → plays a good-but-not-best move (subtle — MEDIUM)
   *   'none'        → never blunders (HARD)
   */
  blunder_kind: 'random' | 'suboptimal' | 'none'
  /** 0 = balanced; >0 biases the AI to BLOCK the player rather than attack
   * (0.5-1 makes the AI visibly defensive — used to make EASY easy). */
  defensive?: number
}

/** Pure RUNTIME config — all difficulties share the same strong base weights. */
export const DIFFICULTY: Record<Difficulty, DifficultyConfig> = {
  // Approx AI win rates: easy ~65%, medium ~80%, hard ~95%.
  easy: {
    depth: 1,
    top_k: 3,
    sampling: false,
    max_nodes: 220,
    entry_temp: 1.0,
    entry_moves: 3,
    wrong_move_budget: 6,
    mistake_rate: 0.25,
    move_temp: 1.1,
    blunder_kind: 'random',
    defensive: 1,
  },
  medium: {
    depth: 3,
    top_k: 3,
    sampling: false,
    max_nodes: 220,
    entry_temp: 0.6,
    entry_moves: 1,
    wrong_move_budget: 1,
    mistake_rate: 0.25,
    move_temp: 0.5,
    blunder_kind: 'suboptimal',
  },
  hard: {
    depth: 4,
    top_k: 3,
    sampling: false,
    max_nodes: 220,
    entry_temp: 0.3,
    entry_moves: 1,
    wrong_move_budget: 0,
    mistake_rate: 0.0,
    move_temp: 0.1,
    blunder_kind: 'none',
  },
}

/** The engine must implement evalPosition(); returns (engine, error). */
export function requireEngine(
  model: EvalEngine | null,
): { engine: EvalEngine | null; error: string | null } {
  if (model === null) {
    return { engine: null, error: 'no model' }
  }
  if (typeof (model as { evalPosition?: unknown }).evalPosition !== 'function') {
    return {
      engine: null,
      error:
        'model must implement evalPosition() — wrap torch graphs with training.torch_loader.TorchModelAdapter',
    }
  }
  return { engine: model, error: null }
}

export class ModelMover {
  readonly board: Board
  readonly model: EvalEngine | null
  readonly error: string | null
  readonly temperature: number
  private rng: () => number

  constructor(
    board: Board,
    path: string | null = null,
    model: EvalEngine | null = null,
    temperature = 0.0,
    difficulty: Difficulty = 'hard',
    rng: () => number = Math.random,
  ) {
    this.board = board
    this.temperature = DIFFICULTY_TEMPERATURE[difficulty] ?? temperature
    if (model !== null) {
      const { engine, error } = requireEngine(model)
      this.model = engine
      this.error = error
    } else {
      // Python loads the C++ engine from a path; on-device the caller passes a
      // native engine, so a bare path is unsupported here.
      this.model = null
      this.error = path !== null ? `no weights at ${path}` : 'no model'
    }
    this.rng = rng
  }

  call(player: Cell): number {
    if (this.model === null) {
      throw new Error(this.error ?? 'no model')
    }
    const { policy } = this.model.evalPosition(this.board.cells, player)
    if (this.temperature <= 0) {
      return argmax(policy)
    }
    const probs = softmax(policy, this.temperature)
    return sampleIndex(probs, this.rng)
  }

  moveDistribution(player: Cell): number[] {
    if (this.model === null) {
      throw new Error(this.error ?? 'no model')
    }
    const { policy } = this.model.evalPosition(this.board.cells, player)
    return softmax(policy, 1.0)
  }
}

export class LookaheadMover {
  readonly model: EvalEngine | null
  readonly error: string | null
  readonly board: Board
  readonly predictor: OpponentPredictor
  readonly depth: number
  readonly topK: number
  readonly sampling: boolean
  readonly maxNodes: number
  readonly entryTemp: number
  readonly entryMoves: number
  mistakeRate: number
  readonly shiftRate: number
  wrongMoveBudget: number
  moveTemp: number
  readonly blunderKind: 'random' | 'suboptimal' | 'none'
  readonly defensive: number
  lastDecision: AiDecision | null = null
  /** Player style: +1 attacker … -1 defender. Biases opponent-reply prediction. */
  aggression = 0
  /** Adaptive strength from recent results: +1 ease … -1 harden. */
  adaptiveLevel = 0
  /** Whether the last _strongMove was forced (immediate win/block). */
  private _lastForced: 'win' | 'block' | null = null
  private rng: Rng
  private _nodes = 0
  private _wrongMovesUsed = 0
  private _lastScored: Array<[number, number]> | null = null
  private _yieldT = 0
  private _baseMistakeRate = 0
  private _baseMoveTemp = 0
  private _baseWrongMoveBudget = 0

  /**
   * Cooperatively yield control back to the JS thread every ~16ms so the AI
   * search never freezes animations or gesture handling. No-op when the last
   * yield was recent; the search result is unaffected.
   */
  private async _maybeYield(): Promise<void> {
    const now = Date.now()
    if (now - this._yieldT > 16) {
      this._yieldT = now
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
    }
  }

  /**
   * Node budget scaled by cube size so a move's compute stays roughly constant
   * as the board grows (a 6×6×6 forward is ~60× the FLOPs of 3×3×3). n=3 is
   * unchanged (factor 1), so parity and existing decisions are unaffected.
   */
  private _effMaxNodes(): number {
    const n = this.board.n
    const k = 27 / (n * n * n)
    return Math.max(24, Math.round(this.maxNodes * k))
  }

  /** Set the human's style profile: +1 attacker … -1 defender. */
  setAggression(aggression: number): void {
    this.aggression = Math.max(-1, Math.min(1, aggression))
  }

  /**
   * Adapt the AI's strength around the difficulty's base params.
   * level > 0 eases up (more blunders/randomness — for a struggling player),
   * level < 0 hardens (fewer blunders — for a dominant player).
   */
  setAdaptive(level: number): void {
    this.adaptiveLevel = Math.max(-1, Math.min(1, level))
    this.moveTemp = Math.max(0, Math.min(1.2, this._baseMoveTemp + level * 0.4))
    this.wrongMoveBudget = Math.max(0, Math.round(this._baseWrongMoveBudget + level * 2))
  }

  /**
   * Reinforcement-weighted mistake rate (the "fuzzy" lever): starts from the
   * difficulty's base, then eases up when the player is losing more than the
   * difficulty target (adaptiveLevel from stored win/loss stats) and adds a
   * small style fuzz (+ for attackers, − for defenders). Clamped to [0, 0.8].
   */
  private _effectiveMistakeRate(): number {
    return Math.max(0, Math.min(0.8, this._baseMistakeRate + this.adaptiveLevel * 0.3 + this.aggression * 0.1))
  }

  /** Does the human playing `r` count as an attacking reply? (win or threat). */
  private _attackIndicator(board: Board, opp: Cell, r: number): number {
    if (wouldWin(board, opp, r)) return 1
    const prev = board.cells[r]
    board.cells[r] = opp
    const threat = hasWinInOne(board, opp)
    board.cells[r] = prev
    return threat ? 1 : 0
  }

  /** Does the human playing `r` count as a defensive reply? (blocks our win). */
  private _defendIndicator(board: Board, ai: Cell, r: number): number {
    return wouldWin(board, ai, r) ? 1 : 0
  }

  /**
   * Re-weights the opponent-reply distribution by the player's style: an
   * attacker's replies that make threats get up-weighted, a defender's blocks
   * get up-weighted. Returns [biasedDist, total] (weights ≥ 0).
   */
  private _styleWeighted(
    board: Board,
    ai: Cell,
    opp: Cell,
    dist: Array<[number, number]>,
  ): Array<[number, number]> {
    if (Math.abs(this.aggression) < 0.2 || dist.length < 2) return dist
    const a = this.aggression
    const damp = 0.7
    return dist.map(([r, pr]) => {
      const att = this._attackIndicator(board, opp, r)
      const def = this._defendIndicator(board, ai, r)
      const w = Math.max(0.02, pr * (1 + a * damp * (att - def)))
      return [r, w] as [number, number]
    })
  }

  constructor(
    model: EvalEngine | null,
    board: Board,
    predictor: OpponentPredictor,
    difficulty: Difficulty = 'hard',
    rng: Rng = defaultRng,
  ) {
    const { engine, error } = requireEngine(model)
    this.model = engine
    this.error = error
    this.board = board
    this.predictor = predictor
    const cfg = DIFFICULTY[difficulty] ?? DIFFICULTY.hard
    this.depth = cfg.depth
    this.topK = cfg.top_k
    this.sampling = cfg.sampling
    this.maxNodes = cfg.max_nodes
    this.entryTemp = cfg.entry_temp
    this.entryMoves = cfg.entry_moves
    this.mistakeRate = cfg.mistake_rate
    this.shiftRate = 0.0
    this.wrongMoveBudget = cfg.wrong_move_budget
    this.moveTemp = cfg.move_temp
    this.blunderKind = cfg.blunder_kind ?? 'random'
    this.defensive = cfg.defensive ?? 0
    this._baseMistakeRate = cfg.mistake_rate
    this._baseMoveTemp = cfg.move_temp
    this._baseWrongMoveBudget = cfg.wrong_move_budget
    this.rng = rng
  }

  private _require(): EvalEngine {
    if (this.model === null) {
      throw new Error(this.error ?? 'no model')
    }
    return this.model
  }

  /** One forward: (win-prob for player, policy logits). */
  private _forward(board: Board, player: Cell): [number, number[]] {
    const { value, policy } = this._require().evalPosition(board.cells, player)
    return [sigmoid(value), policy]
  }

  /** Same tokenization as NativeEngine.evalPosition (side-to-move = token 1). */
  private _normalize(cells: readonly Cell[], side: Cell): { norm: number[]; mask: number[] } {
    const norm = new Array<number>(cells.length)
    const mask = new Array<number>(cells.length)
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i]
      norm[i] = c === 0 ? 0 : c === side ? 1 : 2
      mask[i] = c === 0 ? 1 : 0
    }
    return { norm, mask }
  }

  /** One forward: (win-prob for player, greedy move for player). */
  private _evaluate(board: Board, player: Cell): [number, number] {
    const [value, policy] = this._forward(board, player)
    return [value, argmax(policy)]
  }

  private _greedy(board: Board, player: Cell): number {
    const [, policy] = this._forward(board, player)
    return argmax(policy)
  }

  private _valueFor(board: Board, player: Cell): number {
    const [v] = this._forward(board, player)
    return v
  }

  private _terminal(board: Board, ai: Cell): number | null {
    const { winner: w, over } = board.outcome()
    if (!over) return null
    if (w === ai) return 1.0
    if (w === EMPTY) return 0.5
    return 0.0
  }

  /**
   * Expected value (for AI) after the opponent's likely replies. Operates on
   * a board COPY — the live game board is never mutated.
   */
  private async _opponentExpected(board: Board, ai: Cell, depth: number, topK?: number): Promise<number> {
    if (topK === undefined) topK = this.topK
    this._nodes += 1
    if (this._nodes > this._effMaxNodes()) {
      return this._evaluate(board, ai)[0]
    }

    const opp: Cell = ai === P1 ? P2 : P1
    const dist = this.predictor.likelyMoves(opp, Math.max(1, topK), board)
    if (dist.length === 0) {
      return this._evaluate(board, ai)[0]
    }

    // Bias the predicted replies by the human's style (attacker/defender).
    const replies = this._styleWeighted(board, ai, opp, dist)

    let total = 0.0
    for (const [, p] of replies) total += p

    // leaf level: the replies' values are independent → evaluate in one batch
    if (depth <= 1) {
      return this._opponentLeaves(board, ai, replies, total)
    }

    let exp = 0.0
    const nextTopK = Math.max(1, topK - 1)
    for (const [r, pr] of replies) {
      await this._maybeYield()
      const w = pr / total
      board.apply(r, opp)
      const t = this._terminal(board, ai)
      let rv: number
      if (t !== null) {
        rv = t
      } else {
        const a = this._evaluate(board, ai)[1]
        board.apply(a, ai)
        rv = await this._opponentExpected(board, ai, depth - 1, nextTopK)
        board.cells[a] = EMPTY
      }
      exp += w * rv
      board.cells[r] = EMPTY
    }
    return exp
  }

  /**
   * Batches the depth-1 "opponent reply" leaf evaluations for one node. Same
   * positions, same values as the sequential path — just fewer native calls
   * (and parallel computation on the native side).
   */
  private async _opponentLeaves(
    board: Board,
    ai: Cell,
    dist: Array<[number, number]>,
    total: number,
  ): Promise<number> {
    const opp: Cell = ai === P1 ? P2 : P1
    const n = board.n
    const CHUNK = 8
    const rv = new Map<number, number>()
    const pending: Array<{ r: number; cells: Cell[] }> = []

    for (const [r] of dist) {
      await this._maybeYield()
      board.apply(r, opp)
      const t = this._terminal(board, ai)
      if (t !== null) {
        rv.set(r, t)
      } else {
        pending.push({ r, cells: board.cells.slice() })
      }
      board.cells[r] = EMPTY
    }

    const model = this._require()
    for (let i = 0; i < pending.length; i += CHUNK) {
      const chunk = pending.slice(i, i + CHUNK)
      const flatN: number[] = []
      const flatM: number[] = []
      for (const p of chunk) {
        const { norm, mask } = this._normalize(p.cells, ai)
        flatN.push(...norm)
        flatM.push(...mask)
      }
      let vals: number[]
      if (model.evalPositions) {
        vals = model.evalPositions(flatN, flatM, n).values.map(sigmoid)
      } else {
        vals = chunk.map((p) => sigmoid(model.evalPosition(p.cells, ai, n).value))
      }
      for (let j = 0; j < chunk.length; j++) rv.set(chunk[j].r, vals[j])
      if (i + CHUNK < pending.length) await this._maybeYield()
    }

    let exp = 0.0
    for (const [r, pr] of dist) {
      const w = pr / total
      exp += w * (rv.get(r) ?? 0)
    }
    return exp
  }

  /** Expected value of AI playing m on `board`, searching `depth` plies. */
  private async _evalMove(board: Board, m: number, ai: Cell, depth?: number): Promise<number> {
    if (depth === undefined) depth = this.depth
    this._nodes += 1
    if (this._nodes > this._effMaxNodes()) {
      const opp: Cell = ai === P1 ? P2 : P1
      const v = 1.0 - this._valueFor(board, opp)
      board.cells[m] = EMPTY
      return v
    }
    board.apply(m, ai)
    const t = this._terminal(board, ai)
    if (t !== null) {
      board.cells[m] = EMPTY
      return t
    }
    if (depth <= 1) {
      const opp: Cell = ai === P1 ? P2 : P1
      const v = 1.0 - this._valueFor(board, opp)
      board.cells[m] = EMPTY
      return v
    }
    const v = await this._opponentExpected(board, ai, depth)
    board.cells[m] = EMPTY
    return v
  }

  /** Pick the deliberate-mistake cell. 'random' plays anywhere (EASY, visibly
   * dumb); 'suboptimal' plays a good-but-not-best move (MEDIUM, subtle).
   * Both consume a single RNG draw so parity sequences are unchanged. */
  private _pickBlunder(
    moves: number[],
    scored: Array<[number, number]> | null,
    best: number,
  ): number {
    if (this.blunderKind === 'suboptimal') {
      // medium: a good-but-not-best move; when gifting a win, never the win itself.
      if (scored && scored.length > 1) {
        const maxRank = Math.min(scored.length - 1, 3)
        const rank = 1 + Math.floor(this.rng.random() * maxRank)
        return scored[rank][0]
      }
      const others = moves.filter((m) => m !== best)
      return others.length > 0 ? this.rng.choice(others) : this.rng.choice(moves)
    }
    return this.rng.choice(moves)
  }

  /** An empty box adjacent (Chebyshev ≤ 1) to `best`, or `best` if none. */
  private _shiftedMove(best: number): number {
    const n = this.board.n
    const bz = Math.floor(best / (n * n))
    const br = best % (n * n)
    const by = Math.floor(br / n)
    const bx = br % n
    const cands: number[] = []
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          if (dx === 0 && dy === 0 && dz === 0) continue
          const nx = bx + dx
          const ny = by + dy
          const nz = bz + dz
          if (nx >= 0 && nx < n && ny >= 0 && ny < n && nz >= 0 && nz < n) {
            const i = nx + n * (ny + n * nz)
            if (this.board.cells[i] === EMPTY) cands.push(i)
          }
        }
      }
    }
    if (cands.length === 0) return best
    return this.rng.choice(cands)
  }

  /** Equivalent of Python `__call__`: pick the AI's move for `ai`. */
  async chooseMove(ai: Cell): Promise<number> {
    const moves = this.board.moves()
    if (moves.length === 0) return -1

    // fully random (no search) — the weakest possible player
    if (this.mistakeRate >= 1.0) {
      const final = this.rng.choice(moves)
      this._recordDecision(ai, final, null, 'random')
      return final
    }

    const best = await this._strongMove(ai, moves)
    const scored = this._lastScored
    const forced = this._lastForced

    // Deliberate mistakes, gated by difficulty rules:
    //  - easy (random): may blunder ANY move, EXCEPT when it would win or must
    //    block (never throws away a win / never misses a block).
    //  - medium (suboptimal): exactly `wrong_move_budget` (1) mistakes, and only
    //    when it's ABOUT TO WIN — it gifts the win once, but never fails to block.
    // The effective mistake rate is the reinforcement-weighted (fuzzy) rate.
    const rate = this._effectiveMistakeRate()
    const withinBudget = this._wrongMovesUsed < this.wrongMoveBudget
    // EASY get-out-of-jail: if the search thinks the AI is clearly winning
    // (predicament), it gifts a mistake with a high probability.
    const predicament =
      this.blunderKind === 'random' && scored != null && scored.length > 0 && scored[0][1] >= EASY_PREDICAMENT_THRESHOLD
    const effectiveRate = predicament ? Math.max(rate, EASY_PREDICAMENT_RATE) : rate
    const shouldBlunder =
      rate > 0 &&
      withinBudget &&
      (this.blunderKind === 'random'
        ? forced === null && this.rng.random() < effectiveRate
        : this.blunderKind === 'suboptimal'
          ? forced === 'win' && this.rng.random() < rate
          : false)

    if (shouldBlunder) {
      this._wrongMovesUsed += 1
      const final = this._pickBlunder(moves, scored, best)
      this._recordDecision(ai, final, scored, 'blunder')
      return final
    }

    // "shift by a box": statistically play an empty box adjacent to the
    // suggested one instead of the exact box — the AI stays in the right
    // neighbourhood but makes an off-by-one spatial mistake.
    if (this.shiftRate > 0 && this.rng.random() < this.shiftRate) {
      const shifted = this._shiftedMove(best)
      if (shifted !== best) {
        this._recordDecision(ai, shifted, scored, 'shift')
        return shifted
      }
    }
    this._recordDecision(ai, best, scored, 'search')
    return best
  }

  /** Best move for `side` on the live board, ignoring difficulty blunders and
   *  randomness — this is what a HINT button should recommend. Does not mutate
   *  the board. */
  async getHint(side: Cell): Promise<number> {
    const moves = this.board.moves()
    if (moves.length === 0) return -1
    const search = new Board(this.board.n, this.board.cells)
    // immediate win
    for (const m of moves) {
      search.apply(m, side)
      const { winner: w } = search.outcome()
      search.cells[m] = EMPTY
      if (w === side) return m
    }
    // immediate block: never let the AI win next move
    const opp: Cell = side === P1 ? P2 : P1
    for (const m of moves) {
      search.apply(m, opp)
      const { winner: w } = search.outcome()
      search.cells[m] = EMPTY
      if (w === opp) return m
    }
    // Deep lookahead regardless of difficulty (HINT_DEPTH plies), so the hint
    // is strong even on Easy/Medium — it avoids moves the AI can punish.
    this._nodes = 0
    const scored = (await this._scored(side, moves, search, HINT_DEPTH)).sort((a, b) => b[1] - a[1])
    return scored.length > 0 ? scored[0][0] : -1
  }

  private _recordDecision(
    ai: Cell,
    chosen: number,
    scored: Array<[number, number]> | null,
    kind: AiDecision['kind'],
  ): void {
    const top: ScoredMove[] = (scored ?? []).slice(0, 5).map(([m, v]) => ({
      index: m,
      value: pyRound(v),
    }))
    this.lastDecision = {
      player: ai,
      chosen,
      coord: this.board.coord(chosen),
      kind,
      value: top.length > 0 ? pyRound(top[0].value) : null,
      depth: this.depth,
      scored: top,
      line: this._predictedLine(ai, chosen),
    }
  }

  /** Best-guess future line if the AI plays `chosen`. */
  private _predictedLine(ai: Cell, chosen: number): LineStep[] {
    const native = this._require().predictedLine
    if (native) {
      const r = native(this.board.cells, ai, chosen, this.depth, this.board.n)
      return r.indices.map((idx, i) => ({
        player: r.players[i] as Cell,
        index: idx,
        coord: this.board.coord(idx),
      }))
    }
    const rb = new Board(this.board.n, this.board.cells)
    const line: LineStep[] = [{ player: ai, index: chosen, coord: rb.coord(chosen) }]
    rb.apply(chosen, ai)
    let player: Cell = ai === P1 ? P2 : P1
    let plies = 0
    while (plies < this.depth * 2 && !rb.outcome().over) {
      let idx: number
      if (player === ai) {
        idx = this._greedy(rb, player)
      } else {
        const dist = this.predictor.likelyMoves(player, 1, rb)
        if (dist.length === 0) break
        idx = dist[0][0]
      }
      line.push({ player, index: idx, coord: rb.coord(idx) })
      rb.apply(idx, player)
      player = player === P1 ? P2 : P1
      plies += 1
    }
    return line
  }

  /**
   * The intended strong move: immediate win, block, or search pick. All
   * search happens on a COPY of the board so the live board is never mutated.
   */
  private async _strongMove(ai: Cell, moves: number[]): Promise<number> {
    const search = new Board(this.board.n, this.board.cells)
    const opp: Cell = ai === P1 ? P2 : P1
    this._nodes = 0
    this._lastScored = null
    this._lastForced = null

    // immediate win
    for (const m of moves) {
      search.apply(m, ai)
      const { winner: w } = search.outcome()
      search.cells[m] = EMPTY
      if (w === ai) {
        this._lastForced = 'win'
        return m
      }
    }

    // immediate block: opponent wins next move unless we take that cell
    for (const m of moves) {
      search.apply(m, opp)
      const { winner: w } = search.outcome()
      search.cells[m] = EMPTY
      if (w === opp) {
        this._lastForced = 'block'
        return m
      }
    }

    const scored = (await this._scored(ai, moves, search)).sort((a, b) => b[1] - a[1])
    // Deny the player's overused cells (persisted heatmap): the AI prefers to
    // OCCUPY the cells the player keeps winning with, instead of leaving them
    // open. Skips the Hint path entirely.
    const human: Cell = ai === P1 ? P2 : P1
    const heat = this.predictor.affinity.get(human)
    if (heat) {
      for (const e of scored) {
        const freq = heat.get(e[0]) ?? 0
        if (freq >= 1) e[1] += DENY_WEIGHT * freq
      }
      scored.sort((a, b) => b[1] - a[1])
    }
    // Defensive bias (EASY): prefer moves that BLOCK the player's threats and
    // avoid moves that build the AI's own threats, so the AI plays reactively.
    if (this.defensive > 0) {
      for (const e of scored) {
        const block = this._defendIndicator(search, human, e[0])
        const attack = this._attackIndicator(search, ai, e[0])
        e[1] += DEFENSIVE_WEIGHT * this.defensive * (block - attack)
      }
      scored.sort((a, b) => b[1] - a[1])
    }
    this._lastScored = scored

    // opening-phase temperature: sample among good moves so the AI does not
    // always open the same way (e.g. always center)
    const movesPlayed = this.board.n ** 3 - this.board.emptyCount()
    if (
      this.entryTemp > 0 &&
      this.entryMoves > 0 &&
      movesPlayed < this.entryMoves &&
      scored.length > 1
    ) {
      const top = scored.slice(0, Math.min(5, scored.length))
      const probs = softmax(
        top.map(([, v]) => v),
        this.entryTemp,
      )
      return top[sampleIndex(probs, this.rng.random)][0]
    }

    if (this.sampling && scored.length > 1) {
      const top = scored.slice(0, 3)
      const probs = softmax(
        top.map(([, v]) => v),
        1.0 / 3.0,
      )
      return top[sampleIndex(probs, this.rng.random)][0]
    }

    // sample the final move from the top scores so repeated positions still
    // produce varied play; hard stays near-greedy
    if (this.moveTemp > 0 && scored.length > 1) {
      const top = scored.slice(0, Math.min(3, scored.length))
      const probs = softmax(
        top.map(([, v]) => v),
        this.moveTemp,
      )
      return top[sampleIndex(probs, this.rng.random)][0]
    }
    return scored[0][0]
  }

  /** (move, expected-value) list from the lookahead search. At depth 1 every
   * candidate is a single net evaluation, so they're batched into chunked
   * native calls (parallel on-device); deeper searches fall back to the
   * sequential recursion (identical values either way). */
  private async _scored(
    ai: Cell,
    moves: number[],
    search: Board,
    depthOverride?: number,
  ): Promise<Array<[number, number]>> {
    const depth = depthOverride ?? this.depth
    if (depth <= 1 && this._require().evalPositions) {
      return this._scoredBatched(ai, moves, search)
    }
    const eng = this._require()
    // Native expectimax: the whole depth>1 lookahead runs in one C++ call. The
    // async variant runs it on a background thread so the UI never blocks.
    if (eng.searchScoredAsync) {
      const r = await eng.searchScoredAsync(
        search.cells,
        ai,
        depth,
        this.topK,
        this._effMaxNodes(),
        this.aggression,
        search.n,
      )
      const out: Array<[number, number]> = []
      for (let i = 0; i < r.moves.length; i++) out.push([r.moves[i], r.values[i]])
      return out
    }
    const native = eng.searchScored
    if (native) {
      const r = native(
        search.cells,
        ai,
        depth,
        this.topK,
        this._effMaxNodes(),
        this.aggression,
        search.n,
      )
      const out: Array<[number, number]> = []
      for (let i = 0; i < r.moves.length; i++) out.push([r.moves[i], r.values[i]])
      return out
    }
    const out: Array<[number, number]> = []
    for (const m of moves) {
      await this._maybeYield()
      out.push([m, await this._evalMove(search, m, ai, depth)])
    }
    return out
  }

  private async _scoredBatched(ai: Cell, moves: number[], search: Board): Promise<Array<[number, number]>> {
    const opp: Cell = ai === P1 ? P2 : P1
    const n = search.n
    const CHUNK = 8
    const values = new Map<number, number>()
    const pending: Array<{ m: number; cells: Cell[] }> = []
    let overCap = false

    for (const m of moves) {
      await this._maybeYield()
      this._nodes += 1
      if (this._nodes > this._effMaxNodes()) {
        overCap = true
        continue
      }
      search.apply(m, ai)
      const t = this._terminal(search, ai)
      if (t !== null) {
        values.set(m, t)
      } else {
        pending.push({ m, cells: search.cells.slice() })
      }
      search.cells[m] = EMPTY
    }

    let baseValue = 0
    if (overCap) {
      const r = this._require().evalPosition(search.cells, opp, n)
      baseValue = 1 - sigmoid(r.value)
    }

    const model = this._require()
    for (let i = 0; i < pending.length; i += CHUNK) {
      const chunk = pending.slice(i, i + CHUNK)
      const flatN: number[] = []
      const flatM: number[] = []
      for (const p of chunk) {
        const { norm, mask } = this._normalize(p.cells, opp)
        flatN.push(...norm)
        flatM.push(...mask)
      }
      let vals: number[]
      if (model.evalPositions) {
        vals = model.evalPositions(flatN, flatM, n).values.map(sigmoid)
      } else {
        vals = chunk.map((p) => sigmoid(model.evalPosition(p.cells, opp, n).value))
      }
      for (let j = 0; j < chunk.length; j++) values.set(chunk[j].m, 1 - vals[j])
      if (i + CHUNK < pending.length) await this._maybeYield()
    }

    const out: Array<[number, number]> = []
    for (const m of moves) out.push([m, values.get(m) ?? baseValue])
    return out
  }
}