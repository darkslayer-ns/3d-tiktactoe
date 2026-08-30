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

export const DIFFICULTY_TEMPERATURE: Record<Difficulty, number> = {
  easy: 1.5,
  medium: 0.7,
  hard: 0.0,
}

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
}

/** Pure RUNTIME config — all difficulties share the same strong base weights. */
export const DIFFICULTY: Record<Difficulty, DifficultyConfig> = {
  easy: {
    depth: 2,
    top_k: 3,
    sampling: false,
    max_nodes: 220,
    entry_temp: 1.0,
    entry_moves: 2,
    wrong_move_budget: 1,
    mistake_rate: 0.5,
    move_temp: 0.8,
  },
  medium: {
    depth: 3,
    top_k: 3,
    sampling: false,
    max_nodes: 220,
    entry_temp: 0.7,
    entry_moves: 1,
    wrong_move_budget: 1,
    mistake_rate: 0.5,
    move_temp: 0.5,
  },
  hard: {
    depth: 4,
    top_k: 3,
    sampling: false,
    max_nodes: 220,
    entry_temp: 0.5,
    entry_moves: 1,
    wrong_move_budget: 0,
    mistake_rate: 0.0,
    move_temp: 0.15,
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
  readonly mistakeRate: number
  readonly shiftRate: number
  readonly wrongMoveBudget: number
  readonly moveTemp: number
  lastDecision: AiDecision | null = null
  private rng: Rng
  private _nodes = 0
  private _wrongMovesUsed = 0
  private _lastScored: Array<[number, number]> | null = null

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
  private _opponentExpected(board: Board, ai: Cell, depth: number, topK?: number): number {
    if (topK === undefined) topK = this.topK
    this._nodes += 1
    if (this._nodes > this.maxNodes) {
      return this._evaluate(board, ai)[0]
    }

    const opp: Cell = ai === P1 ? P2 : P1
    const dist = this.predictor.likelyMoves(opp, Math.max(1, topK), board)
    if (dist.length === 0) {
      return this._evaluate(board, ai)[0]
    }

    let total = 0.0
    for (const [, p] of dist) total += p
    let exp = 0.0
    const nextTopK = Math.max(1, topK - 1)
    for (const [r, pr] of dist) {
      const w = pr / total
      board.apply(r, opp)
      const t = this._terminal(board, ai)
      let rv: number
      if (t !== null) {
        rv = t
      } else if (depth > 1) {
        const a = this._evaluate(board, ai)[1]
        board.apply(a, ai)
        rv = this._opponentExpected(board, ai, depth - 1, nextTopK)
        board.cells[a] = EMPTY
      } else {
        rv = this._evaluate(board, ai)[0]
      }
      exp += w * rv
      board.cells[r] = EMPTY
    }
    return exp
  }

  /** Expected value of AI playing m on `board`, searching `depth` plies. */
  private _evalMove(board: Board, m: number, ai: Cell, depth?: number): number {
    if (depth === undefined) depth = this.depth
    this._nodes += 1
    if (this._nodes > this.maxNodes) {
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
    const v = this._opponentExpected(board, ai, depth)
    board.cells[m] = EMPTY
    return v
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
  chooseMove(ai: Cell): number {
    const moves = this.board.moves()
    if (moves.length === 0) return -1

    // fully random (no search) — the weakest possible player
    if (this.mistakeRate >= 1.0) {
      const final = this.rng.choice(moves)
      this._recordDecision(ai, final, null, 'random')
      return final
    }

    const best = this._strongMove(ai, moves)
    const scored = this._lastScored

    // deliberate wrong moves: blunder at most `wrong_move_budget` times per
    // game (the cap makes "easy" beatable while later moves stay sane).
    // Applied AFTER the strong move so it overrides even an immediate
    // win/block — a genuine blunder.
    if (
      this.mistakeRate > 0 &&
      this._wrongMovesUsed < this.wrongMoveBudget &&
      this.rng.random() < this.mistakeRate
    ) {
      this._wrongMovesUsed += 1
      const final = this.rng.choice(moves)
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
  private _strongMove(ai: Cell, moves: number[]): number {
    const search = new Board(this.board.n, this.board.cells)
    const opp: Cell = ai === P1 ? P2 : P1
    this._nodes = 0
    this._lastScored = null

    // immediate win
    for (const m of moves) {
      search.apply(m, ai)
      const { winner: w } = search.outcome()
      search.cells[m] = EMPTY
      if (w === ai) return m
    }

    // immediate block: opponent wins next move unless we take that cell
    for (const m of moves) {
      search.apply(m, opp)
      const { winner: w } = search.outcome()
      search.cells[m] = EMPTY
      if (w === opp) return m
    }

    const scored = this._scored(ai, moves, search).sort((a, b) => b[1] - a[1])
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

  /** (move, expected-value) list from the lookahead search. */
  private _scored(ai: Cell, moves: number[], search: Board): Array<[number, number]> {
    const out: Array<[number, number]> = []
    for (const m of moves) {
      out.push([m, this._evalMove(search, m, ai)])
    }
    return out
  }
}