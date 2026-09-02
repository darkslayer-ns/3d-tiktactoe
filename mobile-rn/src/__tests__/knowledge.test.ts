import { Board } from '../game/board'
import { buildKnowledgeSnapshot } from '../dev/knowledge'
import { createMockEngine } from '../ai/engine'
import { OpponentPredictor } from '../ai/predictor'
import { LookaheadMover } from '../ai/mover'
import { PlayerProfile, PerceptionProfile } from '../ai/profile'
import { emptyStats } from '../ai/stats'

function fixedEngine() {
  // Deterministic: value = 1.0, policy = index/100 (so higher cells rank higher).
  return createMockEngine((cells, _player, n) => {
    const size = n ** 3
    const policy = Array.from({ length: size }, (_, i) => i / 100)
    for (let i = 0; i < size; i++) {
      if (cells[i] !== 0) policy[i] = -Infinity
    }
    return { value: 1.0, policy }
  })
}

describe('buildKnowledgeSnapshot', () => {
  it('computes win prob, best move and predictions from the engine', () => {
    const board = new Board(3)
    board.apply(0, 1)
    const engine = fixedEngine()
    const predictor = new OpponentPredictor(board, engine)
    const mover = new LookaheadMover(engine, board, predictor, 'hard')
    const snap = buildKnowledgeSnapshot({
      engine,
      board,
      humanSide: 1,
      difficulty: 'hard',
      predictor,
      mover,
      profile: new PlayerProfile(),
      perception: new PerceptionProfile(),
      stats: emptyStats(),
      adaptive: 0,
    })

    expect(snap.winProbHuman).toBeCloseTo(0.731, 3) // sigmoid(1.0)
    // Cell 0 is occupied by the human -> best legal is the highest-index empty cell.
    expect(snap.bestMoveIndex).toBe(26)
    // Occupied cell 0 is never predicted; every legal cell appears exactly once.
    const indices = snap.predictions.map((p) => p.index)
    expect(indices).not.toContain(0)
    expect(indices).toContain(26)
    expect(new Set(indices).size).toBe(snap.predictions.length)
    expect(snap.predictions.every((p) => p.prob > 0)).toBe(true)
    // Ranked descending by probability.
    for (let i = 1; i < snap.predictions.length; i++) {
      expect(snap.predictions[i - 1].prob).toBeGreaterThanOrEqual(snap.predictions[i].prob)
    }
  })

  it('exposes affinity, style and form data', () => {
    const board = new Board(3)
    const engine = fixedEngine()
    const predictor = new OpponentPredictor(board, engine)
    predictor.record(1, 12)
    predictor.record(1, 12)
    predictor.record(1, 5)
    const profile = new PlayerProfile()
    profile.record('attack')
    profile.record('attack')
    profile.record('defend')
    const perception = new PerceptionProfile()
    perception.record('space')
    const stats = emptyStats()
    stats.wins = 5
    stats.losses = 5

    const snap = buildKnowledgeSnapshot({
      engine,
      board,
      humanSide: 1,
      difficulty: 'medium',
      predictor,
      mover: null,
      profile,
      perception,
      stats,
      adaptive: 0,
    })

    expect(snap.affinity[0].index).toBe(12)
    expect(snap.affinity[0].weight).toBe(2)
    expect(snap.aggression).toBeGreaterThan(0)
    expect(snap.perception).toBeGreaterThan(0.5)
    expect(snap.winRate).toBeCloseTo(0.5)
    expect(snap.lastDecision).toBeNull()
  })

  it('tolerates a missing engine', () => {
    const board = new Board(3)
    const snap = buildKnowledgeSnapshot({
      engine: null,
      board,
      humanSide: 1,
      difficulty: 'easy',
      predictor: null,
      mover: null,
      profile: null,
      perception: null,
      stats: emptyStats(),
      adaptive: 0,
    })
    expect(snap.winProbHuman).toBeNull()
    expect(snap.predictions).toEqual([])
    expect(snap.affinity).toEqual([])
    expect(snap.winRate).toBe(0.5)
  })
})