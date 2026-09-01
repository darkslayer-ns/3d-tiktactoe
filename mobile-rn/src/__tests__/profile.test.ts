import { Board, P1, P2 } from '../game/board'
import { analyzeMove, classifyMove, hasWinInOne, PerceptionProfile, PlayerProfile, wouldWin } from '../ai/profile'

describe('player profile (attacker/defender)', () => {
  const b = () => new Board(3)

  it('wouldWin detects an immediate win', () => {
    const board = b()
    board.apply(board.idx(0, 0, 0), P1)
    board.apply(board.idx(1, 0, 0), P1)
    expect(wouldWin(board, P1, board.idx(2, 0, 0))).toBe(true)
    expect(wouldWin(board, P1, board.idx(0, 1, 0))).toBe(false)
  })

  it('hasWinInOne detects a pending threat', () => {
    const board = b()
    board.apply(board.idx(0, 0, 0), P1)
    board.apply(board.idx(1, 0, 0), P1)
    expect(hasWinInOne(board, P1)).toBe(true)
    expect(hasWinInOne(b(), P1)).toBe(false)
  })

  it('classifies a block as defend', () => {
    const board = b()
    board.apply(board.idx(0, 0, 0), P2)
    board.apply(board.idx(1, 0, 0), P2)
    expect(classifyMove(board, P1, board.idx(2, 0, 0))).toBe('defend')
  })

  it('classifies a threat-creating move as attack', () => {
    const board = b()
    board.apply(board.idx(0, 0, 0), P1)
    expect(classifyMove(board, P1, board.idx(1, 0, 0))).toBe('attack')
  })

  it('classifies a setup move as neutral', () => {
    expect(classifyMove(b(), P1, b().idx(0, 0, 0))).toBe('neutral')
  })

  it('classifyMove does not mutate the board', () => {
    const board = b()
    const before = board.cells.slice()
    classifyMove(board, P1, board.idx(0, 0, 0))
    expect(board.cells).toEqual(before)
  })

  it('PlayerProfile aggregates aggression', () => {
    const attacker = new PlayerProfile()
    expect(attacker.aggression()).toBe(0)
    for (let i = 0; i < 5; i++) attacker.record('attack')
    expect(attacker.aggression()).toBeGreaterThan(0.9)

    const defender = new PlayerProfile()
    for (let i = 0; i < 5; i++) defender.record('defend')
    expect(defender.aggression()).toBeLessThan(-0.9)
  })

  it('analyzeMove reports the line axis (axis/face/space)', () => {
    // axis win: (0,0,0)-(2,0,0)
    const ax = b()
    ax.apply(ax.idx(0, 0, 0), P1)
    ax.apply(ax.idx(1, 0, 0), P1)
    expect(analyzeMove(ax, P1, ax.idx(2, 0, 0))).toEqual({ style: 'attack', axis: 'axis' })

    // face-diagonal win: (0,0,0)-(2,2,0)
    const face = b()
    face.apply(face.idx(0, 0, 0), P1)
    face.apply(face.idx(1, 1, 0), P1)
    expect(analyzeMove(face, P1, face.idx(2, 2, 0))).toEqual({ style: 'attack', axis: 'face' })

    // space-diagonal block: AI (P2) threatens (0,0,0)-(2,2,2), human blocks
    const space = b()
    space.apply(space.idx(0, 0, 0), P2)
    space.apply(space.idx(1, 1, 1), P2)
    expect(analyzeMove(space, P1, space.idx(2, 2, 2))).toEqual({ style: 'defend', axis: 'space' })
  })

  it('PerceptionProfile scores deeper lines higher', () => {
    const p = new PerceptionProfile(1.0)
    expect(p.score()).toBe(0.5) // unknown
    p.record('axis')
    p.record('axis')
    expect(p.score()).toBeCloseTo(0)
    const p2 = new PerceptionProfile(1.0)
    p2.record('space')
    p2.record('space')
    expect(p2.score()).toBeCloseTo(1)
    const p3 = new PerceptionProfile(1.0)
    p3.record('face')
    p3.record('space')
    expect(p3.score()).toBeCloseTo(0.75)
  })
})