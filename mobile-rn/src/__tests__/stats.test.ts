import { EMPTY, P1, P2 } from '../game/types'
import { adaptiveLevel, emptyStats, recordResult, winRate } from '../ai/stats'

describe('adaptive-difficulty stats', () => {
  it('records results from the human perspective', () => {
    const s = emptyStats()
    recordResult(s, P1, P1, 1.0) // human (P1) wins
    expect(s.wins).toBe(1)
    recordResult(s, P2, P1, 1.0) // human loses
    expect(s.losses).toBe(1)
    recordResult(s, EMPTY, P1, 1.0) // draw
    expect(s.draws).toBe(1)
  })

  it('winRate ignores draws and is neutral while few games', () => {
    expect(winRate(emptyStats())).toBe(0.5)
    const s = emptyStats()
    recordResult(s, P1, P1, 1.0)
    recordResult(s, P2, P1, 1.0)
    expect(winRate(s)).toBe(0.5)
  })

  it('adaptiveLevel eases a losing player and hardens a winning one', () => {
    const losing = emptyStats()
    for (let i = 0; i < 10; i++) recordResult(losing, P2, P1, 1.0) // human loses 10x
    expect(adaptiveLevel(losing)).toBeGreaterThan(0) // ease up

    const winning = emptyStats()
    for (let i = 0; i < 10; i++) recordResult(winning, P1, P1, 1.0) // human wins 10x
    expect(adaptiveLevel(winning)).toBeLessThan(0) // harden
  })
})