import { P1, P2, EMPTY } from '../game/types'
import {
  applyResult,
  emptyAffinity,
  parseAffinity,
  serializeAffinity,
  WIN_BOOST,
  LOSS_DECAY,
  type Affinity,
} from '../ai/opponentMemory'

function seed(): Affinity {
  const aff: Affinity = new Map()
  const p1 = new Map<number, number>([
    [4, 3.0],
    [13, 1.0],
  ])
  const p2 = new Map<number, number>([[13, 2.0]])
  aff.set(P1, p1)
  aff.set(P2, p2)
  return aff
}

describe('opponentMemory', () => {
  it('boosts the winner and penalizes the loser', () => {
    const aff = seed()
    applyResult(aff, P1, P2)
    expect(aff.get(P1)!.get(4)).toBeCloseTo(3.0 * WIN_BOOST)
    expect(aff.get(P1)!.get(13)).toBeCloseTo(1.0 * WIN_BOOST)
    expect(aff.get(P2)!.get(13)).toBeCloseTo(2.0 * LOSS_DECAY)
  })

  it('is a no-op on a draw', () => {
    const aff = seed()
    applyResult(aff, EMPTY, P1)
    expect(aff.get(P1)!.get(4)).toBe(3.0)
    expect(aff.get(P2)!.get(13)).toBe(2.0)
  })

  it('round-trips through serialization', () => {
    const aff = seed()
    const parsed = parseAffinity(serializeAffinity(aff))
    expect(parsed.get(P1)!.get(4)).toBe(3.0)
    expect(parsed.get(P1)!.get(13)).toBe(1.0)
    expect(parsed.get(P2)!.get(13)).toBe(2.0)
  })

  it('returns an empty map for missing or corrupt data', () => {
    expect(parseAffinity(null)).toEqual(new Map())
    expect(parseAffinity('not json{')).toEqual(new Map())
    expect(emptyAffinity()).toEqual(new Map())
  })

  it('clones side/cell keys numerically', () => {
    const parsed = parseAffinity(serializeAffinity(seed()))
    for (const [side, row] of parsed) {
      expect(typeof side).toBe('number')
      for (const cell of row.keys()) expect(typeof cell).toBe('number')
    }
  })
})