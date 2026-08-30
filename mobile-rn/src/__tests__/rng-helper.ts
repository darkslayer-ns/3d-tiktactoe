/**
 * Deterministic scripted RNG for tests. Pops recorded draws in order and
 * validates them, so a TS/Python divergence in the number or order of random
 * calls fails loudly instead of silently producing a different move.
 */

import type { Rng } from '../ai/math'

export interface ScriptedRngEntry {
  kind: 'random' | 'choice'
  value: number
  len?: number
}

export function scriptedRng(entries: ScriptedRngEntry[]): {
  rng: Rng
  remaining: () => number
} {
  let i = 0
  return {
    rng: {
      random(): number {
        const e = entries[i]
        if (!e || e.kind !== 'random') {
          throw new Error(`expected a 'random' draw at log[${i}] (got ${JSON.stringify(e)})`)
        }
        i++
        return e.value
      },
      choice<T>(arr: readonly T[]): T {
        const e = entries[i]
        if (!e || e.kind !== 'choice') {
          throw new Error(`expected a 'choice' draw at log[${i}] (got ${JSON.stringify(e)})`)
        }
        if (e.len !== undefined && e.len !== arr.length) {
          throw new Error(
            `choice length mismatch at log[${i}]: expected ${e.len}, got ${arr.length}`,
          )
        }
        i++
        return arr[e.value]
      },
    },
    remaining: () => entries.length - i,
  }
}