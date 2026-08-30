/**
 * EvalEngine factories.
 *
 * - createNativeEngine(): the real on-device engine backed by the C++ JSI
 *   module (src/native/TfmEngine). Falls back to an error-throwing engine if
 *   the native module is missing (e.g. web/dev).
 * - createMockEngine(): injects a fixed eval function — used by tests and by
 *   the parity gate to replay recorded backend outputs.
 */

import type { Cell } from '../game/types'
import type { EvalEngine, EvalResult } from './types'
import { evalPosition, isAvailable, load } from '../native/TfmEngine'

class NativeEngine implements EvalEngine {
  constructor() {
    load()
  }

  evalPosition(cells: readonly Cell[], player: Cell, n?: number): EvalResult {
    const size = n ?? Math.round(Math.cbrt(cells.length))
    const norm: number[] = new Array(cells.length)
    const mask: number[] = new Array(cells.length)
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i]
      norm[i] = c === 0 ? 0 : c === player ? 1 : 2
      mask[i] = c === 0 ? 1 : 0
    }
    return evalPosition(norm, mask, size)
  }
}

export function createNativeEngine(): EvalEngine {
  if (!isAvailable()) {
    throw new Error(
      'Native TfmEngine unavailable — build the app via `npx expo run:android|ios`',
    )
  }
  return new NativeEngine()
}

export function createMockEngine(
  fn: (cells: readonly Cell[], player: Cell, n: number) => EvalResult,
): EvalEngine {
  return {
    evalPosition(cells, player, n) {
      return fn(cells, player, n ?? Math.round(Math.cbrt(cells.length)))
    },
  }
}