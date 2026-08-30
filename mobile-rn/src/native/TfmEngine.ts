/**
 * JSI host module `TfmEngine` — the on-device C++ transformer.
 *
 * Implemented in native/ (JSI host functions) and surfaced to JS through this
 * file. On device these call straight into the compiled cpp/ engine (the same
 * one the backend uses). In tests/jest there is no native module, so
 * `isAvailable()` returns false and callers use a mock engine instead.
 *
 * Native contract (installed on `globalThis.__TfmEngine`):
 *   load(): boolean
 *   evalPosition(board: number[], mask: number[], n: number): { value, policy }
 *   numel(): number
 */

export interface TfmResult {
  value: number
  policy: number[]
}

interface TfmEngineNative {
  load: () => boolean
  evalPosition: (board: number[], mask: number[], n: number) => TfmResult
  numel: () => number
}

/**
 * Resolve (and, if needed, install) the native binding. The C++ side registers
 * a TurboModule named `TfmEngine` whose `installJSIBindingsWithRuntime` sets
 * `globalThis.__TfmEngine`. Requesting the module through the runtime's
 * `__turboModuleProxy` triggers that install; the read below then finds the
 * HostObject. When no native module is registered (jest / web) this is a no-op.
 */
function resolveNative(): TfmEngineNative | null {
  const g = globalThis as any
  try {
    if (typeof g.__turboModuleProxy === 'function') {
      g.__turboModuleProxy('TfmEngine')
    }
  } catch {
    // The module may still be present below; otherwise we're on jest/web.
  }
  const native: TfmEngineNative | undefined = g.__TfmEngine
  return native ?? null
}

const Native = resolveNative()

/** True when the native module has been installed. */
export function isAvailable(): boolean {
  return Native !== null
}

/** Load the embedded weights (idempotent). Returns success. */
export function load(): boolean {
  if (!Native) return false
  try {
    return Native.load()
  } catch {
    return false
  }
}

/**
 * Synchronous forward pass. `board` is already normalized to tokens {0,1,2}
 * from the side-to-move's perspective; `mask[i]` = 1 for empty/legal cells.
 */
export function evalPosition(
  board: number[],
  mask: number[],
  n: number,
): TfmResult {
  if (!Native) throw new Error('TfmEngine not available')
  return Native.evalPosition(board, mask, n)
}

/** Total parameter count of the loaded model (sanity check). -1 if absent. */
export function numel(): number {
  if (!Native) return -1
  try {
    return Native.numel()
  } catch {
    return -1
  }
}