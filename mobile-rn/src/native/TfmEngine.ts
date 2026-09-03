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

export interface TfmBatchResult {
  values: number[]
  policies: number[]
}

interface TfmEngineNative {
  load: () => boolean
  evalPosition: (board: number[], mask: number[], n: number) => TfmResult
  evalPositions: (boards: number[], masks: number[], n: number) => TfmBatchResult
  numel: () => number
  searchScored: (
    cells: number[],
    ai: number,
    depth: number,
    topK: number,
    maxNodes: number,
    aggression: number,
    n: number,
  ) => { moves: number[]; values: number[] }
  searchScoredAsync?: (
    cells: number[],
    ai: number,
    depth: number,
    topK: number,
    maxNodes: number,
    aggression: number,
    n: number,
  ) => Promise<{ moves: number[]; values: number[] }>
  predictedLine: (
    cells: number[],
    ai: number,
    chosen: number,
    depth: number,
    n: number,
  ) => { players: number[]; indices: number[] }
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
    if (g.nativeModuleProxy != null && g.nativeModuleProxy.TfmEngine != null) {
      // Bridgeless mode: `nativeModuleProxy.TfmEngine` is the TurboModule proxy
      // itself. Prefer the C++ HostObject (`__TfmEngine`) if installJSIBindings
      // ran, otherwise use the proxy directly (same methods, sync).
      const proxy = g.nativeModuleProxy.TfmEngine
      const host = g.__TfmEngine
      return host ?? proxy
    }
  } catch (e) {
    console.error('[TfmEngine] resolve threw:', e)
  }
  return g.__TfmEngine ?? null
}

// Lazy resolution: the runtime may not be ready when this module is first
// imported (esp. Android bridgeless), so re-resolve until the binding appears.
let cachedNative: TfmEngineNative | null = null
function native(): TfmEngineNative | null {
  if (!cachedNative) cachedNative = resolveNative()
  return cachedNative
}

/** True when the native module has been installed. */
export function isAvailable(): boolean {
  return native() !== null
}

/** Load the embedded weights (idempotent). Returns success. */
export function load(): boolean {
  const N = native()
  if (!N) return false
  try {
    return N.load()
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
  const N = native()
  if (!N) throw new Error('TfmEngine not available')
  return N.evalPosition(board, mask, n)
}

/**
 * Batched forward over several positions at once (single native call, computed
 * in parallel on the native side). `boards`/`masks` are flat, concatenated
 * n^3-length slices. Returns `count` values and `count * n^3` policies.
 */
export function evalPositions(
  boards: number[],
  masks: number[],
  n: number,
): TfmBatchResult {
  const N = native()
  if (!N) throw new Error('TfmEngine not available')
  return N.evalPositions(boards, masks, n)
}

/** Total parameter count of the loaded model (sanity check). -1 if absent. */
export function numel(): number {
  const N = native()
  if (!N) return -1
  try {
    return N.numel()
  } catch {
    return -1
  }
}

/**
 * Native expectimax: scores every legal move on `cells` (raw {0,1,2}) in one
 * native call. Returns parallel `moves`/`values` arrays in ascending index
 * order. Throws when the native module is unavailable.
 */
export function searchScored(
  cells: number[],
  ai: number,
  depth: number,
  topK: number,
  maxNodes: number,
  aggression: number,
  n: number,
): { moves: number[]; values: number[] } {
  const N = native()
  if (!N) throw new Error('TfmEngine not available')
  return N.searchScored(cells, ai, depth, topK, maxNodes, aggression, n)
}

/**
 * Async search: routed through the TurboModule proxy, which runs the whole
 * lookahead on a background thread and resolves a Promise (the JS thread / UI
 * stays responsive). Falls back to the sync host search if the async method
 * isn't exposed.
 */
export function searchScoredAsync(
  cells: number[],
  ai: number,
  depth: number,
  topK: number,
  maxNodes: number,
  aggression: number,
  n: number,
): Promise<{ moves: number[]; values: number[] }> {
  const g = globalThis as any
  const proxy = g.nativeModuleProxy?.TfmEngine
  if (proxy && typeof proxy.searchScoredAsync === 'function') {
    return proxy.searchScoredAsync(cells, ai, depth, topK, maxNodes, aggression, n)
  }
  return Promise.resolve().then(() =>
    searchScored(cells, ai, depth, topK, maxNodes, aggression, n),
  )
}

/**
 * Native predicted-line telemetry (greedy/likely continuation after `chosen`).
 * Throws when the native module is unavailable.
 */
export function predictedLine(
  cells: number[],
  ai: number,
  chosen: number,
  depth: number,
  n: number,
): { players: number[]; indices: number[] } {
  const N = native()
  if (!N) throw new Error('TfmEngine not available')
  return N.predictedLine(cells, ai, chosen, depth, n)
}