/**
 * Fake native TfmEngine installed on globalThis.__TfmEngine BEFORE any app
 * module imports, so src/native/TfmEngine.ts's resolveNative() finds it and
 * createNativeEngine() works. Deterministic, plausible outputs — enough to
 * make the AI drive games for screenshots.
 */

function plausibleEval(board: number[], mask: number[], n: number) {
  const N = n ** 3
  // center bias → the AI "prefers" central cells like a real trained net
  const cx = Math.floor(n / 2)
  const center = cx + n * (cx + n * cx)
  const empties = mask.reduce((a, b) => a + (b ? 1 : 0), 0)
  const value = 0.5 + (empties - 13) * 0.01
  const policy: number[] = new Array(N)
  for (let i = 0; i < N; i++) {
    if (!mask[i]) policy[i] = -Infinity
    else policy[i] = i === center ? 2.0 : 1.0 - Math.abs(i - center) * 0.02
  }
  return { value, policy }
}

const fake: any = {
  load: () => true,
  numel: () => 106690,
  evalPosition: (board: number[], mask: number[], n: number) =>
    plausibleEval(board, mask, n),
  evalPositions: (boards: number[], masks: number[], n: number) => {
    const N = n ** 3
    const values: number[] = []
    const policies: number[] = []
    for (let i = 0; i < boards.length; i += N) {
      const r = plausibleEval(boards.slice(i, i + N), masks.slice(i, i + N), n)
      values.push(r.value)
      policies.push(...r.policy)
    }
    return { values, policies }
  },
}

;(globalThis as any).__TfmEngine = fake
export {}