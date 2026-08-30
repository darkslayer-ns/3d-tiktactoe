/**
 * Tiny numeric helpers, byte-for-byte port of the helpers in
 * backend/ml/cpp_inference.py. Pure functions — no RN/three imports.
 */

/** Stable two-branch sigmoid (mirrors cpp_inference.sigmoid). */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x)
    return 1.0 / (1.0 + z)
  }
  const z = Math.exp(x)
  return z / (1.0 + z)
}

/** Max-subtracted softmax with a temperature floor of 1e-8. */
export function softmax(logits: readonly number[], temperature = 1.0): number[] {
  const t = Math.max(temperature, 1e-8)
  const m = Math.max(...logits)
  const exps = logits.map((v) => Math.exp((v - m) / t))
  let total = 0.0
  for (const e of exps) total += e
  return exps.map((e) => e / total)
}

/** Sample an index from a probability distribution (cumulative threshold). */
export function sampleIndex(probs: readonly number[], rng: () => number = Math.random): number {
  const r = rng()
  let acc = 0.0
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]
    if (r < acc) return i
  }
  return probs.length - 1
}

/** Index of the largest logit (ties keep the first occurrence). */
export function argmax(logits: readonly number[]): number {
  let best = logits[0]
  let bestI = 0
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > best) {
      best = logits[i]
      bestI = i
    }
  }
  return bestI
}

/**
 * Python's round(x, ndigits) — round-half-to-even — for the telemetry values
 * stored in AiDecision.scored. Exact binary-float tie cases are vanishingly
 * rare for these scores; this is close enough for display parity.
 */
export function pyRound(value: number, digits = 3): number {
  const factor = 10 ** digits
  const scaled = value * factor
  const floored = Math.floor(scaled)
  const frac = scaled - floored
  if (frac < 0.5) return floored / factor
  if (frac > 0.5) return (floored + 1) / factor
  return (floored % 2 === 0 ? floored : floored + 1) / factor
}

/** Deterministic, injectable random source replacing Python's `random` module. */
export interface Rng {
  random(): number
  choice<T>(arr: readonly T[]): T
}

export const defaultRng: Rng = {
  random: () => Math.random(),
  choice: (arr) => arr[Math.floor(Math.random() * arr.length)],
}