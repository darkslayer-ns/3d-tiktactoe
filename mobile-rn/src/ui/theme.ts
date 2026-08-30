/**
 * Neon Cube theme — mirrors the web frontend's neon palette
 * (frontend/src/styles.css) as typed constants for the RN UI.
 */

export const Theme = {
  /** app background (deep slate-black) */
  bg: '#020617',
  /** panel / cell slot fill */
  panel: '#0f172a',
  /** panel border / wireframe shell */
  border: '#1e293b',
  /** primary neon — X marks, brand, accents */
  cyan: '#22d3ee',
  /** darker cyan used under pulsing slots */
  cyanDark: '#164e63',
  /** deeper cyan for primary button gradient */
  cyanDeep: '#0ea5e9',
  /** secondary neon — O marks */
  pink: '#f472b6',
  /** winning beam glow */
  purple: '#e879f9',
  /** pending-cell highlight */
  yellow: '#fef08a',
  /** predicted-move glow */
  lime: '#a3e635',
  /** primary text */
  text: '#e2e8f0',
  /** secondary text / labels */
  muted: '#64748b',
  /** tertiary text */
  subtle: '#94a3b8',
  /** destructive / error text */
  danger: '#f87171',
  white: '#ffffff',
} as const

/** 4px-based spacing scale: spacing(n) = n*4px. */
export const spacing = (n: number): number => n * 4

/** Font-size scale (plain passthrough so call sites read semantically). */
export const fontSize = (n: number): number => n

/** Corner-radius scale: radius(n) = n*4px. */
export const radius = (n: number): number => n * 4