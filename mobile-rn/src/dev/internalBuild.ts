/**
 * Internal-build gate.
 *
 * Dev/diagnostic tooling (e.g. the ModelKnowledgePanel) is compiled ONLY into
 * internal builds, never into shipped release binaries:
 *
 *   - `__DEV__` is true in every Metro dev build (and removed in release).
 *   - `EXPO_PUBLIC_INTERNAL_DEBUG=1` forces the tooling into RELEASE builds too,
 *     for internal QA/test distribution. Set it when producing an internal
 *     build, e.g.:
 *
 *       EXPO_PUBLIC_INTERNAL_DEBUG=1 npx expo run:ios --configuration Release
 *       EXPO_PUBLIC_INTERNAL_DEBUG=1 npx expo run:android --variant release
 *
 *   `EXPO_PUBLIC_INTERNAL_BUILD=1` is accepted as a legacy alias.
 *
 * babel-preset-expo inlines literal `process.env.EXPO_PUBLIC_*` member
 * expressions at bundle time, so the value is baked in per-build — an ordinary
 * release build compiles this out. (That is why the access must be the literal
 * `process.env.EXPO_PUBLIC_INTERNAL_DEBUG` chain, not an indirection.)
 */

declare const process: { env: Record<string, string | undefined> }

export const IS_INTERNAL_BUILD: boolean =
  __DEV__ === true ||
  process.env.EXPO_PUBLIC_INTERNAL_DEBUG === '1' ||
  process.env.EXPO_PUBLIC_INTERNAL_BUILD === '1'

/** Debug affordances (opening the panel, etc.) — only ever wired when internal. */
export function onlyInternal<T>(value: T): T | undefined {
  return IS_INTERNAL_BUILD ? value : undefined
}