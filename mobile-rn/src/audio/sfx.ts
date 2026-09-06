/**
 * Pure SFX playback core — no React Native / expo imports, so it runs under
 * jest. `SoundManager.ts` adapts `expo-audio` players to the `SfxTrack`
 * interface and wires in haptics.
 *
 * Policy (overlap): each sound restarts only ITS OWN track; other tracks keep
 * playing (they mix via `interruptionMode: 'mixWithOthers'`). Nothing is cut,
 * deferred, or priority-ranked — so a game-over sting is never swallowed by
 * the sound that preceded it. The rewind is awaited before `play()` because
 * calling `play()` while the track is still at the end of its clip silently
 * no-ops.
 */

export type SfxName =
  | 'intro'
  | 'select'
  | 'place'
  | 'ai'
  | 'win'
  | 'lose'
  | 'draw'
  | 'click'

export interface SfxTrack {
  seekTo(pos: number): Promise<unknown>
  play(): void
}

export interface SfxController {
  play(name: SfxName): Promise<void>
}

export function createSfxController(opts: {
  getTrack: (name: SfxName) => SfxTrack | null
  haptic: (name: SfxName) => void
}): SfxController {
  const { getTrack, haptic } = opts
  return {
    play(name: SfxName): Promise<void> {
      haptic(name)
      const track = getTrack(name)
      if (!track) return Promise.resolve()
      return track
        .seekTo(0)
        .then(() => track.play())
        .catch(() => track.play())
    },
  }
}