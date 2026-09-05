/**
 * Central audio manager for ISOCUBE. Wraps `expo-audio` players for the
 * one-shot SFX (select, place, AI move). Assets live in assets/sounds/*.ogg.
 */

import { createAudioPlayer, setAudioModeAsync, type AudioPlayer } from 'expo-audio'
import * as Haptics from 'expo-haptics'

const SFX = {
  intro: require('../../assets/sounds/intro.ogg'),
  select: require('../../assets/sounds/select.ogg'),
  place: require('../../assets/sounds/place.ogg'),
  ai: require('../../assets/sounds/ai_move.ogg'),
  win: require('../../assets/sounds/win.ogg'),
  lose: require('../../assets/sounds/lose.ogg'),
  draw: require('../../assets/sounds/draw.ogg'),
  click: require('../../assets/sounds/click.ogg'),
} as const

export type SfxName = keyof typeof SFX

let sfxPlayers: Partial<Record<SfxName, AudioPlayer>> = {}
let initialized = false

/** JSON-compatible shape of the runtime `require()`'d asset — only `uri` is used. */
function uriOf(asset: unknown): string | null {
  if (asset && typeof asset === 'object' && 'uri' in (asset as Record<string, unknown>)) {
    return (asset as { uri: string }).uri as string
  }
  return null
}

/**
 * Configure the global audio session and lazily create the players. Safe to
 * call more than once (subsequent calls no-op). Players are created against
 * the resolved asset URI because the Expo bundler resolves `require()` of
 * audio assets to a JSON descriptor rather than a string path.
 */
export function ensureAudio(): void {
  if (initialized) return
  initialized = true

  void setAudioModeAsync({
    playsInSilentMode: true,
    shouldPlayInBackground: false,
    interruptionMode: 'mixWithOthers',
  }).catch(() => {})

  for (const [name, mod] of Object.entries(SFX) as Array<[SfxName, unknown]>) {
    const uri = uriOf(mod)
    if (!uri) continue
    const p = createAudioPlayer(uri)
    p.loop = false
    p.volume = 1
    sfxPlayers[name] = p
  }
}

/** Play a one-shot SFX, restarting from the beginning if it was still ringing. */
export function playSfx(name: SfxName): void {
  ensureAudio()
  const p = sfxPlayers[name]
  if (!p) return
  p.seekTo(0).catch(() => {})
  p.play()
  hapticFor(name)
}

/** Fire the matching haptic alongside a SFX (no-op when unsupported). */
function hapticFor(name: SfxName): void {
  switch (name) {
    case 'select':
      void Haptics.selectionAsync().catch(() => {})
      break
    case 'place':
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {})
      break
    case 'click':
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {})
      break
    case 'win':
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {})
      break
    case 'lose':
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {})
      break
    case 'draw':
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {})
      break
    default:
      // 'ai' — no haptic (machine action); 'intro' — no haptic
      break
  }
}

/** Selection-tick haptic for menu toggles / slider (no sound). */
export function hapticSelection(): void {
  void Haptics.selectionAsync().catch(() => {})
}
