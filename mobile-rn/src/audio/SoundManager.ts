/**
 * Central audio manager for ISOCUBE. Thin adapter: maps SFX names to
 * `expo-audio` players and haptics, then delegates to the pure controller in
 * ./sfx (overlap policy — sounds mix, never cut). Assets live in
 * assets/sounds/*.m4a (AAC — the only portable codec: AVFoundation on iOS won't
 * decode .ogg).
 */

import { createAudioPlayer, setAudioModeAsync, type AudioPlayer } from 'expo-audio'
import * as Haptics from 'expo-haptics'
import { createSfxController, type SfxName, type SfxTrack } from './sfx'

const SFX = {
  intro: require('../../assets/sounds/intro.m4a'),
  select: require('../../assets/sounds/select.m4a'),
  place: require('../../assets/sounds/place.m4a'),
  ai: require('../../assets/sounds/ai_move.m4a'),
  win: require('../../assets/sounds/win.m4a'),
  lose: require('../../assets/sounds/lose.m4a'),
  draw: require('../../assets/sounds/draw.m4a'),
  click: require('../../assets/sounds/click.m4a'),
} as const

export type { SfxName }

let sfxPlayers = new Map<SfxName, SfxTrack>()
let audioModeReady = false

function ensureAudioMode(): void {
  if (audioModeReady) return
  audioModeReady = true
  void setAudioModeAsync({
    // .playback category → the game follows the SYSTEM volume, so the hardware
    // volume buttons control it continuously (turn volume to zero to mute).
    playsInSilentMode: true,
    shouldPlayInBackground: false,
    interruptionMode: 'mixWithOthers',
  }).catch(() => {})
}

/** Lazily create (on first use) the expo-audio player for `name`. The
 * `require()` result is a numeric asset ID (Metro's `registerAsset`), which
 * `expo-audio` resolves via `expo-asset`. */
function getPlayer(name: SfxName): SfxTrack | null {
  const existing = sfxPlayers.get(name)
  if (existing) return existing
  try {
    const p: AudioPlayer = createAudioPlayer(SFX[name])
    p.loop = false
    p.volume = 1
    const track: SfxTrack = {
      seekTo: (pos) => p.seekTo(pos),
      play: () => p.play(),
    }
    sfxPlayers.set(name, track)
    return track
  } catch {
    return null
  }
}

const controller = createSfxController({
  getTrack: getPlayer,
  haptic: hapticFor,
})

export function playSfx(name: SfxName): Promise<void> {
  ensureAudioMode()
  return controller.play(name)
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

/** Selection-tick haptic for menu toggles / slider / cell selection. */
export function hapticSelection(): void {
  void Haptics.selectionAsync().catch(() => {})
}