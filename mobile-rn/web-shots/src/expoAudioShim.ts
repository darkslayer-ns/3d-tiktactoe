/**
 * No-op web shim for `expo-audio` — the screenshot harness runs in a browser
 * (react-native-web) where the native audio module doesn't exist. SFX simply
 * don't play in screenshots.
 */

export interface AudioPlayer {
  loop: boolean
  volume: number
  play(): void
  seekTo(position: number): Promise<void>
}

export function createAudioPlayer(): AudioPlayer {
  return {
    loop: false,
    volume: 1,
    play() {},
    seekTo: async () => {},
  }
}

export async function setAudioModeAsync(): Promise<void> {}