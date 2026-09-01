/**
 * AsyncStorage persistence for the opponent memory. Only this file touches the
 * native module — keep pure logic in ./opponentMemory so jest can run it.
 */

import AsyncStorage from '@react-native-async-storage/async-storage'
import { type Affinity, parseAffinity, serializeAffinity } from './opponentMemory'
import type { ProfileCounts } from './profile'
import type { GameStats } from './stats'

const KEY = 'isocube.opponentAffinity.v1'
const WELCOME_KEY = 'isocube.welcomed.v1'
const PROFILE_KEY = 'isocube.playerProfile.v1'
const STATS_KEY = 'isocube.gameStats.v1'
const PERCEPTION_KEY = 'isocube.perception.v1'

export async function loadAffinity(): Promise<Affinity> {
  try {
    return parseAffinity(await AsyncStorage.getItem(KEY))
  } catch {
    return parseAffinity(null)
  }
}

export async function saveAffinity(aff: Affinity): Promise<void> {
  await AsyncStorage.setItem(KEY, serializeAffinity(aff))
}

/** True once the user has already seen the first-launch welcome. */
export async function getWelcomed(): Promise<boolean> {
  try {
    return (await AsyncStorage.getItem(WELCOME_KEY)) === '1'
  } catch {
    return false
  }
}

export async function setWelcomed(): Promise<void> {
  try {
    await AsyncStorage.setItem(WELCOME_KEY, '1')
  } catch {
    // non-fatal
  }
}

export async function loadProfile(): Promise<ProfileCounts | null> {
  try {
    const raw = await AsyncStorage.getItem(PROFILE_KEY)
    if (!raw) return null
    const p = JSON.parse(raw)
    if (typeof p.attack !== 'number') return null
    return {
      attack: p.attack ?? 0,
      defend: p.defend ?? 0,
      neutral: p.neutral ?? 0,
    }
  } catch {
    return null
  }
}

export async function saveProfile(p: ProfileCounts): Promise<void> {
  try {
    await AsyncStorage.setItem(PROFILE_KEY, JSON.stringify(p))
  } catch {
    // non-fatal
  }
}

export async function loadStats(): Promise<GameStats | null> {
  try {
    const raw = await AsyncStorage.getItem(STATS_KEY)
    if (!raw) return null
    const s = JSON.parse(raw)
    if (typeof s.wins !== 'number') return null
    return { wins: s.wins ?? 0, losses: s.losses ?? 0, draws: s.draws ?? 0 }
  } catch {
    return null
  }
}

export async function saveStats(s: GameStats): Promise<void> {
  try {
    await AsyncStorage.setItem(STATS_KEY, JSON.stringify(s))
  } catch {
    // non-fatal
  }
}

export async function loadPerception(): Promise<{ axis: number; face: number; space: number } | null> {
  try {
    const raw = await AsyncStorage.getItem(PERCEPTION_KEY)
    if (!raw) return null
    const p = JSON.parse(raw)
    if (typeof p.space !== 'number') return null
    return { axis: p.axis ?? 0, face: p.face ?? 0, space: p.space ?? 0 }
  } catch {
    return null
  }
}

export async function savePerception(p: { axis: number; face: number; space: number }): Promise<void> {
  try {
    await AsyncStorage.setItem(PERCEPTION_KEY, JSON.stringify(p))
  } catch {
    // non-fatal
  }
}