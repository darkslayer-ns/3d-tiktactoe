/**
 * AsyncStorage persistence for the opponent memory. Only this file touches the
 * native module — keep pure logic in ./opponentMemory so jest can run it.
 */

import AsyncStorage from '@react-native-async-storage/async-storage'
import { type Affinity, parseAffinity, serializeAffinity } from './opponentMemory'

const KEY = 'isocube.opponentAffinity.v1'
const WELCOME_KEY = 'isocube.welcomed.v1'

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