// Install the fake native engine FIRST (side effect) so the app's module
// resolution sees a working __TfmEngine if any component touches the AI.
import './fakeEngine'

import { createRoot } from 'react-dom/client'
import { Animated } from 'react-native'
import { GestureHandlerRootView } from './gestureShim'

// Asset URLs — static imports are hoisted and resolved before the require
// shim below runs, so map every in-app `require(...)` path to the right URL.
import logoUrl from '../../assets/isocube_logo.png?url'
import fidgetXUrl from '../../assets/sprites/fidget_x.png?url'
import fidgetOUrl from '../../assets/sprites/fidget_o.png?url'
import podiumXUrl from '../../assets/models/podium_celebration_x.png?url'
import podiumOUrl from '../../assets/models/podium_celebration_o.png?url'
import podiumWinXUrl from '../../assets/models/podium_win_x.png?url'
import trophyUrl from '../../assets/models/trophy.png?url'

;(globalThis as any).require = (p: unknown) => {
  const s = String(p)
  if (s.includes('fidget_x')) return { uri: fidgetXUrl }
  if (s.includes('fidget_o')) return { uri: fidgetOUrl }
  if (s.includes('podium_celebration_x')) return { uri: podiumXUrl }
  if (s.includes('podium_celebration_o')) return { uri: podiumOUrl }
  if (s.includes('podium_win_x')) return { uri: podiumWinXUrl }
  if (s.includes('trophy')) return { uri: trophyUrl }
  return { uri: logoUrl }
}

// react-native-web has no native animated module — strip the useNativeDriver
// flag so the app's native-driven Animated.Values fall back to JS animation.
class ValueShim extends (Animated.Value as any) {
  constructor(v: number, _cfg?: any) {
    super(v)
  }
}
Animated.Value = ValueShim as any

// Load the state renderer AFTER the require shim is installed, so module-level
// `require(...)` in TurnFidget / WinnerCelebration resolve to real asset URLs.
async function boot() {
  const { States } = await import('./states')
  const root = document.getElementById('root')!
  createRoot(root).render(
    <GestureHandlerRootView style={{ height: '100%' }}>
      <States />
    </GestureHandlerRootView>,
  )
}
void boot()
