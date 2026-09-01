// Install the fake native engine FIRST (side effect) so the app's module
// resolution sees a working __TfmEngine if any component touches the AI.
import './fakeEngine'

// Serve the ISOCUBE logo for react-native-web's Image#source.
import logoUrl from '../../assets/isocube_logo.png?url'
;(globalThis as any).require = (p: string) => ({ uri: logoUrl })

import { createRoot } from 'react-dom/client'
import { Animated } from 'react-native'
import { GestureHandlerRootView } from './gestureShim'
import { States } from './states'

// react-native-web has no native animated module — strip the useNativeDriver
// flag so the app's native-driven Animated.Values (size slider) fall back to
// JS animation instead of throwing.
class ValueShim extends (Animated.Value as any) {
  constructor(v: number, _cfg?: any) {
    super(v)
  }
}
Animated.Value = ValueShim as any

const root = document.getElementById('root')!
createRoot(root).render(
  <GestureHandlerRootView style={{ height: '100%' }}>
    <States />
  </GestureHandlerRootView>,
)