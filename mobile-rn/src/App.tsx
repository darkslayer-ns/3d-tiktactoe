/**
 * Neon Cube app shell. GestureHandlerRootView is required by
 * react-native-gesture-handler (Board3D's orbit/pinch gestures).
 * SafeAreaProvider supplies per-component insets (RN's built-in SafeAreaView
 * is iOS-only — on Android edge-to-edge the bottom bar would slide under the
 * system navigation bar without it).
 */

import { GestureHandlerRootView } from 'react-native-gesture-handler'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import { StatusBar } from 'expo-status-bar'
import { useEffect } from 'react'
import { StyleSheet } from 'react-native'
import { Theme } from './ui/theme'
import { GameScreen } from './ui/screens/GameScreen'
import { playSfx } from './audio/SoundManager'

export default function App() {
  useEffect(() => {
    void playSfx('intro')
  }, [])

  return (
    <SafeAreaProvider>
      <GestureHandlerRootView style={styles.root}>
        <StatusBar style="light" />
        <GameScreen />
      </GestureHandlerRootView>
    </SafeAreaProvider>
  )
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Theme.bg,
  },
})