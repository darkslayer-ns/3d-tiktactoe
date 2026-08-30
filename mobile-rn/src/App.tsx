/**
 * Neon Cube app shell. GestureHandlerRootView is required by
 * react-native-gesture-handler (Board3D's orbit/pinch gestures).
 */

import { GestureHandlerRootView } from 'react-native-gesture-handler'
import { SafeAreaView, StyleSheet } from 'react-native'
import { StatusBar } from 'expo-status-bar'
import { Theme } from './ui/theme'
import { GameScreen } from './ui/screens/GameScreen'

export default function App() {
  return (
    <GestureHandlerRootView style={styles.root}>
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <GameScreen />
      </SafeAreaView>
    </GestureHandlerRootView>
  )
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Theme.bg,
  },
})